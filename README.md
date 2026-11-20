<img src="assets/icon.svg" width="64" height="64" alt="">

# sparkforge

A Rust inference engine for one target: **NVIDIA DGX Spark (GB10, compute capability 12.1, `sm_121a`)**.
You give it a model's `config.json`. It works out every tensor shape, picks kernel configurations for
those exact shapes under GB10's limits, and generates CUDA source with all dimensions fixed at compile time.
That source is compiled with NVRTC into a cubin that is cached under a hash of the model, plan, and arch.

```
config.json ─▶ ModelSpec ─▶ Plan ─────────────▶ Codegen ─────────▶ NVRTC (sm_121a) ─▶ cubin cache ─▶ Runtime
               (shapes)     (tiles, grids,      (constexpr N/K/     (--features cuda)    build/<hash>/
                            splits, budget)      head_dim/GQA…)
```

## Why specialize per model

General engines ship kernels that read shapes at runtime and dispatch to a fixed set of tile configs
that were mostly tuned for datacenter GPUs. On GB10 that leaves performance on the table, for three reasons:

1. **Different tensor-core ISA.** SM121 has warp-level `mma.sync` (BF16, FP8, and block-scaled FP4 via
   `kind::mxf4nvf4`). It has **no** `tcgen05`, TMEM, or WGMMA, so FlashAttention-4 and CUTLASS's flagship
   Blackwell kernels do not apply ([SM121 hardware facts](https://github.com/sovereignbrah/gb10-inference-wiki/blob/main/17-sm121-hardware-facts.md),
   [sm121-kernels architecture guide](https://github.com/blake-snc/sm121-kernels/blob/master/docs/sm120_architecture_guide.md)).
2. **Small on-chip budget.** 48 SMs and about 99 KB of shared memory per block (101,376 B opt-in), 24 MB L2
   ([GB10 core facts](https://github.com/sovereignbrah/gb10-inference-wiki/blob/main/science/core.md)).
   Tile shapes that fit on H100/B200 don't fit here.
3. **Decode is bandwidth-bound.** 273 GB/s nominal LPDDR5X ([NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)),
   about 200–247 GB/s measured ([GPUwerk measurements](https://gpuwerk.com/dgx-spark/guides/dgx-spark-memory-bandwidth-measured/)).
   Every decode kernel has to keep DRAM busy across all 48 SMs, and the right grid depends on N, K, and GQA ratio.

When `N`, `K`, `head_dim`, `GQA group`, and `hidden` are `constexpr`, NVRTC can fully unroll loops, keep
per-lane state in registers, and remove bounds checks. The planner makes the launch-shape decisions
that would otherwise be runtime heuristics.

## What exists today

| Module | Role | Status |
|---|---|---|
| `config.rs` | Parses HF configs (Llama/Qwen/Mistral-style dense, Qwen3-MoE/Mixtral/DeepSeek-style MoE aliases, quantization_config) into a `ModelSpec`. | Working |
| `target.rs` | GB10 hardware model: SMs, smem, regs, L2, effective bandwidth, usable unified memory. | Working |
| `plan.rs` | Derives every linear (fused QKV, fused gate/up, experts, lm_head), picks decode GEMV grids, prefill GEMM tiles (smem/regs/wave-quantization scoring), flash-decoding split count, RMSNorm threading, memory budget, and decode roofline. | Working |
| `codegen.rs` | Emits `model.cu` + `gemm_tiles.cuh` with shapes baked in, plus `manifest.json` (entry, grid, block, dyn smem, args). | Working |
| `runtime.rs` | NVRTC compile for `sm_121a`, checks cc 12.1, loads every generated entry point via `cudarc`. | Builds; needs a Spark to run |

Generated kernels (all verified to compile for `sm_121a` with NVRTC 13, no register spills):

- `rmsnorm` — one CTA per row, 16-byte vector loads, thread count chosen so `hidden = threads × iters × 8` exactly.
- `rope_kv_append` — NeoX RoPE on q/k in the fused QKV buffer, fused with K/V cache append.
- `attn_decode_split` + `attn_decode_combine` — GQA-aware flash-decoding. One CTA per (kv_head, split) serves
  all query heads in the group, so each K/V row is read once per split. Split count is chosen so
  `kv_heads × splits ≥ 2 × 48 SMs`.
- `silu_mul` — fused SwiGLU on the fused gate/up output.
- `gemv_<linear>` — one kernel per projection with its own N, K, rows-per-warp, and grid; weights streamed with evict-first loads.

Planned (configs chosen, bodies TODO): prefill `mma.sync` GEMMs per linear, grouped MoE GEMM, W4A16 and NVFP4 decode GEMVs.

## Usage

```bash
cargo build --release
./target/release/sparkforge plan configs/llama-3.1-8b.json
./target/release/sparkforge plan configs/qwen3-30b-a3b.json --format nvfp4 --ctx 65536
./target/release/sparkforge emit /path/to/model/config.json --out build      # -> build/<hash>/{model.cu,gemm_tiles.cuh,plan.json,manifest.json}

# On a Spark:
cargo run --release --features cuda -- load configs/llama-3.1-8b.json --cuda-include /usr/local/cuda/include
cargo test
```

Example plan output (Llama 3.1 8B, BF16): weights 15.0 GiB, decode needs about 19.3 GB per token, so the
ceiling is about 12 tok/s at 235 GB/s. Attention runs 8 kv-heads × 12 splits = 96 CTAs. Prefill tiles land on
128×128×32, 3 stages, 48 KB smem. Qwen3-30B-A3B in NVFP4: about 5.4 GB per token, ceiling about 43 tok/s,
with grouped-GEMM tiles sized for about 128 tokens per expert at a 2,048-token prompt.

## Roadmap

Ordered by what moves tokens/s the most on GB10.

1. **Weight loader + CPU reference.** safetensors via `memmap2`. Since memory is unified, use `cudaHostRegister` or
   managed memory and skip copies. Write a scalar Rust reference forward pass and diff every generated kernel against it.
2. **End-to-end BF16 decode** for one dense model using the existing kernels and a CUDA Graph per decode step
   (the launch count is fixed, so graph capture is simple). Measure against the roofline the planner prints.
3. **Quantized decode GEMVs.** W4A16 (AWQ/GPTQ layouts) and NVFP4 (e2m1 + e4m3 per-16 scales) with dequant in
   registers. This is the biggest decode win: 4× fewer bytes than BF16.
4. **Prefill GEMMs** from `gemm_tiles.cuh`: `ldmatrix` + `mma.sync.m16n8k16` BF16 with `cp.async`/TMA multistage
   pipelines, then `kind::mxf4nvf4.block_scale m16n8k64` for true W4A4 NVFP4 (needs activation quantization, so the
   FP4 tensor cores are actually used). Reference implementations: [sm121-kernels](https://github.com/blake-snc/sm121-kernels),
   [nvfp4bench design notes](https://github.com/secYOUre/nvfp4bench/blob/main/DESIGN.md).
5. **Prefill attention**: FA2-style with `mma.sync`, tiles sized to the 99 KB budget for the model's `head_dim`.
6. **Autotune loop.** The planner proposes the top-k configs per shape; the runtime benchmarks them on the actual
   Spark once and writes the winner into the cache (the cache key already covers model + plan + arch).
7. **Paged KV cache + batching.** Batch size becomes another compile-time parameter (B = 1, 2, 4, 8 variants).
8. **Multi-Spark.** Tensor parallel over ConnectX-7 RoCE for models that don't fit. The planner already flags this.
   Note there's no GPUDirect RDMA on GB10, so NCCL stages through host memory ([GB10 core facts](https://github.com/sovereignbrah/gb10-inference-wiki/blob/main/science/core.md)).

## Design rules

- Every number the planner uses comes from the config or `target.rs`. No per-model hand-tuning in code.
- Plan against measured bandwidth (235 GB/s), not the 273 GB/s spec.
- Use `sm_121a`; the `a` suffix is required for block-scaled FP4 MMA.
- No datacenter-Blackwell assumptions: no `tcgen05`, TMEM, WGMMA, clusters larger than 1, or TMA multicast.
- Keep the GPU budget below `MemAvailable` (about 114 GiB). Pinned allocations bypass the OOM killer, and running
  out freezes the node instead of returning a CUDA error.

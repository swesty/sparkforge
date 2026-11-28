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
| `config.rs` | Parses HF configs (Llama/Qwen/Mistral-style dense, Qwen3-MoE/Mixtral MoE aliases, `quantization_config`, Llama 3 `rope_scaling`) into a `ModelSpec`. Anything it cannot plan correctly is an error, not a silent default (see below). | Working |
| `target.rs` | GB10 hardware model: SMs, smem, regs, L2, effective bandwidth, usable unified memory. | Working |
| `plan.rs` | Derives every linear (fused QKV, fused gate/up, experts, lm_head), picks decode GEMV grids, prefill GEMM tiles (smem/regs/wave-quantization scoring), prefill attention tiles, flash-decoding split count, RMSNorm threading, memory budget, and decode roofline. | Working |
| `codegen.rs` | Emits `model.cu` + `gemm_tiles.cuh` with shapes baked in, plus `manifest.json` (entry, grid, block, dyn smem, args). | Working |
| `runtime.rs` | NVRTC compile for `sm_121a`, checks cc 12.1, loads every generated entry point via `cudarc`, opts kernels above 48 KB into larger dynamic smem. | Verified on GB10 |
| `verify.rs` | Runs every generated kernel on random bf16 inputs, diffs against an f64 CPU reference, and optionally times it with weights rotated out of L2. `verify-weights` does the same on a real checkpoint. | Verified on GB10 |
| `weights.rs`, `device_weights.rs` | Opens HF safetensors checkpoints (single file or sharded), validates every tensor against the plan, and copies each linear into GPU memory with q\|k\|v and gate\|up packed, behind a `MemAvailable` guard. Dense BF16 Llama-family only. | Verified on GB10 |
| `reference.rs` | Scalar CPU reference for one dense decode step in f64, rounding to bf16 where HF transformers does. It reads random weights or streams a checkpoint. | Tested in CI |
| `forward.rs` | The generated kernels wired into a dense decode step whose token and position live on the GPU, captured as one CUDA Graph per step, and a prefill that runs the prompt through the GEMMs and `attn_prefill` in one pass per chunk, then hands the token and position to the graph. `verify` diffs decode against the reference at every layer boundary and prefill against decode; `generate` runs both on real weights. | Verified on GB10 |
| `tokenizer.rs` | Token ids to text for a byte-level BPE `tokenizer.json` (decoding only). | Tested in CI |
| `inventory.rs` | Every tensor a Qwen3.8 Flash Next checkpoint must hold, with the dtype and shape of each module's format (BF16, MXFP8, NVFP4). `weights` checks all 301,593 tensors of the reference checkpoint against it, and `load-weights` copies them into GPU memory. | Checked on GB10 (host only) |

Generated kernels (all compile for `sm_121a` with NVRTC 13 and match the CPU reference; see "Measured on GB10"):

- `rmsnorm` — one CTA per row, 16-byte vector loads, thread count chosen so `hidden = threads × iters × 8` exactly.
- `rmsnorm`, `rope_kv_append`, `residual_add`, `silu_mul` and `embed_gather` take one row per `blockIdx` (y, or x for
  `rmsnorm`): one row in a decode step, one per prompt token in prefill, so a prompt row is computed exactly as a
  decode step computes it.
- `rope_kv_append` — NeoX RoPE on q/k in the fused QKV buffer, fused with K/V cache append. `inv_freq` is a
  host-computed `__constant__` table (HF's f32 formula, plus Llama 3 scaling); sin/cos use double `sincos`
  because fast-math `__sincosf` loses accuracy at large angles (error 0.55 at position 131071 before the fix).
- `attn_decode_split` + `attn_decode_combine` — GQA-aware flash-decoding. One CTA per (kv_head, split) serves
  all query heads in the group, so each K/V row is read once per split. Split count is chosen so
  `kv_heads × splits ≥ 2 × 48 SMs`.
- `silu_mul` — fused SwiGLU on the fused gate/up output.
- `residual_add` — in place, summed in f32.
- `embed_gather`, `argmax_advance` — the step's token lookup and greedy choice on the GPU; `argmax_advance` also
  records the token and advances the device-side position, so one captured graph replays every step.
- `gemv_<linear>` — one kernel per projection with its own N, K, rows-per-warp, and grid; weights streamed with evict-first loads.
- `gemv_<linear>` for NVFP4 weights — BF16 activations against 4-bit weights (W4A16), reading the weight loader's
  tile layout (FlashInfer's `GemvBf16Fp4Sm12x` design, re-derived for GB10): one `cvt.rn.f16x2.e2m1x2` per weight
  byte, f32 accumulation, no shared memory. K is split across blocks until about 20 warps per SM are busy, and the
  last split to finish each tile sums the partials in a fixed order in the same launch, so the result is bitwise
  reproducible. Routed experts get one launch for all active experts, with expert ids read from the GPU.

- `gemm_<linear>` — BF16 prefill GEMM (C = A · Wᵀ, f32 accumulation) with the prompt length as an argument: a
  `cp.async` pipeline into shared memory padded 16 bytes per row, `ldmatrix`, `mma.sync m16n8k16` on 64x64 or 64x32
  warp tiles, tiles walked down M when the activations fit in L2. `lm_head` runs once at prefill, for the last position,
  as the decode GEMV.
- `attn_prefill` — causal prefill attention, FlashAttention-2 style: each warp owns 16 query rows with its Q fragments in
  registers, K/V tiles stream from the decode KV cache through a `cp.async` pipeline that reuses the Q tile's shared
  memory, `mma.sync m16n8k16` for Q·Kᵀ and P·V (`ldmatrix.trans` for V), fp32 online softmax in exp2. Prompt rows start
  at any cache position, so long prompts run in chunks. Tiles (128 query rows × 64 keys, 2 stages, 70,144 B for
  `head_dim` 128) are planned for resident warps per SM, then K/V reuse. A masked key's probability is 0, but
  0 × NaN is NaN, so a tile holding keys some row doesn't attend checks its V values at the barrier it already
  has; the rare tile with one that isn't finite zeroes it and marks NaN only in the rows that attend it.

Planned (configs chosen, bodies TODO): grouped MoE prefill GEMM, NVFP4 and MXFP8 prefill GEMMs, W4A16 (AWQ/GPTQ) decode GEMVs.

## Usage

```bash
cargo build --release
./target/release/sparkforge plan configs/llama-3.1-8b.json
./target/release/sparkforge plan configs/qwen3-30b-a3b.json --format nvfp4 --ctx 65536
./target/release/sparkforge plan configs/qwen3.8-flash-next-nvfp4.json   # hybrid: 34 op classes, 7.34 GB/token (the two NVFP4 expert GEMVs are generated)
./target/release/sparkforge emit /path/to/model/config.json --out build      # -> build/<hash>/{model.cu,gemm_tiles.cuh,plan.json,manifest.json}

# On a Spark:
cargo run --release --features cuda -- load configs/llama-3.1-8b.json --cuda-include /usr/local/cuda/include
cargo run --release --features cuda -- verify configs/llama-3.1-8b.json --bench   # correctness + timing per kernel
./target/release/sparkforge weights /path/to/checkpoint          # validate a safetensors checkpoint (host only)
cargo run --release --features cuda -- verify-weights /path/to/checkpoint   # GEMVs on the real weights vs f64
cargo run --release --features cuda -- generate /path/to/llama-3.1-8b --tokens 128000,791,... \
    --steps 48 --check 3 --tokenizer tokenizer.json --bench-ctx 1024,4096,32768 \
    --bench-prefill 2048,8192   # prompt in one prefill pass, then greedy decode, one graph per step
cargo test
```

Example plan output (Llama 3.1 8B, BF16): weights 15.0 GiB, decode needs about 19.3 GB per token, so the
ceiling is about 12 tok/s at 235 GB/s. Attention runs 8 kv-heads × 12 splits = 96 CTAs. Prefill tiles land on
128×128×32, 3 stages, 48 KB smem. Qwen3-30B-A3B in NVFP4: about 5.4 GB per token, ceiling about 43 tok/s,
with grouped-GEMM tiles sized for about 128 tokens per expert at a 2,048-token prompt.

## Measured on GB10

One DGX Spark, GB10 (cc 12.1), CUDA 13.0, driver 580.178.04, 2026-11-20, with another vLLM engine resident on
the same GPU (idle during the run). `verify --bench`; times are isolated kernels, back-to-back launches,
no CUDA graph.

| Kernel (Llama 3.1 8B BF16) | Max error | µs | GB/s |
|---|---|---|---|
| `gemv_lm_head` (1.05 GB of weights, cleanest bandwidth number) | 3.8e-3 | 4286 | 245 |
| `gemv_gate_up_proj` | 3.8e-3 | 951 | 247 |
| `gemv_down_proj` | 3.6e-3 | 500 | 235 |
| `attn_decode` split+combine, seq 32768 | 3.7e-3 | 603 | 223 |
| `rmsnorm`, `rope_kv_append`, `silu_mul` | ≤3.8e-3 | 2–4 each | launch-bound |

- **Correctness:** every generated kernel is within 3.8e-3 of the f64 reference on both example models
  (tolerance 1/64 ≈ two bf16 ulps). RoPE is checked at positions up to `max_position_embeddings - 1`.
- **Speed:** the summed kernel times give 12.4 tok/s for Llama 3.1 8B BF16 at a 32K context (planner ceiling 12.2)
  and 24.7 tok/s for Qwen3-30B-A3B run in BF16 (ceiling 25.3). The BF16 GEMV is already at about 90% of the
  273 GB/s spec, so the remaining decode gains come from fewer bytes (NVFP4/W4A16) and fewer launches, not a
  faster BF16 GEMV. Run-to-run variation was about 2%.
- **ptxas (`-Xptxas -v`, nvcc 13.0):** `gemv_down_proj` at K=14336 spills 8 bytes; `attn_decode_split` with GQA
  group 8 uses exactly 128 registers, the `__launch_bounds__(256, 2)` cap, so group 16 would spill.
- Small MoE GEMVs are latency-bound (`gemv_router` 95 GB/s; each BF16 expert GEMV is a separate launch). The NVFP4
  expert GEMVs cover all top-k experts in one launch.

**End-to-end decode** (`generate`, Llama 3.1 8B BF16, spark-5, CUDA 13.0, 2026-11-26, the GPU otherwise idle; one CUDA
Graph per step, 32 replays timed at each context, 14.96 GiB loaded in 8 s):

| Context | Measured | Planner ceiling | Sum of isolated kernel times |
|---|---|---|---|
| 1K | 15.5 tok/s (64.7 ms) | 15.5 | 15.5 |
| 4K | 15.0 tok/s (66.5 ms) | 15.1 | 15.6 |
| 32K | 12.1 tok/s (82.4 ms) | 12.2 | 12.2 |

The first 24 steps' logits match the CPU reference on the real weights (worst 1.37e-2, tolerance 1/64), and the greedy
tokens match its argmax in 24 of 24. The output is coherent text.

**Prefill** (`generate`, Llama 3.1 8B BF16, spark-5, CUDA 13.0.3, 2026-11-26, the GPU otherwise idle; chunks of 2048
tokens, mean of 3 after a warm-up, from an empty cache to the first token):

| Prompt | Time to first token | Prefill rate |
|---|---|---|
| 20 tokens | 93.9 ms | 213 tok/s |
| 128 tokens | 99.8 ms | 1,283 tok/s |
| 2048 tokens | 490.5 ms | 4,176 tok/s |
| 8192 tokens | 2,122.9 ms | 3,859 tok/s |

Before prefill, `generate` ran the prompt as decode steps at about 65 ms per token (about 133 s for 2048 tokens). A short
prompt costs about one pass over the 15 GB of weights. The prefill's logits match the CPU reference on the real weights
(8.5e-3), and so do the next three steps'.

`attn_prefill` (`verify --bench`, 32 heads, 8 KV heads, `head_dim` 128): 67.7–68.5 TFLOP/s for a 2048-token prompt and
78.1–79.6 TFLOP/s at 8192, within 3.68e-3 of f64 causal attention (scaled by the output's largest value, since P is
rounded to bf16 for the P·V product). The NaN guard (harden-decode-prefill-runtime) costs about 2–3%: the same
binary without it measured 68.7–70.9 and 80.3–81.7 in runs alternated with it on spark-5 (2026-11-28); a first
version that checked every V fragment in the product loop cost 25%. Tile sweep at 2048 / 8192 tokens (TFLOP/s),
measured before the guard:

| Query rows × keys, stages | 2048 | 8192 |
|---|---|---|
| 128 × 64, 2 (planned) | 69.4 | 80.3 |
| 192 × 64, 2 | 67.3 | 85.3 |
| 192 × 32, 2 | 65.8 | 81.6 |
| 64 × 32, 2 | 65.5 | 73.9 |
| 128 × 32, 2 | 64.2 | 75.8 |
| 128 × 32, 3 | 61.9 | 72.5 |
| 64 × 64, 2 | 50.9 | 55.6 |
| 64 × 32, 3 | 41.9 | 45.0 |

**Prefill GEMMs** (`verify --bench`, spark-5, CUDA 13.0, 2026-11-26, with a vLLM engine resident and idle; M = 2048):

| Llama 3.1 8B | Tile | TFLOP/s | cuBLAS (PyTorch 2.13) |
|---|---|---|---|
| `gemm_qkv_proj` 2048x6144x4096 | 128x256x32, 3 stages | 73.7–74.4 | 100.8 |
| `gemm_o_proj` 2048x4096x4096 | 128x256x32, 3 stages | 66.3–69.5 | 89.4 |
| `gemm_gate_up_proj` 2048x28672x4096 | 128x256x32, 3 stages | 75.0–75.1 | 81.3 |
| `gemm_down_proj` 2048x4096x14336 | 128x256x32, 3 stages | 65.6–66.0 | 94.1 |

Every GEMM is within 3.82e-3 of an f64 reference, at M = 2048 and at a ragged 2011. The GEMMs of a 2048-token prompt take
about 403 ms (about 5,080 tok/s); with attention, the isolated kernel times sum to about 417 ms (norms, RoPE and
activations not included). Walking M down each weight tile
(the activations fit in L2) measured 66–75 TFLOP/s, L2 groups of N tiles 64–72, and a plain N-major order 20–50.

| NVFP4 GEMV (spark-5, CUDA 13.0, 2026-11-25) | Bytes per launch | Max error | µs | GB/s |
|---|---|---|---|---|
| Flash Next `gemv_expert_gate_up` (10 experts) | 18.4 MB | 3.7e-3 | 83.4 | 220-221 |
| Flash Next `gemv_expert_down` (10 experts) | 9.2 MB | 3.8e-3 | 44.1 | 209-210 |
| Qwen3-30B-A3B `gemv_expert_gate_up` (8 experts) | 14.2 MB | 3.7e-3 | 58.9 | 240-241 |
| Qwen3-30B-A3B `gemv_expert_down` (8 experts) | 7.1 MB | 3.8e-3 | 30.4 | 231-233 |

  Expert GEMVs are timed so that no expert is reread within 3x the L2; random expert draws from one table read
  up to 10% faster from L2 reuse. With 20 or 40 experts per launch the gate/up GEMV reads 228-234 GB/s, so about
  3% of the 10-expert time is per-launch cost: the GPU-side drain and ramp at each kernel boundary, which a
  CUDA graph does not remove (220-223 GB/s captured) and PDL should overlap. The Llama 3.1 8B BF16 `gemv_lm_head` measured 242-246 GB/s.
  Qwen3-30B-A3B in NVFP4 sums to 39.4 tok/s (ceiling 43.5), with its BF16 `lm_head` 2.6 ms of the 25 ms.

## Not supported yet (rejected or flagged, never silently wrong)

- Rejected at parse time: `rope_scaling` other than `llama3` (yarn, dynamic, linear, longrope); MLA attention
  (`kv_lora_rank`, DeepSeek V2/V3); GPTQ/AWQ with per-channel scales (`group_size: -1`); modelopt quant
  algorithms other than NVFP4 and MIXED_PRECISION; any other `quant_method`; hybrid linear-attention models other
  than `qwen4_exp`; partial rotary on dense models. `--format` overrides the quantization check.
- Mostly planned: Qwen3.8 Flash Next (`qwen4_exp`). `plan` derives its op classes and byte budget. `emit`
  generates its two NVFP4 routed-expert GEMVs and lists every other op as planned, and `verify` checks those two;
  `verify-weights` refuses it. `weights` checks its checkpoint tensor by tensor, and `load-weights` copies it into
  GPU memory.
- Flagged in `plan` notes, output wrong until implemented: Qwen3 per-head q/k RMSNorm (`q_norm`/`k_norm`),
  q/k/v projection bias (Qwen2, `attention_bias: true`).
- Also ignored today: sliding-window attention (Mistral v0.1) and DeepSeek-style dense-first layers.

## Roadmap

Ordered by what moves tokens/s the most on GB10.

1. **Weight loader + CPU reference.** The loader is done (safetensors, validated, copied into GPU memory: Llama 3.1 8B
   loads in about 10 s on a Spark). Reading the file mapping in place through ATS was measured and rejected: about
   0.3 GB/s on first touch and about 35% below device memory warm (`openspec/changes/add-weight-loading/design.md`).
   Next: a scalar Rust reference forward pass to diff every layer against.
2. **End-to-end BF16 decode** for one dense model using the existing kernels and a CUDA Graph per decode step
   (the launch count is fixed, so graph capture is simple). Measure against the roofline the planner prints.
3. **Quantized decode GEMVs.** NVFP4 (e2m1 + e4m3 per-16 scales) is done, dense and grouped. Next: W4A16
   (AWQ/GPTQ layouts) and MXFP8 with dequant in registers.
4. **Prefill GEMMs** from `gemm_tiles.cuh`: `ldmatrix` + `mma.sync.m16n8k16` BF16 with `cp.async`/TMA multistage
   pipelines, then `kind::mxf4nvf4.block_scale m16n8k64` for true W4A4 NVFP4 (needs activation quantization, so the
   FP4 tensor cores are actually used). Reference implementations: [sm121-kernels](https://github.com/blake-snc/sm121-kernels),
   [nvfp4bench design notes](https://github.com/secYOUre/nvfp4bench/blob/main/DESIGN.md).
5. **Prefill attention**: dense causal FA2-style `attn_prefill` is done, and `generate` prefills through it and the GEMMs.
   Next: QSA sparse prefill attention for Qwen3.8 Flash Next (SHA-757), which reuses its core.
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

# Proposal

## Why

Every kernel check so far runs on random weights; sparkforge cannot read a real checkpoint, so
end-to-end decode (and any real accuracy claim) is blocked. GB10 memory is unified LPDDR5X and the
GPU translates through the CPU's page tables (`nvidia-smi` reports Addressing Mode ATS on our
GB10), so a memory-mapped safetensors file can be read by kernels in place.

## What Changes

- New `weight-loading` capability: open an HF safetensors checkpoint (single file or sharded with
  `model.safetensors.index.json`), memory-map it, and validate every tensor the plan needs (name,
  dtype, shape, byte range) before any GPU work.
- Bind each planned linear to GPU-readable memory in one of two modes, selectable per run:
  - **mapped**: kernels read the mmap address directly (no copy, no pinning); requires the device to
    report pageable memory access.
  - **copied**: tensors are copied into GPU allocations.
  Fused `qkv_proj` and `gate_up_proj` are packed (one copy) in both modes, because the checkpoint
  stores q/k/v and gate/up as separate tensors.
- A memory guard refuses any load whose copied bytes would leave less than 4 GiB of
  `MemAvailable`, because pinned and device allocations on GB10 bypass the OOM killer.
- `sparkforge verify-weights <checkpoint>` runs the existing GEMV kernels on loaded weights against
  an f64 reference, and reports load time, mapped/copied bytes and GB/s per mode.
- Scope: dense BF16 Llama-family checkpoints (Llama, Mistral, Qwen2/Qwen3 dense naming). MoE and
  non-BF16 checkpoints are rejected with a clear error until their kernels exist.

## Expected effect on GB10

No change to decode tok/s yet (nothing consumes real weights end to end). Measured instead, on one
GB10 with Llama 3.2 1B and Llama 3.1 8B (BF16): load time, resident memory, and GEMV GB/s from
mapped versus copied memory. The mapped mode is only worth defaulting to if its GEMV bandwidth is
within a few percent of copied memory (245-251 GB/s measured for the BF16 GEMV on device memory).

## Non-goals

- Quantized (NVFP4, W4A16) or MoE checkpoints; their kernels are separate tickets.
- Kernels that read q/k/v or gate/up from separate pointers (which would make fused linears
  zero-copy too).
- Running a full decode step; that is the end-to-end decode ticket.

## Capabilities

### New Capabilities
- `weight-loading`: reading, validating and GPU-binding HF safetensors checkpoints for a plan.

### Modified Capabilities

(none; `verify-weights` is specified inside `weight-loading`)

## Impact

- New dependency `memmap2`. New host module for checkpoint parsing and binding; new CUDA-feature
  module for GPU binding; new `verify-weights` CLI subcommand.
- spark-5 test data: ungated mirrors `unsloth/Llama-3.2-1B-Instruct@5a8abab` and
  `unsloth/Meta-Llama-3.1-8B-Instruct@a285619`.

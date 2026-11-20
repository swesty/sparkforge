# Proposal

## Why

Every kernel check so far runs on random weights; sparkforge cannot read a real checkpoint, so
end-to-end decode (and any real accuracy claim) is blocked. GB10 memory is unified LPDDR5X and the
GPU translates through the CPU's page tables (`nvidia-smi` reports Addressing Mode ATS on our
GB10), so whether kernels can read a memory-mapped safetensors file in place is worth measuring.

## What Changes

- New `weight-loading` capability: open an HF safetensors checkpoint (single file or sharded with
  `model.safetensors.index.json`), validate the file structure (index-to-shard mapping, exact
  coverage of each data section, overflow-checked offsets) and every tensor the plan needs (name,
  dtype, shape) before any GPU work. All reads are fallible positioned reads.
- Copy each planned linear into GPU memory, packing fused `qkv_proj` (q|k|v rows) and
  `gate_up_proj` (gate|up rows) in kernel order. Reading the file mapping in place through ATS was
  prototyped and measured, and rejected (see design.md, decision 2).
- A memory guard refuses any load whose copied bytes would leave less than 4 GiB of
  `MemAvailable`, because pinned and device allocations on GB10 bypass the OOM killer.
- `sparkforge verify-weights <checkpoint>` runs the existing GEMV kernels on loaded weights against
  an f64 reference, and reports load time, bytes copied and GB/s per linear.
- Scope: dense BF16 Llama-family checkpoints (Llama, Mistral, Qwen2/Qwen3 dense naming). MoE and
  non-BF16 checkpoints are rejected with a clear error until their kernels exist.

## Expected effect on GB10

No change to decode tok/s yet (nothing consumes real weights end to end). Measured on one GB10
(spark-5, CUDA 13.0, driver 580.178.04): Llama 3.1 8B loads 13.98 GiB in 10.3-10.8 s and Llama 3.2
1B 2.30 GiB in 1.84 s; GEMVs on the loaded weights match the f64 reference (3.5e-3 to 3.8e-3) and
run at 225-252 GB/s, the same as the random-weight `verify --bench` numbers.

## Non-goals

- Quantized (NVFP4, W4A16) or MoE checkpoints; their kernels are separate tickets.
- Zero-copy weights: reading file pages through ATS was measured slower (design.md, decision 2).
- Running a full decode step; that is the end-to-end decode ticket.

## Capabilities

### New Capabilities
- `weight-loading`: reading, validating and GPU-binding HF safetensors checkpoints for a plan.

### Modified Capabilities

(none; `verify-weights` is specified inside `weight-loading`)

## Impact

- No new dependency (an earlier revision used `memmap2`; see design.md decision 1). New host module
  for checkpoint parsing and binding; new CUDA-feature module for GPU binding; new `weights` and
  `verify-weights` CLI subcommands.
- spark-5 test data: ungated mirrors `unsloth/Llama-3.2-1B-Instruct@5a8abab` and
  `unsloth/Meta-Llama-3.1-8B-Instruct@a285619`.

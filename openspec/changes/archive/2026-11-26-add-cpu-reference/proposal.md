# Proposal

## Why

`sparkforge verify` proves each kernel matches a reference in isolation, but not that the kernels are wired together correctly: layouts, residual adds, KV indexing and the GQA mapping (SHA-746). End-to-end decode needs that before any tok/s number means anything.

## What Changes

- **CPU reference** (`src/reference.rs`, no CUDA, tested in CI): one decode step of a dense Llama-shaped model in f64, rounding to bf16 where HF transformers does in bf16. RMSNorm rounds `x * r` before the weight multiply; each op's bf16 output is rounded; attention probabilities and the logits are not.
- **GPU decode step** (`src/forward.rs`): the generated kernels wired in the reference's order, one stream, no CUDA Graph. SHA-755 builds the runtime on it.
- **Kernels:**
  - A new `residual_add` (in place, f32 sum, bf16 out).
  - `rmsnorm` now rounds `x * r` before the weight, as HF does.
- **`verify`:**
  - Checks `residual_add`, and checks `rmsnorm` against the new rounding.
  - Runs a 2-layer decode step of random weights in two cases: four steps from an empty cache, and two steps after a random 1000-position prefix. It reports one row per layer boundary and the logits, so error growth per layer is visible.
- **Layer-boundary metric:** max|gpu - ref| / max|ref|. The per-element metric fails a correct step: both sides round every op to bf16, and a residual add that cancels to near zero turns a 1-ulp difference into a large relative error (measured: 4.4e-2 at layer 0 to 4.2e-1 at the logits). Against the tensor's scale, the same run measures 2.9e-3 to 6.6e-3. Three injected wiring bugs measure 0.6 to 3.9.

- **codex-review cycle 1 fixes:**
  - `fp:edf3261e6c96`: `reference::check_shapes` checks every weight and cache length, and K and V layer counts, before `DeviceModel::upload` copies anything.
  - `fp:4159fada9500`: the `rmsnorm` check also requires the kernel to match the double rounding on at least 90% of the elements where the two orders differ. The new kernel matches 4138 of 4138; the old expression matches 0 and fails.
  - `fp:688a8d36f2cc`: `f64_to_bf16` rounds f64 results once.
  - `fp:cf7c641e5701`: the KV-slot test checks each slot and that step 1 preserves slot 0.

**Expected GB10 effect:** none on decode speed (`residual_add` measures 2.1 us; it isn't in the per-token estimate yet). Llama 3.1 8B's estimate stays 12.1 tok/s.

## Non-goals

- Real weights, a long-running KV cache, CUDA Graphs and tok/s: SHA-755.
- MoE, hybrid, q/k-norm and q/k/v-bias models: the check prints why it is skipped.

## Capabilities

### Modified Capabilities
- `gpu-verification`: `residual_add` is checked; the decode-step check and the reference's rounding are added; the worst per-kernel error is restated as 3.9e-3 (measured 3.82e-3).
- `kernel-codegen`: `model.cu` gains `residual_add`; `rmsnorm` rounds before the weight.

## Impact

- New: `src/reference.rs`, `src/forward.rs`, `tests/reference.rs`.
- Changed: `src/codegen.rs`, `src/verify.rs`, `src/main.rs`, `src/lib.rs`.

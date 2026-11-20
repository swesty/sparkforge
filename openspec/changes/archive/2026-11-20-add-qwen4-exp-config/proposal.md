# Proposal

## Why

The Qwen3.8 Flash Next engine (`add-inference-engine`, milestone M0) needs sparkforge to understand the checkpoint before any kernel or loader work. Today `src/config.rs` rejects it at parse time:
- it bails on ModelOpt `MIXED_PRECISION`;
- it has no `text_config` nesting;
- it has no notion of the hybrid layout: Gated DeltaNet and sparse-attention layers, the HyperConnection residual, the n-gram embedding (PLE), and the MTP layer.

Every later Flash Next ticket needs the layer schedule, the per-module format map, and a byte budget to measure against.

## What Changes

- `model-config`:
  - Multimodal configs are read from `text_config`.
  - The `qwen4_exp` hybrid layout is parsed into a per-layer schedule plus its Gated DeltaNet, sparse-attention, HyperConnection, PLE, partial interleaved mRoPE and MTP parameters.
  - ModelOpt `MIXED_PRECISION` maps each listed module to NVFP4, weight-only NVFP4 or MXFP8, with everything else BF16.
  - `rope_parameters` is read.
  - Partial rotary on a dense model is rejected.
  - The shared-expert width honors `shared_expert_intermediate_size`.
- `kernel-planning`:
  - A hybrid model is planned as one op class per distinct kernel: 34 for Flash Next, against about 900 per-layer instances.
  - Each op class has shapes, format, instances per token and bytes per token.
  - A decode budget counts per-module formats, recurrent state and the sparse-attention context terms.
  - Format-map entries the plan never consumes are reported.
- `kernel-codegen`: hybrid plans list every op class as planned and generate no CUDA source.
- `gpu-verification`: `verify` refuses a model with no generated kernels.
- `weight-loading`: hybrid and mixed-precision checkpoints are refused until the mixed-format loader exists.

**Expected GB10 effect.** No kernel changes, so there is no runtime effect. `sparkforge plan configs/qwen3.8-flash-next-nvfp4.json` reports, matching the checkpoint's safetensors headers within 1%:
- 7.338 GB of context-free decode reads per token, which is a 33 tok/s ceiling at 245 GB/s (31.9 at the planner's 235);
- 7.375 GB per token at a 32K context;
- 105.4 GB of resident weights.

Measured by `cargo test` (`tests/flash_next.rs`) against the header-derived per-group figures.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `model-config`: text_config nesting, the qwen4_exp hybrid schedule, MIXED_PRECISION format maps, rope_parameters, dense partial-rotary rejection, shared-expert width.
- `kernel-planning`: hybrid op classes, hybrid decode budget, unused format-map reporting.
- `kernel-codegen`: hybrid plans generate nothing.
- `gpu-verification`: verify refuses models without generated kernels.
- `weight-loading`: hybrid and mixed-precision checkpoints are refused.

## Impact

- **Code:** `src/config.rs`, `src/plan.rs`, `src/codegen.rs`, `src/verify.rs`, `src/weights.rs` and `src/main.rs`. The new config is `configs/qwen3.8-flash-next-nvfp4.json`, and the new tests are `tests/flash_next.rs`.
- **Unchanged:** Llama 3.1 8B and Qwen3-30B-A3B plans and generated kernels. Their cache keys and plan JSON are identical to `main`.

## Non-goals

- Generating any Flash Next kernel (milestone M2), loading its weights (SHA-767), or prefill tiles for MXFP8/NVFP4 (M7).
- Planning MTP speculative steps. The MTP weights count toward resident memory only.
- Hybrid models other than `qwen4_exp`, such as Qwen3-Next. These are rejected.

# Proposal

## Why

sparkforge generates and verifies individual decode kernels for GB10, but it cannot run a model. The goal is a batch-1 engine that runs Qwen3.8-Flash-Next-NVFP4 on one DGX Spark faster than the fleet's vLLM engine does today. That engine measured ~50 tok/s decode with MTP k=4 and ~3,060 tok/s prefill on spark-5.

The research behind this change (Linear plan document "Qwen3.8 Flash Next engine plan (2026-11-20)") found that batch-1 decode of this model is bound by bytes read per token: 7.34 GB/token at checkpoint precision. So the engine must:
- be byte-exact about what it reads;
- use the checkpoint's MTP head for speculative decoding;
- take byte reductions the fleet engine cannot;
- be measured against a stated ceiling at every milestone.

This contract should exist before the kernels and runtime are built.

## What Changes

- New capability `inference-engine`. It covers:
  - a batch-1 runtime that executes a planned model as a fixed step program, captured as one CUDA Graph per step shape;
  - a memory budget enforced before any allocation;
  - sampling;
  - MTP speculative decoding that is exact under greedy decoding;
  - recurrent-state rollback;
  - golden-activation and logprob correctness gates;
  - reporting of tok/s against the planner's byte ceiling.
- A `sparkforge generate` command. It runs a prompt through the engine and prints decode tok/s, time to first token, and the ceiling for the chosen configuration.
- Numerics-changing byte reductions (MXFP8 HyperConnection weights, a truncated-vocabulary draft head) become opt-in engine options. Each is gated by the logprob check.
- Expected GB10 effect, measured by `sparkforge generate` on spark-5 with the fleet engine's A/B prompts (4 prompts, 256 tokens, temperature 0) against the same prompts on the fleet engine:
  - **Decode without speculation:** at least 28 tok/s. That is 85% of the 33 tok/s byte ceiling at 245 GB/s.
  - **Decode with MTP k=4:** at least 50 tok/s at checkpoint precision, matching the fleet. The target is 58–72 tok/s with the opt-in byte reductions.
  - **Prefill:** 16K-token time to first token below the fleet's 4.89 s.
  - **Launches:** about 330–370 kernel launches per decode step after fusion, down from about 900 for a per-op engine, counted from the step program.

## Capabilities

### New Capabilities
- `inference-engine`: batch-1 execution of a planned model on one GB10. It covers the step program and CUDA Graph capture, the memory budget, sampling, speculative decoding with recurrent-state rollback, correctness gates against reference activations and logprobs, and performance reporting against the byte ceiling.

### Modified Capabilities
None in this change. Each of the following lands as its own change against its own capability, before or alongside the engine milestone that needs it:
- the Qwen3.8 Flash Next config (`model-config`);
- the mixed NVFP4/MXFP8/BF16 loader (`weight-loading`, arriving with PR swesty/sparkforge#4);
- the new kernel families (`kernel-planning`, `kernel-codegen`): MXFP8 and grouped NVFP4 GEMVs, HyperConnection, Gated DeltaNet, sparse attention and its indexer, PLE, and MoE routing;
- the HTTP front end (a later `serving-api` capability).

## Impact

- **Code:** a new `src/engine/` module (step program, arenas, graph capture, sampler, speculative decoding), a `generate` subcommand, and the use of `plan.rs` and `codegen.rs` output through the existing kernel manifest.
- **Dependencies:** HF `tokenizers` (tokenization) and `minijinja` (chat template) when the CLI takes chat input.
- **Hardware:** full-model runs need about 76 GiB (PLE table on disk) to about 104 GiB (resident) of one GB10. On spark-5, each full-model run stops the experimental Flash Next serving lane, with the user's approval before every stop, and restarts it afterwards. Kernel-level work runs beside the fleet engine.

## Non-goals

- Batch sizes above 1, continuous batching, paged KV cache and prefix caching. The engine serves one stream; prefix caching is a later change.
- Multi-Spark tensor parallelism. The model fits one GB10.
- An OpenAI-compatible HTTP server, tool-call and reasoning parsers, and guided JSON decoding. These belong to the later `serving-api` change.
- A whole-model persistent megakernel. It is estimated at 3–5% over graph plus PDL and needs a grid-barrier measurement first.
- Models other than Qwen3.8 Flash Next. Llama 3.1 8B BF16 is used only as the runtime proving ground (SHA-755).
- Bit-identical output to the fleet engine. Agreement is judged within the fleet's own logprob noise band.

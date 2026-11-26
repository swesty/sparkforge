# Tasks

- [x] 1.1 `src/reference.rs`: dense decode step in f64 with HF bf16 rounding, random weights, KV cache, `require_dense`
- [x] 1.2 `tests/reference.rs`: GQA head mapping, attention weighting, RoPE at position 0, RMSNorm double rounding (checked to differ from a single rounding), KV slots, determinism, dense-only guard
- [x] 1.3 Kernels: `residual_add`; `rmsnorm` rounds `x * r` before the weight
- [x] 1.4 `src/forward.rs`: the generated kernels wired into one decode step
- [x] 1.5 `verify`: `residual_add` check, `rmsnorm` reference with the new rounding, 2-layer decode step (fresh and 1000-position prefix) with the scaled metric, skip line for other models
- [x] 1.6 GB10 (spark-5): Llama 3.1 8B and Qwen3-30B-A3B pass; three injected wiring bugs fail
- [x] 1.7 `openspec validate add-cpu-reference --strict`, archive, `verify.sh`

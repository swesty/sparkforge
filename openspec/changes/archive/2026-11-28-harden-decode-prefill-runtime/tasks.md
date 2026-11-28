# Tasks

- [x] 1.1 Runtime: `Kernels`/`PrefillKernels` bound to their model and context, refused by `Runtime::new`, `from_host` and `enable_prefill`; attention scratch from the compiled splits
- [x] 1.2 `cursor::Cursor`: stale after a step, traced step or prefill fails partway; refused until `set` (host tests)
- [x] 1.3 `prefill::Gemm` binding, tracked views in `gemm`, `unsafe fn gemm_raw`, context check
- [x] 1.4 `attn_prefill` NaN guard: `__syncthreads_or` in diagonal tiles, zero and record per dimension, NaN marking at the end; shared memory plus 4 B per head dimension; goldens regenerated
- [x] 1.5 `generate`: reference cache in admission and resident before release; span ceiling (`plan::decode_ceiling_over`); unknown-id warnings
- [x] 1.6 `CheckpointWeights::check` requires the step model's dimensions; `Decoder::decode` shows undefined ids; MoE estimate bounded by tokens × experts_per_tok
- [x] 1.7 `verify`: missing planned prefill kernels fail; positive-row misalignment probe; second-stream copy with event tracking; NaN-past-row cases; foreign kernels refused
- [x] 1.8 GB10 (spark-5): both models' `verify --bench`, Llama at `--prefill-tokens 1`, mutations (no guard, untracked launch) fail `verify`, attention timing alternated with `main`, Llama `generate` with `--check 4`, `--bench-ctx 1,1024`, `--bench-prefill 128,2048,8192`, and the 8192-token `--check 1` admission
- [x] 1.9 `openspec validate harden-decode-prefill-runtime --strict`, archive, `verify.sh`

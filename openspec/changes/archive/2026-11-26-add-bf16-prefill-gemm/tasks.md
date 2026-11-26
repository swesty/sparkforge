# Tasks

- [x] 1.1 Planner: padded shared memory, warp width per tile, register budget, intensity cap, `group_n`, no `lm_head` GEMM
- [x] 1.2 Codegen: `gemm_<linear>` with a `cp.async` pipeline, `ldmatrix`, `mma.sync m16n8k16`, runtime M; generated for BF16 linears
- [x] 1.3 `verify`: f64 checks at the planned and a ragged M, rows past M untouched; TFLOP/s; prefill estimate
- [x] 1.4 CI tests: no `lm_head` GEMM, padded shared memory and warp counts, the L2 order rule, generated vs planned by format, the estimate; goldens regenerated
- [x] 1.5 GB10 (spark-5): both models pass; tile sweep and raster comparison; cuBLAS baseline from PyTorch
- [x] 1.6 `openspec validate add-bf16-prefill-gemm --strict`, archive, `verify.sh`

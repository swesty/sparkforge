# Tasks

## 1. Checkpoint reading (host, no GPU)

- [x] 1.1 safetensors header parser and shard mmap with byte-range checks; unit tests on generated tiny checkpoints, including a truncated shard (`cargo test`)
- [x] 1.2 Index (`model.safetensors.index.json`) and single-file discovery; test with a two-shard generated checkpoint (`cargo test`)

## 2. Binding against the plan (host, no GPU)

- [x] 2.1 Expected tensor list per layer from `ModelSpec` (dense Llama-family names, tied embeddings, q/k norm, q/k/v bias); tests for missing, wrong shape, wrong dtype, tied, unexpected (`cargo test`)
- [x] 2.2 Reject MoE and non-BF16 checkpoints with a clear message (`cargo test`)

## 3. GPU binding (`--features cuda`)

- [x] 3.1 Mapped (ATS) prototype with 16-byte alignment fallback, measured on GB10 and then removed (design.md decision 2)
- [x] 3.2 Copy every linear into GPU memory, packing q|k|v and gate|up (`verify-weights` checks packed linears against the separate checkpoint tensors)
- [x] 3.3 MemAvailable guard before any allocation; unit test of the guard arithmetic (`cargo test`)
- [x] 3.4 Positioned reads for bulk loads (Llama 3.1 8B 40.5 s -> 10.3 s on spark-5)

## 3b. codex-review cycle 1

- [x] 3b.1 Overflow-checked offsets validated against the data section; header-length and shape overflow are errors (`cargo test`)
- [x] 3b.2 Index entries must match shard headers, both directions (`cargo test`: swapped and missing entries)
- [x] 3b.3 Non-empty tensors must tile each data section exactly (`cargo test`: aliased and gapped ranges, empty tensor allowed)
- [x] 3b.4 Drop mmap; fallible positioned reads with a shard-size check (`cargo test`: truncation after open is an error)
- [x] 3b.5 Bounded staging and streaming f64 reference; guard budgets both (`cargo test`; Llama 3.1 8B verify peaks at 225 MB host RSS on GB10)

## 4. Verification on GB10

- [x] 4.1 `verify-weights` subcommand: f64 reference GEMVs for first/last layer and lm_head, timing across all layers, load report
- [x] 4.2 `verify-weights` on Llama 3.2 1B on a GB10: all checks pass (3.5e-3 to 3.8e-3), 2.30 GiB in 1.84 s, 225-243 GB/s
- [x] 4.3 `verify-weights` on Llama 3.1 8B on a GB10: all checks pass (3.6e-3 to 3.8e-3), 13.98 GiB in 10.3-10.8 s, 235-252 GB/s, 11.3 GiB still available
- [x] 4.4 `sparkforge verify --bench` on a GB10 for both example configs: all kernels pass, unchanged
- [x] 4.5 Mode decision recorded in design.md (copy; mapped rejected)

## Workflow follow-up

- Archive the change after merge so `weight-loading` lands in `openspec/specs/`.

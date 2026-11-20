# Tasks

## 1. Checkpoint reading (host, no GPU)

- [ ] 1.1 safetensors header parser and shard mmap with byte-range checks; unit tests on generated tiny checkpoints, including a truncated shard (`cargo test`)
- [ ] 1.2 Index (`model.safetensors.index.json`) and single-file discovery; test with a two-shard generated checkpoint (`cargo test`)

## 2. Binding against the plan (host, no GPU)

- [ ] 2.1 Expected tensor list per layer from `ModelSpec` (dense Llama-family names, tied embeddings, q/k norm, q/k/v bias); tests for missing, wrong shape, wrong dtype, tied, unexpected (`cargo test`)
- [ ] 2.2 Reject MoE and non-BF16 checkpoints with a clear message (`cargo test`)

## 3. GPU binding (`--features cuda`)

- [ ] 3.1 Mapped mode: pageable-memory-access check, direct mmap pointers, 16-byte alignment fallback (`cargo check --features cuda`)
- [ ] 3.2 Copied mode and fused packing (q|k|v, gate|up) into GPU allocations
- [ ] 3.3 MemAvailable guard before any allocation; unit test of the guard arithmetic (`cargo test`)

## 4. Verification on GB10

- [ ] 4.1 `verify-weights` subcommand: f64 reference GEMVs for first/last layer and lm_head, timing across all layers, load report
- [ ] 4.2 Run `verify-weights --mode both` on Llama 3.2 1B on a GB10; record load time, bytes mapped/copied, GB/s per mode
- [ ] 4.3 Run `verify-weights` on Llama 3.1 8B on a GB10 within the memory guard; record the same
- [ ] 4.4 Run `sparkforge verify --bench` on a GB10 for both example configs to confirm no kernel changed
- [ ] 4.5 Choose the default mode from the 4.2/4.3 bandwidth numbers and record the decision in design.md

## Workflow follow-up

- Archive the change after merge so `weight-loading` lands in `openspec/specs/`.

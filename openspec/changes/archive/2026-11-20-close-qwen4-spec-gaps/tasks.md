# Tasks

- [x] 1.1 `n_shared_experts` defaults to 0. Verify: `cargo test` parses a 128-expert config without it and gets 0 shared experts.
- [x] 1.2 A nested `quantization_config` conflict errors only without an override, and a mixed declaration at either level survives an override. Verify: `cargo test`.
- [x] 1.3 Binding refuses an FP8 projection behind a config without `quantization_config`, before GPU work. Verify: `cargo test` (`tests/weights.rs`).

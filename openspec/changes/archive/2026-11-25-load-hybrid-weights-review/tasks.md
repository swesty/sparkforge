# Tasks

- [x] 1.1 Admission reserves `max(scratch, gather peak)`, with 8 bytes of row ID per gathered row. Test: a limit that fits the old sum but not the gather peak is refused, and the test fails with the old sum restored
- [x] 1.2 MTP budget counts both pre-FC norm vectors once. Test: the zero-, one- and two-layer test requires the extra 640 bytes, and fails with the term removed
- [x] 1.3 `Admission::confirm` returns the `MemAvailable` it checked; `load-weights` and `verify-weights` report it. Test: memory freed between admission and allocation shows up in the returned baseline
- [x] 1.4 `verify.sh` on macOS (`cargo test`, `cargo check --features cuda`)
- [x] 1.5 The CUDA build, `cargo test --release` and the printed admission line on spark-5 (lane up: refused before CUDA; 3,008.0 MiB gather peak at 1,048,576 tokens)

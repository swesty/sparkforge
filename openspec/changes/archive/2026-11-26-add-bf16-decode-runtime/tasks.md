# Tasks

- [x] 1.1 Kernels: position from device memory in `rope_kv_append` and `attn_decode_split`; `embed_gather`; `argmax_advance`
- [x] 1.2 `forward::Runtime` with private capacity, device token and position, graph capture on a non-default stream; `WeightPtrs`, `UploadedWeights`
- [x] 1.3 `device_weights::upload_dense` (linears, norms, embedding) and `weights::admit_runtime`
- [x] 1.4 CPU reference through a `Weights` trait (`DenseWeights`, `CheckpointWeights`); `decode_step` validates first and returns errors; `f64_to_bf16` subnormals
- [x] 1.5 `tokenizer::Decoder`; `sparkforge generate` with `--check`, `--tokenizer`, `--bench-ctx`
- [x] 1.6 `verify`: `embed_gather`, `argmax_advance`, graph replay; CI tests for subnormals, refused steps and the tokenizer
- [x] 1.7 GB10 (spark-5): `verify` for both models; Llama 3.1 8B coherent output; 24 steps checked; tok/s at 1K, 4K, 32K (the `:8112` engine stopped for 32K with the user's OK, 16:33:39 to 16:37:13 UTC)
- [x] 1.8 `openspec validate add-bf16-decode-runtime --strict`, archive, `verify.sh`
- [x] 1.9 Review cycle 1: replay count (`fp:b4f0ef0952f4`), bounded greedy gather (`fp:83bcb9e0f559`), weight ownership (`fp:c7921f52e588`), reference restore (`fp:96a508102cf4`), benchmark prefix (`fp:207ec0ac8dbc`), tokenizer ids (`fp:dc612d7ba3fc`), argmax with -inf (`fp:0f8e3bef027c`), binding shapes (`fp:36f5a8a8b697`)

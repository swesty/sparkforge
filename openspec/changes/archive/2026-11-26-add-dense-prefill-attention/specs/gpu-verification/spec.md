## MODIFIED Requirements

### Requirement: Decoded logits match the CPU reference on real weights

`generate --check N` SHALL compare the prefill's logits, then the next N - 1 steps', with the CPU reference. The reference streams the checkpoint and runs the whole prompt first. It SHALL use the scaled metric and the 1/64 tolerance and fail if any step exceeds it. The reference SHALL refuse a model whose shapes differ from the checkpoint's binding, and a failed step SHALL restore every cache slot it wrote.

#### Scenario: Llama 3.1 8B
- **WHEN** Llama 3.1 8B is checked at 4 positions on a GB10 after a 20-token prefill
- **THEN** every position passes (8.5e-3 at the prefill, 5.4e-3 to 6.9e-3 after it) and the greedy token equals the reference's argmax in 4 of 4

### Requirement: Prefill GEMMs report TFLOP/s and a prompt estimate

With `--bench`, `verify` SHALL time each generated prefill GEMM at the planned M and report TFLOP/s. It SHALL print a prefill estimate for the planned prompt: the layers' GEMMs (shared experts included) and `attn_prefill`, plus one `gemv_lm_head`, stating that norms, RoPE and activations are not included. Otherwise it SHALL name the kernel whose timing is missing.

#### Scenario: Llama 3.1 8B on GB10
- **WHEN** `verify --bench` runs for Llama 3.1 8B on spark-5
- **THEN** the GEMMs run at 65.6 to 75.1 TFLOP/s and the estimate is about 417 ms for 2048 tokens (about 4,900 tok/s)

#### Scenario: Qwen3-30B-A3B on GB10
- **WHEN** `verify --bench` runs for Qwen3-30B-A3B in BF16
- **THEN** the dense GEMMs run at 68 to 69 TFLOP/s and the M = 128 expert shapes at 25 to 33

## ADDED Requirements

### Requirement: Prefill attention matches f64 causal attention

`verify` SHALL check `attn_prefill` against f64 causal attention at 1, 128, the planned length, 2048 and 8192 rows from position 0, and 128 rows from position 1000, on all rows up to 256 and otherwise on sampled rows at tile edges, the middle and the end. The error SHALL be scaled by the largest output, since P is rounded to bf16. Rewriting the cache past a row SHALL leave it bitwise unchanged.

#### Scenario: Both example models on GB10
- **WHEN** `verify --bench` runs for Llama 3.1 8B and Qwen3-30B-A3B
- **THEN** every case passes (worst 3.68e-3), causality holds, and attention runs at 69.4 to 70.4 TFLOP/s at 2048 rows and 79.3 to 82.2 at 8192

#### Scenario: Mutations are caught
- **WHEN** the kernel attends one future key, skips a tile a row needs, or drops the O rescale
- **THEN** `verify` fails

### Requirement: Prefill matches token-by-token decode

`verify` SHALL prefill a 100-token prompt on two layers of random weights, in one chunk and in chunks of 48, and compare with decode steps over the same tokens. Every layer's KV cache and the last logits SHALL agree within the scaled tolerance. The greedy token SHALL be their argmax, and the captured graph SHALL decode on from the prefill as a plain step does. Both chunkings SHALL give bitwise the same logits.

#### Scenario: Llama 3.1 8B shapes on GB10
- **WHEN** `verify` runs for Llama 3.1 8B
- **THEN** the KV caches agree within 7.25e-3, the logits within 7.53e-3, and the handoff and chunking checks pass

#### Scenario: RoPE row position dropped
- **WHEN** `rope_kv_append` ignores the row and rotates every prompt row at the chunk's first position
- **THEN** the KV cache, logits and handoff checks fail

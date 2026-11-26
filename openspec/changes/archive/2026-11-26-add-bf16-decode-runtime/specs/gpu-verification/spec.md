## MODIFIED Requirements

### Requirement: Each generated kernel is exercised where it is most likely to break

Verification SHALL check: `rmsnorm` on 4 rows; `rope_kv_append` at positions 1, 4095, 32767 and
`max_position_embeddings - 1` (those below the maximum); decode attention at sequence lengths 1, 37,
1000 and the planned context; `silu_mul`; `residual_add`; `embed_gather` (bitwise rows);
`argmax_advance` (planted ties and NaNs, the recorded token and the advanced position); and every
generated GEMV. Kernels whose status is `planned` SHALL be listed as skipped rather than silently
omitted.

#### Scenario: NVFP4 plan
- **WHEN** `verify` runs for Qwen3-30B-A3B with NVFP4 weights
- **THEN** the four NVFP4 GEMVs are checked, and no GEMV appears in the "not generated yet (skipped)" list

#### Scenario: Flash Next plan
- **WHEN** `verify` runs for `configs/qwen3.8-flash-next-nvfp4.json`
- **THEN** `gemv_expert_gate_up` and `gemv_expert_down` are checked with 10 slots each, and every other op class is listed as skipped

### Requirement: A dense decode step matches the CPU reference at every layer boundary

For a dense BF16 model, `verify` SHALL run a decode step of 2 layers of random weights wired from
the generated kernels against the CPU reference, and compare the hidden state after each layer's
attention and MLP residual adds, and the logits. The same steps replayed from a captured CUDA Graph
SHALL give bitwise the same logits. Any other model SHALL print why the check was skipped.

#### Scenario: Llama 3.1 8B
- **WHEN** `verify` runs for Llama 3.1 8B on a GB10
- **THEN** four steps from an empty cache and two steps after a random 1000-position prefix pass at every boundary, one row per layer boundary and case (measured 7.2e-4 to 6.4e-3), and the graph replay matches bitwise with the argmax recorded as the next token

#### Scenario: Mixture-of-experts model
- **WHEN** `verify` runs for Qwen3-30B-A3B
- **THEN** it prints that the decode step check was skipped because mixture-of-experts layers are not covered

## ADDED Requirements

### Requirement: Decoded logits match the CPU reference on real weights

`generate --check N` SHALL compare the logits of the first N steps (prompt, then generated) with the
CPU reference, which streams the weights from the checkpoint, with the scaled layer-boundary metric
and the 1/64 tolerance, and SHALL fail if any step exceeds it. The reference SHALL refuse a model
whose tensor shapes differ from the checkpoint's binding, and a step that fails partway SHALL restore
every cache slot it wrote.

#### Scenario: Llama 3.1 8B
- **WHEN** 24 steps of Llama 3.1 8B are checked on a GB10
- **THEN** every step passes (worst 1.37e-2 at position 3; positions 5 to 23 measure 4.3e-3 to 8.2e-3) and the greedy token equals the reference's argmax in 24 of 24

# Spec Delta

## Purpose

Reads Hugging Face safetensors checkpoints, validates them against a plan, and makes every planned
weight readable by the generated kernels on GB10.

## ADDED Requirements

### Requirement: Checkpoints are discovered and every byte range is checked

Opening a checkpoint directory SHALL use `model.safetensors.index.json` when present (every shard
it names MUST exist) and `model.safetensors` otherwise. Each shard's header SHALL be parsed and
every tensor's byte range MUST lie inside the file and match its dtype and shape; otherwise opening
fails naming the shard and tensor.

#### Scenario: Sharded checkpoint
- **WHEN** a directory holds an index and four shards
- **THEN** every tensor in the index resolves to a byte range inside its shard

#### Scenario: Truncated shard
- **WHEN** a tensor's byte range ends past the end of its shard
- **THEN** opening fails naming the shard and the tensor

### Requirement: Every tensor the plan needs is present with the planned shape and dtype

For each layer the loader SHALL require the attention projections, MLP projections and both norms,
plus the embedding, the final norm and `lm_head`, each with the shape the plan implies and dtype
BF16. A missing tensor or a wrong shape or dtype MUST fail before any GPU work, naming the tensor
and the expected and found values.

#### Scenario: Wrong shape
- **WHEN** `model.layers.0.self_attn.o_proj.weight` has shape [4096, 2048] but the plan needs [2048, 2048]
- **THEN** loading fails with the tensor name and both shapes

### Requirement: Optional tensors follow the model's flags

`q_norm`/`k_norm` SHALL be required when the spec sets `qk_norm`, and q/k/v biases when it sets
`qkv_bias`. When embeddings are tied, `lm_head` SHALL bind to the embedding tensor. Tensors present
but not needed SHALL be reported, not rejected.

#### Scenario: Tied embeddings
- **WHEN** the config sets `tie_word_embeddings` and the checkpoint has no `lm_head.weight`
- **THEN** `lm_head` binds to `model.embed_tokens.weight`

### Requirement: Unsupported checkpoints are rejected

Loading SHALL be refused, with a message saying the format is not supported yet, for MoE models and
for weight formats other than BF16.

#### Scenario: MoE
- **WHEN** a Qwen3-MoE config is loaded
- **THEN** loading fails stating MoE weight loading is not supported yet

### Requirement: Fused linears are packed in kernel row order

`qkv_proj` SHALL be packed as the q rows, then k rows, then v rows, and `gate_up_proj` as the gate
rows then the up rows, into one GPU allocation each.

#### Scenario: Packed GEMV matches the reference
- **WHEN** the `qkv_proj` GEMV runs on packed layer-0 weights with a random input
- **THEN** its output matches an f64 reference computed from the separate checkpoint tensors within 1/64

### Requirement: Weights are GPU-readable in mapped or copied mode

In mapped mode, unfused tensors SHALL be read by kernels at their memory-mapped address with no
copy and no pinning, and the mode MUST be refused if the device does not report pageable memory
access. A tensor whose data is not 16-byte aligned SHALL be copied even in mapped mode. In copied
mode every tensor SHALL be copied into GPU memory. The load report SHALL state bytes mapped and
bytes copied.

#### Scenario: Mapped mode on GB10
- **WHEN** Llama 3.2 1B is loaded in mapped mode on a GB10
- **THEN** unfused, aligned tensors are reported as mapped and fused linears as copied

### Requirement: Loads never exhaust host memory

On Linux, a load SHALL be refused before allocating if the bytes it would copy exceed
`MemAvailable` minus 4 GiB.

#### Scenario: Too large to copy
- **WHEN** the bytes to copy exceed `MemAvailable` minus 4 GiB
- **THEN** loading fails before any GPU allocation, stating both numbers

### Requirement: Loaded weights are verified on the GPU

`sparkforge verify-weights <checkpoint>` SHALL run the generated GEMV for every linear of the first
and last layer and for `lm_head` on loaded weights, compare each against an f64 reference computed
from the checkpoint bytes (tolerance 1/64), time each linear across all layers, and report load
time and GB/s for each requested mode. It MUST exit non-zero if any check fails.

#### Scenario: Llama 3.1 8B
- **WHEN** `verify-weights` runs on the Llama 3.1 8B BF16 checkpoint on a GB10
- **THEN** every check passes and the report shows load time and GB/s for mapped and copied modes

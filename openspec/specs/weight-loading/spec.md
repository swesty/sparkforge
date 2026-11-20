# weight-loading Specification

## Purpose

Reads Hugging Face safetensors checkpoints, validates them against a plan, and makes every planned
weight readable by the generated kernels on GB10.

## Requirements

### Requirement: Checkpoints are discovered and every byte range is checked

Opening a checkpoint directory SHALL use `model.safetensors.index.json` when present and
`model.safetensors` otherwise. Each tensor's offsets MUST lie inside its shard's data section,
checked without overflow before forming file positions, and its byte count MUST match its dtype and
shape; otherwise opening fails naming the shard and tensor.

#### Scenario: Truncated shard
- **WHEN** a tensor's byte range ends past the end of its shard
- **THEN** opening fails naming the shard and the tensor

#### Scenario: Wrapping offsets
- **WHEN** a tensor's offsets are near 2^64 so that adding the data start would wrap into the header
- **THEN** opening fails instead of accepting header bytes as tensor data

### Requirement: Index entries match their shards

Every index entry MUST name the shard whose header holds that tensor, every tensor in a shard header
MUST be listed in the index, and every shard the index names MUST exist. Shard names MUST be plain
relative paths inside the checkpoint directory (no absolute paths or `..`); symlinked shards, as in
a Hugging Face cache snapshot, are allowed.

#### Scenario: Sharded checkpoint
- **WHEN** a directory holds an index and four shards
- **THEN** every tensor in the index resolves to a byte range inside the shard it names

#### Scenario: Swapped index entries
- **WHEN** the index assigns a tensor to the other shard
- **THEN** opening fails naming the tensor and the declared shard

#### Scenario: Escaping index
- **WHEN** every index entry names `../previous/model.safetensors` or an absolute path
- **THEN** opening fails before any shard is opened, naming the offending shard name

### Requirement: Repeated JSON keys are rejected

A tensor name that appears twice in the index's `weight_map` or in a shard header MUST make opening
fail, rather than keeping one of the entries.

#### Scenario: Duplicate index key
- **WHEN** the index lists `model.norm.weight` twice, first pointing at a missing shard
- **THEN** opening fails naming the duplicate key

### Requirement: Tensors tile each shard's data section exactly

The non-empty tensors of a shard MUST cover its data section exactly, with no overlapping ranges and
no unaccounted bytes; empty tensors are allowed.

#### Scenario: Aliased projection
- **WHEN** `up_proj` is given the same byte range as `gate_proj`
- **THEN** opening fails reporting the overlap

### Requirement: A shard that changes after opening is an error

Tensor reads SHALL be fallible positioned reads. A read MUST fail, rather than terminate the process
or return bytes from a different version, when the shard's size, identity or modification stamp
differs from when it was opened, checked after each read (to filesystem timestamp granularity).

#### Scenario: Truncated after validation
- **WHEN** a shard is truncated after the checkpoint was opened and before a tensor is read
- **THEN** the read fails stating that the shard changed size

#### Scenario: Same-length rewrite between chunks
- **WHEN** a tensor's later bytes are rewritten in place, keeping the file length, after its first chunk was read
- **THEN** reading the next chunk fails stating that the shard was modified

### Requirement: Shards are regular files opened without blocking

Shards SHALL be opened without blocking, and the opened file MUST be a regular file.

#### Scenario: FIFO shard
- **WHEN** the shard path is a FIFO with no writer
- **THEN** opening fails promptly stating that it is not a regular file

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

### Requirement: Weights are copied into GPU memory

Every planned linear SHALL be copied into a GPU allocation before kernels use it. Kernels SHALL NOT
read checkpoint file mappings in place: on GB10 that measured about 0.3 GB/s on first touch and
about 35% below device memory once warm. The load report SHALL state the bytes copied and the time.

#### Scenario: Llama 3.1 8B
- **WHEN** the Llama 3.1 8B BF16 checkpoint is loaded on a GB10
- **THEN** 13.98 GiB is copied into GPU memory and the report states the load time

### Requirement: Loads never exhaust host memory

On Linux, a load SHALL be refused before allocating if the bytes it would copy, plus its bounded
staging buffer and, when verifying, the reference computation's scratch, exceed `MemAvailable`
minus 4 GiB. Admission SHALL be serialized across the user's sparkforge processes until the admitted
allocations are made. Neither uploads nor references SHALL hold a whole tensor in host memory.

#### Scenario: Too large to copy
- **WHEN** the bytes to copy plus overhead exceed `MemAvailable` minus 4 GiB
- **THEN** loading fails before any GPU allocation, stating both numbers

#### Scenario: Concurrent loads
- **WHEN** two sparkforge loads of 14 GiB start together with 20 GiB available
- **THEN** the second waits until the first has allocated, then is refused

#### Scenario: Wide vocabulary
- **WHEN** `verify-weights` checks Llama 3.1 8B's `lm_head` (1.05 GB of weights)
- **THEN** the process's peak host memory stays bounded (measured 225 MB) instead of scaling with the tensor

### Requirement: Loaded weights are verified on the GPU

`sparkforge verify-weights <checkpoint>` SHALL run the generated GEMV for every linear of the first
and last layer and for `lm_head` on loaded weights, compare each against an f64 reference computed
from the checkpoint bytes (tolerance 1/64), time each linear across all layers, and report load
time and GB/s. It MUST exit non-zero if any check fails.

#### Scenario: Llama 3.1 8B
- **WHEN** `verify-weights` runs on the Llama 3.1 8B BF16 checkpoint on a GB10
- **THEN** every check passes and the report shows load time and GB/s for every linear

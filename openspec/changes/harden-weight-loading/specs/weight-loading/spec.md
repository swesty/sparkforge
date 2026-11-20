# Spec Delta

## MODIFIED Requirements

### Requirement: Checkpoints are discovered and every byte range is checked

Opening a checkpoint directory SHALL resolve it once to its canonical path and read the config,
index and shards from it. It SHALL use `model.safetensors.index.json` when that directory entry
exists (a dangling link is an error) and `model.safetensors` otherwise. Each tensor's offsets MUST
lie inside its shard's data section, checked without overflow, and its byte count MUST match its
dtype and shape.

#### Scenario: Truncated shard
- **WHEN** a tensor's byte range ends past the end of its shard
- **THEN** opening fails naming the shard and the tensor

#### Scenario: Wrapping offsets
- **WHEN** a tensor's offsets are near 2^64 so that adding the data start would wrap into the header
- **THEN** opening fails instead of accepting header bytes as tensor data

#### Scenario: Symlink switched after opening
- **WHEN** a checkpoint is opened through a symlink that is then switched to another generation
- **THEN** every read still comes from the generation that was opened

#### Scenario: Dangling index link
- **WHEN** `model.safetensors.index.json` is a dangling symlink beside a valid `model.safetensors`
- **THEN** opening fails naming the index instead of loading the single file

### Requirement: Shards are regular files opened without blocking

Shards, `config.json` and the index SHALL be opened without blocking, and each opened file MUST be
a regular file; metadata files are also size-capped.

#### Scenario: FIFO shard
- **WHEN** the shard path is a FIFO with no writer
- **THEN** opening fails promptly stating that it is not a regular file

#### Scenario: FIFO metadata
- **WHEN** `config.json` or the index is a FIFO with no writer
- **THEN** reading it fails within 5 seconds stating that it is not a regular file

### Requirement: Every tensor the plan needs is present with the planned shape and dtype

For each layer the loader SHALL require the attention projections, MLP projections and both norms,
plus the embedding, the final norm and the resolved `lm_head` binding (the embedding when tied),
each with the planned shape and dtype BF16. A missing tensor or a wrong shape or dtype MUST fail
before any GPU work, naming the tensor and the expected and found values.

#### Scenario: Wrong shape
- **WHEN** `model.layers.0.self_attn.o_proj.weight` has shape [4096, 2048] but the plan needs [2048, 2048]
- **THEN** loading fails with the tensor name and both shapes

#### Scenario: Untied model without lm_head
- **WHEN** an untied config is bound to a checkpoint without `lm_head.weight`
- **THEN** binding fails before GPU allocation naming `lm_head.weight`, while the same checkpoint binds when the config ties embeddings

### Requirement: Loads never exhaust host memory

A load SHALL be refused before any CUDA call if its copied bytes, bounded staging buffer and, when
verifying, reference scratch exceed `MemAvailable` minus 4 GiB. Admission SHALL be serialized,
until the allocations are made, across all users' sparkforge processes when the node-wide lock is
provisioned, else across this user's, at a location chosen by ownership, never by environment.

#### Scenario: Too large to copy
- **WHEN** the bytes to copy plus overhead exceed `MemAvailable` minus 4 GiB
- **THEN** loading fails before any CUDA call, stating both numbers

#### Scenario: Concurrent loads
- **WHEN** two sparkforge loads of 14 GiB start together with 20 GiB available
- **THEN** the second waits until the first has allocated, then is refused

#### Scenario: Loads by two users
- **WHEN** the node-wide lock is provisioned and two different users start loads
- **THEN** both use the same lock, so the second waits for the first

#### Scenario: Wide vocabulary
- **WHEN** `verify-weights` checks Llama 3.1 8B's `lm_head` (1.05 GB of weights)
- **THEN** the process's peak host memory stays bounded (measured 225 MB) instead of scaling with the tensor

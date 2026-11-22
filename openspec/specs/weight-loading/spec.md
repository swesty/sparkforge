# weight-loading Specification

## Purpose

Reads Hugging Face safetensors checkpoints, validates them against a plan, and makes every planned
weight readable by the generated kernels on GB10.

## Requirements

### Requirement: Checkpoints are discovered and every byte range is checked

Opening a checkpoint directory SHALL open it once, needing only search permission, and read the
config, index and shards relative to it, following a shard symlink only when its target stays inside
it or is a Hugging Face snapshot's blob. It SHALL use `model.safetensors.index.json` when that entry
exists (a dangling link is an error), else `model.safetensors`. Tensor offsets MUST lie inside the
shard's data section, checked without overflow, and byte counts MUST match dtype and shape.

#### Scenario: Truncated shard
- **WHEN** a tensor's byte range ends past the end of its shard
- **THEN** opening fails naming the shard and the tensor

#### Scenario: Wrapping offsets
- **WHEN** a tensor's offsets are near 2^64 so that adding the data start would wrap into the header
- **THEN** opening fails instead of accepting header bytes as tensor data

#### Scenario: Symlink switched after opening
- **WHEN** a checkpoint is opened through a symlink that is then switched to another generation
- **THEN** every read still comes from the generation that was opened

#### Scenario: Directory replaced after the config read
- **WHEN** the canonical directory is moved aside and replaced by another generation after the config is read
- **THEN** the index and shards still come from the generation the config came from

#### Scenario: Shard linked through an alias
- **WHEN** a shard entry is a symlink whose target passes through another symlink, such as a `current` alias
- **THEN** opening fails naming the shard, while a link straight to a file inside the checkpoint still opens

#### Scenario: Shard linked outside the checkpoint
- **WHEN** a shard links straight to a file in a real directory outside the checkpoint, which is replaced by another generation after the config read
- **THEN** opening fails naming the shard, for absolute and relative targets alike

#### Scenario: Hugging Face snapshot
- **WHEN** a shard in `<repo>/snapshots/<rev>` links to `../../blobs/<sha256>`
- **THEN** it opens, while a blob name that is not a hash, or the same link from a directory that is not a snapshot, is refused

#### Scenario: Blob store replaced after the config read
- **WHEN** a snapshot's `<repo>/blobs` is replaced, after the config read, by one holding other bytes under the same hash names
- **THEN** the shards still come from the blob store the snapshot was opened with

#### Scenario: Snapshot moved during resolution
- **WHEN** a snapshot revision is moved or replaced, or its blob store is replaced or rearranged, at any point during the load
- **THEN** every blob the revision links to is verified against its name (SHA256, or the Git object SHA1), so another generation's bytes are refused rather than paired with the config

#### Scenario: Blob that doesn't match its name
- **WHEN** a snapshot links to `../../blobs/<name>` and that blob's bytes don't hash to `<name>`
- **THEN** opening fails naming the blob and the hash it found, for shards and for metadata files

#### Scenario: Unused blob store
- **WHEN** a snapshot of plain files sits beside a `blobs` path that is a symlink or not a directory
- **THEN** it opens, while a snapshot that links into that blob store fails naming it

#### Scenario: Link through a parent component
- **WHEN** a shard links to `sub/../weights.safetensors`
- **THEN** it opens `weights.safetensors` in the checkpoint directory without resolving `..` through `sub`

#### Scenario: Blob in a shared store
- **WHEN** a snapshot's `../../blobs/<sha256>` is itself a link `../../blobs/<xx>/<name>` into the cache's shared store
- **THEN** that one hop is followed inside the store, with no further link, and the bytes are verified against `<sha256>`

#### Scenario: Blob linking out of the store
- **WHEN** a blob links anywhere else, such as an absolute path to a readable shard
- **THEN** opening fails before that target is opened

#### Scenario: Search-only directory
- **WHEN** the checkpoint directory and a shard's subdirectory grant search but not read permission
- **THEN** the checkpoint opens on Linux

#### Scenario: Dangling index link
- **WHEN** `model.safetensors.index.json` is a dangling symlink beside a valid `model.safetensors`
- **THEN** opening fails naming the index instead of loading the single file

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

Shards, `config.json` and the index SHALL be opened without blocking, and each opened file MUST be
a regular file; metadata files are also size-capped, counting bytes that arrive after the size check.

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

### Requirement: Loaded weights are verified on the GPU

`sparkforge verify-weights <checkpoint>` SHALL run the generated GEMV for every linear of the first
and last layer and for `lm_head` on loaded weights, compare each against an f64 reference computed
from the checkpoint bytes (tolerance 1/64), time each linear across all layers, and report load
time and GB/s. It MUST exit non-zero if any check fails.

#### Scenario: Llama 3.1 8B
- **WHEN** `verify-weights` runs on the Llama 3.1 8B BF16 checkpoint on a GB10
- **THEN** every check passes and the report shows load time and GB/s for every linear

### Requirement: Loads never exhaust host memory on a single-account node

A load SHALL be refused before any CUDA call if its copied bytes, staging buffer and, when
verifying, reference scratch exceed `MemAvailable` minus 4 GiB, and rechecked before allocating.
Admission SHALL be held until that memory is allocated, or freed after verification, and
serialized through one lock per account, under the home directory from the password database.
sparkforge assumes one account runs it per node.

#### Scenario: Too large to copy
- **WHEN** the bytes to copy plus overhead exceed `MemAvailable` minus 4 GiB
- **THEN** loading fails before any CUDA call, stating both numbers

#### Scenario: Concurrent loads
- **WHEN** two sparkforge loads of 14 GiB start together with 20 GiB available
- **THEN** the second waits until the first has allocated, then is refused

#### Scenario: Second load during verification
- **WHEN** a second `verify-weights` starts while the first is copying or verifying, with room for only one
- **THEN** it waits until the first has freed its memory, then is admitted

#### Scenario: Login during a load
- **WHEN** a load is admitted and a login then creates the account's runtime directory
- **THEN** the next load waits on the same lock, which depends only on the home directory

#### Scenario: Lock location from ownership
- **WHEN** the account has no password-database home, or its lock directory is writable by others
- **THEN** admission refuses instead of using an environment variable or a shared temp path

#### Scenario: Stuck holder
- **WHEN** another process holds the lock for 30 minutes
- **THEN** the waiting load fails naming the lock instead of hanging

#### Scenario: Holder released after the wait
- **WHEN** the lock was found held and is released only after the wait limit has passed
- **THEN** the waiting load times out instead of being admitted late

#### Scenario: Lock replaced while waiting
- **WHEN** a load waits on the lock file and the file is replaced before it gets the lock
- **THEN** it locks the new file within the same deadline, so it is never admitted alongside a load holding the new file

#### Scenario: Upgrade from a loader before SHA-765
- **WHEN** sparkforge is upgraded from any earlier version, including one that locked in `/run/user/<uid>`, `$XDG_RUNTIME_DIR`, the temp directory or `/run/sparkforge`
- **THEN** running loads are drained first, since an earlier version may hold a lock this one doesn't take

#### Scenario: State directory deleted during a load
- **WHEN** the account deletes the lock file or `~/.local/state/sparkforge` while a load is admitted
- **THEN** that is unsupported: a load started afterwards locks a new file and isn't coordinated with the admitted one, so the directory must not be removed while sparkforge runs

#### Scenario: Memory used after admission
- **WHEN** `MemAvailable` drops below the guard between admission and upload
- **THEN** the upload refuses before allocating

#### Scenario: Wide vocabulary
- **WHEN** `verify-weights` checks Llama 3.1 8B's `lm_head` (1.05 GB of weights)
- **THEN** the process's peak host memory stays bounded (measured 225 MB) instead of scaling with the tensor

### Requirement: Hybrid checkpoints are checked tensor by tensor

For a hybrid checkpoint, `sparkforge weights` SHALL check every tensor before any GPU work: each tensor the plan needs exists once, with the dtype and shape of the format its module is priced at, and every other tensor belongs to the skipped vision tower. PLE table shard rows SHALL come from the checkpoint's per-head table sizes. Any mismatch MUST fail naming the tensor and the expected and found values.

#### Scenario: Flash Next checkpoint
- **WHEN** `sparkforge weights` opens the Qwen3.8 Flash Next NVFP4 checkpoint (revision 6909a5bed089) on spark-5
- **THEN** 301,260 tensors (105.40 GB) match the plan, the 333 vision-tower tensors are skipped, the PLE shards hold 2,500,012 rows, and the plan prices 105.40 GB of resident text weights

#### Scenario: Wrong dtype or shape
- **WHEN** a GDN projection's MXFP8 scale is F8_E4M3 and an expert's packed weight has twice its columns
- **THEN** the check fails naming both tensors with the expected and found dtype and shape

#### Scenario: Missing scale and unexpected tensor
- **WHEN** an NVFP4 expert lacks its `input_scale` and a text layer holds a tensor the plan doesn't use
- **THEN** the check fails naming both, while vision-tower tensors are not reported

#### Scenario: Two spellings of one tensor
- **WHEN** a tensor exists under both `model.language_model.` and `model.`
- **THEN** the check fails naming both names

#### Scenario: Format map disagrees with the tensors
- **WHEN** the map lists a projection as NVFP4 and the checkpoint holds MXFP8 tensors for it
- **THEN** the check fails naming the NVFP4 tensors that are missing or mismatched

#### Scenario: PLE table sizes
- **WHEN** the per-head table sizes imply a different shard row count, a head below `ngram_vocab_size_base`, or a padded total that doesn't split into `split_ngram_parts`
- **THEN** the check fails saying which

### Requirement: Only dense BF16 checkpoints load into GPU memory

Binding for the GPU MUST fail, before any GPU work, for a hybrid-layout checkpoint (stating that it is checked but not loaded yet), for one whose config declares mixed precision, and for one whose planned tensors are not BF16, naming the architecture or the tensor. A format override used for planning MUST NOT lift this refusal.

#### Scenario: Flash Next checkpoint
- **WHEN** a Qwen3.8 Flash Next checkpoint is bound for the GPU
- **THEN** binding fails stating that hybrid checkpoints are checked tensor by tensor but do not load into GPU memory yet

#### Scenario: Override used for planning
- **WHEN** a dense MIXED_PRECISION config is parsed with a BF16 override and its checkpoint is bound
- **THEN** binding fails stating that mixed-precision checkpoints do not load yet

#### Scenario: Quantized tensors behind a BF16 config
- **WHEN** a config without `quantization_config` is bound to a checkpoint whose `o_proj` is F8_E4M3 with a scale tensor
- **THEN** binding fails before GPU work, naming the tensor and its dtype

### Requirement: The checked PLE table prices the reported budget

`sparkforge weights` SHALL check a hybrid checkpoint's tensors before building any plan, and report the resident budget with the PLE table at the rows it checked. Resolving a spec onto another checkpoint MUST clear rows checked earlier. PLE row totals and padding that cannot be represented in 64 bits MUST fail the check, not wrap or panic.

#### Scenario: One more row per shard
- **WHEN** two checkpoints of one config have per-head sizes making 15- and 16-row shards in 4 NVFP4 shards of 16 columns
- **THEN** both pass, and the second's resident budget is 36 bytes larger

#### Scenario: Overflowing padding
- **WHEN** the per-head sizes sum within 64 bits but their padding to the configured multiple does not
- **THEN** the check fails saying the row count overflows

#### Scenario: Spec reused across checkpoints
- **WHEN** a spec checked against the 15-row checkpoint is resolved onto the 16-row one
- **THEN** its checked rows are cleared and its budget equals a fresh resolution's, until the 16-row checkpoint is checked

### Requirement: A tied embedding carries the head GEMV's tensors

With tied embeddings, the embedding matrix SHALL be checked as a GEMV operand, counted once: in NVFP4 it MUST have its activation scale, and in weight-only NVFP4 it MUST NOT.

#### Scenario: Tied NVFP4 and weight-only NVFP4
- **WHEN** a tied checkpoint's embedding is NVFP4, or W4A16 NVFP4
- **THEN** NVFP4 fails without `embed_tokens.input_scale` and passes with it, and W4A16 NVFP4 passes without it and reports it as unexpected if present

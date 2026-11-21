## MODIFIED Requirements

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

#### Scenario: Search-only directory
- **WHEN** the checkpoint directory and a shard's subdirectory grant search but not read permission
- **THEN** the checkpoint opens on Linux

#### Scenario: Dangling index link
- **WHEN** `model.safetensors.index.json` is a dangling symlink beside a valid `model.safetensors`
- **THEN** opening fails naming the index instead of loading the single file

### Requirement: Loads never exhaust host memory

A load SHALL be refused before any CUDA call if its copied bytes, staging buffer and, when
verifying, reference scratch exceed `MemAvailable` minus 4 GiB, and rechecked before allocating.
Admission SHALL be held until that memory is allocated, or freed after verification, and
serialized across this user's processes and, through a group-only node lock when provisioned,
across all users'. Lock locations come from ownership, never environment.

#### Scenario: Too large to copy
- **WHEN** the bytes to copy plus overhead exceed `MemAvailable` minus 4 GiB
- **THEN** loading fails before any CUDA call, stating both numbers

#### Scenario: Concurrent loads
- **WHEN** two sparkforge loads of 14 GiB start together with 20 GiB available
- **THEN** the second waits until the first has allocated, then is refused

#### Scenario: Loads by two users
- **WHEN** the node-wide lock is provisioned and two different users start loads
- **THEN** both use the same lock, so the second waits for the first

#### Scenario: Lock open to every account
- **WHEN** the node-wide lock file grants access to every account
- **THEN** admission refuses it, naming the mode, instead of letting any account hold it

#### Scenario: Lock in the wrong group
- **WHEN** the node-wide lock is mode 0640 but belongs to a group other than `sparkforge`
- **THEN** admission refuses it, naming both groups

#### Scenario: Lock opened to another account by an ACL
- **WHEN** the node-wide lock is mode 0640 in group `sparkforge` but an ACL entry grants a named user read access
- **THEN** admission refuses it, naming that user (mode bits and ACL entries both count)

#### Scenario: Upgrade from a custom runtime directory
- **WHEN** a load by the loader before SHA-765, run with a custom `XDG_RUNTIME_DIR`, holds its lock there
- **THEN** a new load waits for it, and no lock is created there when none exists

#### Scenario: Second load during verification
- **WHEN** a second `verify-weights` starts while the first is copying or verifying, with room for only one
- **THEN** it waits until the first has freed its memory, then is admitted

#### Scenario: Login during a load
- **WHEN** a user's load is admitted and a login then creates that user's runtime directory
- **THEN** the user's next load waits on the same lock

#### Scenario: Memory used after admission
- **WHEN** `MemAvailable` drops below the guard between admission and upload
- **THEN** the upload refuses before allocating

#### Scenario: Wide vocabulary
- **WHEN** `verify-weights` checks Llama 3.1 8B's `lm_head` (1.05 GB of weights)
- **THEN** the process's peak host memory stays bounded (measured 225 MB) instead of scaling with the tensor

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
- **WHEN** a snapshot's `../../blobs/<sha256>` is itself a link into a shared, deduplicated store under another name
- **THEN** the link is followed and the bytes are verified against `<sha256>`, so other bytes at the end of the chain are refused

#### Scenario: Search-only directory
- **WHEN** the checkpoint directory and a shard's subdirectory grant search but not read permission
- **THEN** the checkpoint opens on Linux

#### Scenario: Dangling index link
- **WHEN** `model.safetensors.index.json` is a dangling symlink beside a valid `model.safetensors`
- **THEN** opening fails naming the index instead of loading the single file

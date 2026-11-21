# Design

## Decisions

### 1. Identities from tensor names

A module is a tensor name without its last component. `...q_proj.weight` and `...q_proj.weight_scale` are one module, and the 128 `ngram_embedding.shard_N.weight` tensors are 128 modules under `ngram_embedding`. Paths are canonical after removing a language-model prefix (`model.language_model.`, `model.`, ...), so `layers.0.mlp.experts` and `model.language_model.layers.0.mlp.experts` are the same identity. Other towers keep their names (`mtp.`, `visual.`, `lm_head`).

### 2. Entries cover subtrees

ModelOpt names module groups (`layers.0.mlp.experts`) and tables (`...ngram_embedding`), not tensors, so an entry applies to every module at or under its path. Coverage replaces alias rules:
- Two spellings of one module cover the same modules and must agree.
- A path where the checkpoint has no module covers nothing. It is reported, not honored.

The checkpoint's second MTP entry, `mtp.layers.48.mlp.experts`, covers nothing because the experts live under `mtp.layers.0`.

### 3. Config-only planning stays, narrowed

Without a checkpoint, the matcher still runs, but only MTP indices below the configured count match, so a stale index never prices the draft layer. A tied head takes the embedding's format in both modes.

### 4. Where the names come from

`plan`/`emit --checkpoint` read the index's keys, which takes 0.34 s for 301,593 names on spark-5, including hashing the index blob. No shard is opened. `weights` uses the opened checkpoint's headers.

### 5. Blobs in a shared store

On spark-5 the cache links each `<repo>/blobs/<sha256>` on to `hub/blobs/<xx>/<id>`, a deduplicated store whose names aren't content hashes. The loader refused this, because it allows no link past the snapshot's own. Since every blob's bytes are verified against the sha256 name (`verify-hf-blobs`), the kernel may follow links from the blob onward. Measured: the sha256 names are the content hashes, for the index (`e65210a6…`) and a 2.79 GB shard (`94994a8d…`).

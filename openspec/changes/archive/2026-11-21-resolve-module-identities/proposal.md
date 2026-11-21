# Proposal

## Why

The planner matched `quantized_layers` paths against module-name patterns. codex-review raised path-matching edge cases in every round of swesty/sparkforge#9 and #10, and each fix exposed another corner. Three were still open:
- **PLE table aliases:** `layers.1.ple.ple_embedding.ngram_embedding` and `ple_embedding.ngram_embedding` were both accepted and never compared.
- **Stale MTP index:** `mtp.layers.999.self_attn.v_proj` priced the real draft projection, because any index matched.
- **Tied head:** with tied weights, `lm_head` ignored a format declared under `embed_tokens`.

The mixed-format loader has to map every tensor to its module anyway, so the identities can come from the checkpoint itself.

## What Changes

- **Module table:** `modules::ModuleTable` holds canonical module paths, built from tensor names (the name minus its last component, with any language-model prefix removed). `modules::resolve` applies each map entry to the modules at or under its path. Entries that cover one module must agree, and an entry that covers nothing is reported as unused.
- **Planner:** `ModelSpec::resolve_modules(names)`, and the planner's lookups go through it when it is set:
  - per-layer modules by exact path;
  - MTP over the layer indices the checkpoint holds;
  - PLE only where its tensors are;
  - `lm_head` from its own tensor, or, when tied, the embedding's format.
- **Config-only planning:** keeps the path matcher, narrowed to MTP indices below the configured MTP layer count, with the same tied-head rule.
- **Embedding:** priced at its own format, both resident and in the decode gather.
- **CLI:** `plan`/`emit --checkpoint <dir>` resolve from the index's tensor names, and `weights` resolves from the opened checkpoint. `weights::tensor_names` reads the index, or the single file's header when there is no index.
- **Loader:** a Hugging Face blob that is itself a link into a shared store (spark-5's cache: `<repo>/blobs/<sha256> -> hub/blobs/<xx>/<id>`) is followed. Its bytes are verified against the sha256 name.

**Expected GB10 effect:** none. On the real checkpoint the plan is byte-identical to the config-only plan and has the same budget and ops as `main`. The one new note reports `mtp.layers.48.mlp.experts` as unused.

## Capabilities

### Modified Capabilities
- `kernel-planning`: formats resolve onto a checkpoint's modules; MTP indices are bounded; a tied head takes the embedding's format.
- `weight-loading`: blob links into a shared store are followed and verified.

## Impact

- New `src/modules.rs`; `src/config.rs` (`resolved`, `resolve_modules`), `src/plan.rs` (`Formats`), `src/weights.rs` (`tensor_names`, `open_file_following`), `src/main.rs`.

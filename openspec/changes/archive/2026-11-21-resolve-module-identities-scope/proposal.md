# Proposal

## Why

codex-review cycle 2 on swesty/sparkforge#17 approved it with seven P2 follow-ups:
- blob links could lead out of the cache;
- a model-level PLE table wasn't checked for mixed formats;
- MTP formats could differ across draft layers;
- MTP index spellings such as `00`;
- tied-head spellings;
- tensors under a projection could reprice it;
- an agreeing child entry was reported unused.

## What Changes

- **Blob links:** a blob may link one hop, as exactly `../../blobs/<xx>/<name>`, walked inside the cache's shared store (opened from the pinned repository) with no further link. Anything else is refused before its target opens. The bytes are still verified.
- **Single-format checks:** they cover every path the PLE lookup accepts, including the model-level ones.
- **MTP indices:** they must be canonical decimals.
- **Pricing:** a lookup prices the module at exactly its path when one exists, and aggregates the subtree only for groups. Group checks look at the whole subtree.
- **Usage:** the resolver keeps every agreeing entry that covers a module, for usage tracking.
- **Already covered:** MTP formats across draft layers can't differ, because a config with more than one MTP layer is refused at parse time, and a test pins that. Tied-head spellings were already rejected by the alias check, and a test pins that too.

**Expected GB10 effect:** none. The real checkpoint's plan is byte-identical, and its blob links open under the one-hop rule.

## Capabilities

### Modified Capabilities
- `kernel-planning`: exact pricing, canonical MTP indices, all PLE paths checked.
- `weight-loading`: one confined hop into the shared store.

## Impact

- `src/modules.rs`, `src/config.rs`, `src/weights.rs`; tests.

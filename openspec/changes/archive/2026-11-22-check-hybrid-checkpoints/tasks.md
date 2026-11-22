# Tasks

## 1. Tensor inventory and check
- [x] 1.1 `inventory::hybrid`: every tensor of the hybrid layout, per format, with dtypes and shapes from the reference headers
- [x] 1.2 `inventory::ple_path`: exactly one PLE location
- [x] 1.3 `weights::check_hybrid`: each wanted tensor once with its dtype and shape, the vision tower skipped, anything else unexpected; PLE shard rows from the per-head sizes
- [x] 1.4 `sparkforge weights` reports the check for hybrid checkpoints; `bind` still refuses them for the GPU
- [x] 1.5 Parse `split_ngram_parts` and `make_ngram_vocab_size_divisible_by`

## 2. codex-review cycle 3 follow-ups on #17
- [x] 2.1 Refuse two live PLE paths (fp:0b213af19511)
- [x] 2.2 Group lookups count every entry under the group (fp:884bf7a92e48)
- [x] 2.3 Dense head and embedding take resolved formats (fp:509d200bd957)
- [x] 2.4 Canonical MTP indices in config-only lookups (fp:ab21934c075a)
- [x] 2.5 `spec_from_checkpoint` builds the spec from the checkpoint's config (fp:4abc58b2da3e)
- [x] 2.6 Exact projection lookups; every planned module must exist (fp:93d321fef646)

## 3. Verification
- [x] 3.1 `cargo test`, `cargo check --features cuda`
- [x] 3.2 `sparkforge weights` on the reference checkpoint on spark-5; `plan --checkpoint` equals the config-only plan

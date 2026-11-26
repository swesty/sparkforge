# Proposal

## Why

codex-review cycle 3 on swesty/sparkforge#19 (head `be6515a`, APPROVE WITH FOLLOWUPS) left three findings (SHA-795). None changes the weights that get loaded:
- **`fp:44b41fc11198`:** admission for `load-weights` reserved twice the gathered bytes plus the load's host scratch. The gather also keeps one `u64` row ID per token and head alive, and the scratch is already freed by then. At the maximum `--gather-tokens` (1,048,576) the row IDs take 128 MiB against 120 MiB of scratch on the reference checkpoint, so 8 MiB came out of the 4 GiB headroom.
- **`fp:04b498446b41`:** the MTP planning budget left out `mtp.pre_fc_norm_embedding.weight` (BF16, hidden width) and `mtp.pre_fc_norm_hidden.weight` (BF16, residual width), which `inventory.rs` lists. The omission predates #19.
- **`fp:69a474a7ea8f`:** `load-weights` read its "before" `MemAvailable` ahead of admission, so memory another process freed while the load waited was attributed to the load.

## What Changes

- **Admission takes the largest phase, with row IDs counted.** `PleTable::gather_peak_bytes` is the gathered bytes twice plus 8 bytes per row. `load_plan::hybrid_overhead_bytes` is the larger of the load scratch (copy and read-back, from `host_scratch_bytes`) and, in disk mode, that gather peak. `load-weights` prints each term next to the reservation.
- **MTP budget:** both pre-FC norm vectors are added once whenever MTP is enabled: 640 bytes on the tiny fixture ((64 + 256) x 2), 25,600 bytes on Flash Next ((2,560 + 4 x 2,560) x 2).
- **Reported baseline:** `Admission::confirm` returns the `MemAvailable` it checked. `load-weights` prints it as "before the copy". `verify-weights` had the same pre-admission reading, labelled "at start"; it now prints the confirmed reading too.

**Expected GB10 effect:** none on decode, prefill or load time. Admission for `load-weights --ple disk` reserves `max(scratch, gather peak)` instead of their sum: less at the default 4,096 tokens (the 120 MiB scratch dominates), and 8 MiB more at the maximum. Checked on spark-5 from the admission line `load-weights` prints, which comes before any CUDA call.

## Non-goals

- Changing the gather itself, or how many row IDs it builds. SHA-792 (parallel disk reads) owns the gather.
- Changing what admission decides for `verify-weights`. Only its printed baseline moves.

## Capabilities

### Modified Capabilities
- `weight-loading`: the hybrid admission covers the largest phase with the gather's row IDs, and the reported "before" figure is the one admission checked.

## Impact

- `src/load_plan.rs` (`gather_peak_bytes`, `hybrid_overhead_bytes`), `src/main.rs` (`load-weights`, `verify-weights` output), `src/weights.rs` (`Admission::confirm` returns the reading), `src/device_weights.rs` and `src/verify.rs` (carry it), `src/plan.rs` (`mtp_weight_bytes`).
- Tests: `tests/load_plan.rs`, `tests/weights.rs`.

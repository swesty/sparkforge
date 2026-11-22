# Proposal

## Why

`sparkforge weights` checks a Qwen3.8 Flash Next checkpoint tensor by tensor, but nothing copies one into GPU memory. The decode kernels need two things from the loader that a byte-for-byte copy doesn't give them:

- **A repacked expert layout.** The routed experts (67.95 GB, 64% of the text weights) are NVFP4 in ModelOpt's `[N, K/2]` layout with linear per-16 E4M3 scales. The BF16 x FP4 decode GEMV reads a tiled, lane-ordered layout with S0E5M3 scales, the layout FlashInfer's SM12x GEMV reads.
- **A choice for the PLE n-gram table.** The table is 28.80 GB. Kept resident, the text weights take 98 GiB of GB10's 119.7 GiB. Read per token from the checkpoint instead, they take 71 GiB.

codex-review also approved #18 with four follow-ups (cycle 1 on head `768ca05d`):

1. **fp:25aaaee34a54:** the MTP input projections were always BF16 in the inventory and the budget.
2. **fp:cf55fb8f7ebd:** MTP mixer norms were priced at 4 bytes, but the checkpoint stores them as BF16.
3. **fp:347106d738b2:** an extra matrix at the PLE table's own path passed checkpoint-backed planning.
4. **fp:d0f47806a572:** an oversized expert count allocated one entry per declared expert before failing.

## What Changes

- **`src/repack.rs` (new):** the NVFP4 tile repack and the E4M3 to S0E5M3 scale conversion, with their inverses. Both are re-derived from FlashInfer `db2373490172` (Apache-2.0) and checked against bytes that FlashInfer's own functions produced.
- **`src/load_plan.rs` (new):** the buffers a hybrid checkpoint becomes on the GPU.
  - Each layer's NVFP4 experts are grouped into fused gate/up and down buffers, plus their scales and scalars.
  - Every other tensor gets one buffer, copied as stored.
  - The PLE table is resident or on disk (`PleMode`), with a row gather for disk mode.
  - Loading goes through a `Device` trait, so the whole load path runs against host memory in tests. Read-back sampling compares the GPU's bytes with the checkpoint.
- **`device_weights`:** the CUDA `Device`, `load_hybrid` (admission, copy, read-back) and a pinned PLE gather timing.
- **`sparkforge load-weights <dir> [--ple disk|resident]`** (`--features cuda`).
- **Follow-ups:**
  - **MTP input projections:** looked up at `mtp.fc_embedding` and `mtp.fc_hidden`, and checked and priced at their own format.
  - **MTP mixer norms:** priced at 2 bytes.
  - **Table root:** only `weight_scale_2` may sit at the table root.
  - **Expert count:** live experts are counted before anything is built per declared expert.
- **Dense binding:** it still refuses hybrid checkpoints, now pointing to `load-weights`.

**Expected GB10 effect:** Flash Next weights become resident on the GPU, 71.34 GiB with the PLE table on disk or 98.16 GiB resident (measured, see tasks). There are no kernel changes. The reference budget drops 61,440 bytes (three MTP mixer norms of 10,240 elements, at 2 bytes less each), which is invisible at the displayed precision.

## Capabilities

### Modified Capabilities
- `weight-loading`, `kernel-planning`.

## Impact

- **Source:** `src/repack.rs`, `src/load_plan.rs`, `src/device_weights.rs`, `src/main.rs`, `src/inventory.rs`, `src/config.rs`, `src/plan.rs`, `src/weights.rs`.
- **Tests:** `tests/repack.rs`, `tests/load_plan.rs`, `tests/weights.rs`, `tests/common/mod.rs`.
- **Golden data and its generator:** `tests/golden/fp4_pack_flashinfer_db23734.json`, `tools/golden/fp4_pack_flashinfer.py`.

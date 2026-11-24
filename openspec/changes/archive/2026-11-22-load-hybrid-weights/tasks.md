# Tasks

## 1. Expert repack
- [x] 1.1 `repack::pack_fp4` / `unpack_fp4`: 16 K x 64 N tiles in lane order (FlashInfer `db2373490172`)
- [x] 1.2 `repack::e4m3_to_s0e5m3` / `scales_s0e5m3`; negative and NaN scales refused
- [x] 1.3 Golden bytes from FlashInfer's own functions on spark-5 (`tools/golden/fp4_pack_flashinfer.py`, torch 2.13.0+cu130); `tests/repack.rs`

## 2. Load plan (host)
- [x] 2.1 Per-layer expert buffers (gate_up / down weights and scales, `weight_scale_2`, `input_scale` for NVFP4 only)
- [x] 2.2 One buffer per other tensor; plan bytes plus on-disk bytes equal the checked bytes
- [x] 2.3 PLE modes; `PleTable::gather` for disk mode
- [x] 2.4 `Device` trait, staged load with parallel fill, read-back sampling and comparison
- [x] 2.5 `tests/load_plan.rs` against host memory

## 3. GPU load
- [x] 3.1 `device_weights::Gpu`, `load_hybrid`, `time_ple_gather`
- [x] 3.2 `sparkforge load-weights <dir> [--ple disk|resident] [--gather-tokens N]`, admission before the CUDA context
- [x] 3.3 Dense binding points hybrid checkpoints to `load-weights`

## 4. codex-review follow-ups on #18
- [x] 4.1 MTP input projections at their own format (fp:25aaaee34a54)
- [x] 4.2 MTP mixer norms priced as BF16 (fp:cf55fb8f7ebd)
- [x] 4.3 Only `weight_scale_2` at the table root (fp:347106d738b2)
- [x] 4.4 Live expert count compared before building per-expert entries (fp:d0f47806a572)

## 5. Verification
- [x] 5.1 `verify.sh` on macOS; `cargo test --release` and `cargo build --release --features cuda` on spark-5
- [x] 5.2 Both PLE modes on GB10 (below)

### Measured on spark-5, 2026-11-22

- **Hardware and software:** NVIDIA GB10 (cc 12.1), driver 580.178.04, CUDA 13.0, user `spark`.
- **Checkpoint:** `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at `6909a5bed089`, read from the Hugging Face cache.
- **GPU state:** the vLLM lane on the node was stopped for the window (19:28:50Z to 19:33:49Z restart-ready), so nothing else held GPU memory.

| | `--ple disk` | `--ple resident` |
|---|---|---|
| Check and plan | 13.63 s | 11.79 s |
| GPU buffers | 1,777 | 2,033 |
| Copied | 76.60 GB (71.34 GiB) | 105.40 GB (98.16 GiB) |
| Copy time | 16.62 s (4.61 GB/s) | 46.33 s (2.27 GB/s) |
| MemAvailable before / resident | 118.6 / 45.6 GiB | 118.6 / 18.5 GiB |
| Read back | 175,998 pieces, 5,866 MB, 8.10 s | 176,510 pieces, 8,013 MB, 24.20 s |
| Host RSS peak | 505 MB | 505 MB |

- **Read-back:** all sampled bytes matched the checkpoint in both modes.
- **PLE gather, disk mode:** 4,096 tokens x 16 rows (5.90 MB) were read into pinned memory with sequential `pread`s.

  | Run | Rows | Read time | Per token |
  |---|---|---|---|
  | 1 (seed 0x5eed) | new | 7,712 ms | 1,883 µs |
  | 2 (seed 0x5eed) | same as run 1, so cached | 160 ms | 39 µs |
  | 3 (seed 0xfeed) | new | 7,541 ms | 1,841 µs |

  Copying the gathered rows to the GPU took 0.15 ms per run.
- **Inference from these numbers:** an uncached row costs about 59 µs (32 reads per token, issued one at a time). Disk mode therefore needs parallel reads or prefetch before it can serve decode. That is follow-up work.
- **Copy rates:** disk mode's 4.61 GB/s is consistent with the experts being served from the page cache after the open's SHA256 pass. Not profiled: whether the resident run's lower rate comes from shards evicted from the page cache as GPU memory filled.

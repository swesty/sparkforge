# Design

## Context

See proposal.md (Why). The generated BF16 GEMV takes a raw weight pointer, reads rows with 16-byte
vector loads, and expects fused layouts: `qkv_proj` rows are q then k then v, and `gate_up_proj`
rows are gate then up. HF checkpoints store `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`
separately. GB10 shares one LPDDR5X pool between CPU and GPU, and its GPU uses ATS, so any CPU
virtual address (including a file mapping) is a valid GPU address.

## Goals / Non-Goals

**Goals:**
- Fail before touching the GPU if the checkpoint does not match the plan exactly.
- Decide zero-copy versus copying on measured GEMV bandwidth from real weights.

**Non-Goals:**
- Zero-copy for any linear (rejected on measurement, decision 2).
- Windows or non-Linux memory accounting.

## Decisions

1. **Parse the safetensors header ourselves and read only with positioned reads.** The header is an
   8-byte little-endian length and JSON (parsed with the existing `serde_json`). Every offset is
   checked against the data section with overflow-checked arithmetic before it becomes a file
   position; the non-empty tensors must tile the data section exactly; index entries must match the
   shard headers. No memory mapping: after decision 2 the mapping bought nothing, and a mapped file
   that another process truncates turns a read into SIGBUS, while `pread` returns an error. Each
   read also re-checks the shard's size. (The first revision used `memmap2`; codex-review cycle 1.)
2. **Copy into GPU memory; do not read the file mapping in place.** GB10 reports Addressing Mode ATS,
   so kernels *can* dereference the mmap address directly (no pinning, no copy). This was built
   (commit "safetensors loader with mapped (ATS) and copied GPU binding" in this change) and
   measured with `verify-weights` on spark-5 (GB10, 4 KB pages, CUDA 13.0, driver 580.178.04):

   | Llama 3.1 8B linear | Mapped, each layer touched once | Mapped, same tensor re-read | Device memory |
   |---|---|---|---|
   | `o_proj` (33.5 MB) | 120,391 µs (~0.3 GB/s) | 208.5 µs (161 GB/s) | 135-143 µs (225-245 GB/s) |
   | `down_proj` (117 MB) | 376,764 µs (~0.3 GB/s) | 734.6 µs (160 GB/s) | 475-493 µs (238-247 GB/s) |

   First touch costs about 3.3 µs per 4 KB page, and even warm reads lose about 35% of bandwidth,
   most likely to translation misses on 4 KB pages (inference, not profiled). Copying 14 GB is a
   one-time 10 s cost, so copying wins. Pinning (`cuMemHostRegister`) was not pursued: it locks the
   same bytes a copy allocates, and exhausting pinned memory freezes a Spark (fleet runbook).
3. **Alignment, recorded for any future zero-copy attempt.** safetensors pads the header length to 8
   bytes, so data starts at `8 + header_len`, which is 0 or 8 mod 16 per *file*; tensor offsets
   inside the data are 16-byte aligned. Llama 3.2 1B's single file and Llama 3.1 8B's shard 4 are
   8-byte aligned throughout, shards 1-3 are 16-byte aligned. The 16-byte-vector GEMV could not use
   misaligned files in place.
4. **Fused linears are packed once into a GPU allocation**, in the kernel's row order.
5. **Memory guard on Linux** reads `MemAvailable` from `/proc/meminfo` and refuses when the bytes
   to copy exceed `MemAvailable - 4 GiB`. Admission is serialized: an advisory `flock` on
   `$XDG_RUNTIME_DIR/sparkforge-gpu-admission.lock` (else a private `sparkforge-<uid>` temp
   directory) is held from the `MemAvailable` reading until every allocation is made and filled,
   so a concurrent load's reading already reflects this one (codex-review cycle 2). It coordinates
   one user's sparkforge processes; other programs, such as a resident vLLM, are seen only through
   `MemAvailable`. Measured on GB10: of two concurrent Llama 3.2 1B loads, the second printed that
   it was waiting, then both passed. The post-load figure is sampled inside the upload while the
   weights are resident.
6. **Binding is name-based and explicit.** For each layer the expected tensor names and shapes are
   generated from the `ModelSpec`; anything missing is an error naming the tensor and the expected
   shape; anything present but unused is listed as unexpected (some checkpoints carry extra
   buffers) without failing.
7. **Bulk loads use positioned reads in bounded pieces.** Copying straight out of an mmap faulted it
   in a few pages at a time: 13.98 GiB took 40.5-41.8 s (~0.35 GB/s) while `dd` read a shard at
   1.6 GB/s; `madvise(MADV_SEQUENTIAL)` made it slower (47.9 s). Reading with `pread` brought the
   load to 10.3-10.8 s with a whole-tensor buffer and 12.5 s with the final 64 MiB staging buffer.
8. **Host memory is bounded and budgeted.** Uploads stage at most 64 MiB; the f64 reference streams
   4 MiB chunks per thread (16 threads). The guard admits a load only if copied bytes plus that
   overhead leave 4 GiB of `MemAvailable`. `verify-weights` on Llama 3.1 8B peaks at 225 MB of host
   RSS (it previously built a 4.2 GB f64 copy of `lm_head`).

9. **Untrusted checkpoints fail closed** (codex-review cycle 2). Index shard names must be plain
   relative paths (no absolute paths or `..`), checked lexically so that Hugging Face cache
   snapshots, whose shards are symlinks to `../../blobs/<hash>`, still open. Repeated keys in
   `weight_map` or a shard header are errors (serde's maps would keep the last one). Shards are
   opened with `O_NONBLOCK` and must be regular files, so a FIFO cannot hang `open`.
10. **One checkpoint version per load.** Each shard's (dev, inode, length, mtime, ctime) is recorded
   at open, re-checked after the header read, and re-checked after every read. A match after a read
   means no write reached the file before those bytes were read, so a load cannot mix two versions.
   Limitation: timestamps have filesystem granularity (Linux uses a coarse clock, a few ms), so a
   write within the same tick as the open can go unseen. Hashing every tensor would close that gap
   at the cost of a second pass over 14 GB and was not pursued.

## Risks / Trade-offs

- [Loads are disk-bound when the page cache cannot hold the checkpoint (only ~4.6 GB stayed cached
  on spark-5 beside the resident vLLM engine)] -> ~1.4 GB/s from NVMe is acceptable for now; a
  pinned double-buffered reader could overlap reads with copies later.
- [Fused packing and copying need the model's full size in free memory] -> the 4 GiB guard refuses
  loads that would push the node into OOM territory; Llama 3.1 8B loaded with 11.3 GiB still free.

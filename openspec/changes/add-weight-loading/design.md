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
- Measure mapped versus copied GEMV bandwidth on real weights so the default is chosen on data.

**Non-Goals:**
- Zero-copy for fused linears (needs multi-pointer kernels).
- Windows or non-Linux memory accounting.

## Decisions

1. **Parse the safetensors header ourselves** (8-byte little-endian length, then JSON) with the
   existing `serde_json`, and map files with `memmap2`. The format is small and stable; a parser
   we own lets us check every byte range against the file length. Alternative: the `safetensors`
   crate. Rejected only to avoid a second JSON stack for a 30-line parser.
2. **Mapped mode passes the mmap address straight to kernels.** No `cuMemHostRegister`: pinning a
   15 GB file mapping locks it out of reclaim and, per the fleet runbook, exhausting pinned memory
   freezes a Spark instead of failing a CUDA call. ATS needs no registration. The device attribute
   `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` must be 1, otherwise mapped mode is refused.
   Alternative: managed memory (`cuMemAllocManaged` + read from file) is still a copy, so it is no
   better than copied mode for this purpose.
3. **Alignment fallback.** The GEMV needs 16-byte aligned rows. safetensors only guarantees
   element alignment, so a tensor whose data is not 16-byte aligned is copied even in mapped mode,
   and counted separately in the report.
4. **Fused linears are packed once into a GPU allocation** in both modes, in the kernel's row order.
5. **Memory guard on Linux** reads `MemAvailable` from `/proc/meminfo` and refuses when the bytes
   to copy exceed `MemAvailable - 4 GiB`. Mapped bytes are page cache and are not counted.
6. **Binding is name-based and explicit.** For each layer the expected tensor names and shapes are
   generated from the `ModelSpec`; anything missing is an error naming the tensor and the expected
   shape; anything present but unused is listed as unexpected (some checkpoints carry extra
   buffers) without failing.

## Risks / Trade-offs

- [Mapped pages can be evicted under memory pressure and re-faulted by the GPU, which would show
  up as GEMV slowdowns] -> `verify-weights` times every linear across all layers in both modes;
  the default mode is set from that measurement, not assumed.
- [ATS reads may use 4 KB host pages and lose bandwidth to TLB misses] -> same measurement; if
  mapped bandwidth is materially lower, copied stays the default and mapped stays available for
  memory-constrained loads.
- [Fused packing costs a copy of about 70% of a dense model's bytes] -> accepted for now; the
  multi-pointer GEMV is a follow-up.

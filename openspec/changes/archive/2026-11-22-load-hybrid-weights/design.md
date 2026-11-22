# Design

## Decision 1: experts are repacked at load time into FlashInfer's SM12x GEMV layout

The decode GEMV for the routed experts is BF16 activations against NVFP4 weights, at batch 1 to 4. Its layout is the one FlashInfer's `GemvBf16Fp4Sm12x` reads, prepared by `_prepare_cute_dsl_sm12x` in `flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py` at `db2373490172` (Apache-2.0):

- **Weights.** The weight is transposed to K-major (`[K/2, N]` bytes) and cut into tiles of 16 K by 64 N (512 bytes). Tiles are stored K-tile major. Each tile's bytes are gathered into 128 little-endian `u32`s in the order a warp's lanes load them.
- **Scales.** The per-16 E4M3 scales are transposed to `(K/16, N)`. Each is rewritten as S0E5M3: bits 14..7 of its FP16 encoding, which the kernel turns into a multiplier with one shift.

`repack.rs` re-derives both from the source. They are not taken on trust: `tools/golden/fp4_pack_flashinfer.py` ran FlashInfer's functions (copied verbatim, with the license notice) on spark-5 under torch 2.13.0+cu130, and `tests/repack.rs` matches its output byte for byte. The checkpoint's scales are linear, so FlashInfer's 128x4 unswizzle step does not apply. Negative and NaN scales are refused, since S0E5M3 has no sign bit.

Repacking happens on the host during the copy, so the GPU holds only the repacked bytes. Repacking moves bytes without changing their count, so the GPU bytes equal the checked bytes, which the plan asserts.

## Decision 2: one buffer per layer's experts, indexed by expert

The gate and up projections of an expert are fused into one `2I` x `H` GEMV (gate rows, then up rows), and the down projection is `H` x `I`. Each layer gets six buffers: `gate_up.weight`, `gate_up.scale`, `down.weight`, `down.scale`, `weight_scale_2` (`[E, 3]` F32) and `input_scale` (`[E, 3]` F32, NVFP4 only). Expert `x` sits at `x` times the per-expert size in every buffer. A routed kernel then needs one base pointer per buffer and the expert index from the router, not 512 pointers per layer. Gate and up keep separate global scales, so the fused GEMV applies the right one to each half.

Everything else is copied as stored, one buffer per tensor, about 1.8k allocations. No kernel consumes these yet, and they stay addressable by canonical name until the kernels fix their layouts.

## Decision 3: the PLE table is resident or on disk, chosen at load

The table holds 320,001,536 rows of 80 FP4 bytes plus 10 scale bytes, in 128 shards. A token reads one row per n-gram head (16 rows, 1,440 bytes).

- **Disk mode:** the shards are not copied. `PleTable::gather` reads rows straight from the checkpoint into a host buffer, and the scheduler copies them to the GPU. The global scale and the size tensors stay resident.
- **Resident mode:** copies the shards as stored.

The measured load and gather times are recorded in the tasks. With 16 sequential reads per token, cold disk reads cost about 1.9 ms per token. Disk mode therefore needs reads issued in parallel, or prefetching, before it serves decode. That is follow-up work, not part of this change.

## Decision 4: the load path is host code behind a `Device` trait

`load_plan::load` and `load_plan::verify` take a `Device` (alloc, write, read). `device_weights::Gpu` implements it with cudarc, and the tests implement it with host vectors. Every rule therefore runs in CI without a GPU: grouping, byte counts, staging in parts, parallel fill, read-back sampling, and error naming.

- **Fill:** pieces that fit the 64 MiB staging buffer are produced on 16 threads (`pread` plus repack), then copied in one write. Larger tensors are copied in staging-sized parts.
- **Admission:** taken before the CUDA context, as for dense loads, covering the copied bytes plus staging, fill and read-back scratch (256 MiB), and the gather buffers.

## Decision 5: read-back compares the GPU's bytes with the checkpoint

After the copy, every buffer up to 64 MiB is read back whole. A larger buffer is read back at its first and last piece: experts 0 and 511 of each expert buffer, or both 4 MiB ends of a long tensor.

- **Stored tensors** are compared byte for byte.
- **Packed weights** are unpacked back to the checkpoint's `[N, K/2]` rows before comparing, so an error in the repack shows up as well as an error in the copy.
- **Scales** are compared against the conversion of the checkpoint's bytes.

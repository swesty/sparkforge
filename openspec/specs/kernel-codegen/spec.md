# kernel-codegen Specification

## Purpose

Emits CUDA C++ in which every model dimension is a `constexpr` (`src/codegen.rs`), a kernel
manifest describing how to launch each entry point, and a cache key. Shapes baked in at compile
time let NVRTC unroll loops, keep per-lane state in registers, and drop bounds checks.

## Requirements

### Requirement: Decode kernels are generated with shapes as constexpr

Generation SHALL emit `model.cu` containing `rmsnorm`, `rope_kv_append`, `attn_decode_split`,
`attn_decode_combine`, `silu_mul`, and one `gemv_<linear>` for each decode GEMV whose status is
`generated` (no source is emitted for a `planned` GEMV), each in its own
namespace of `constexpr` dimensions, with `static_assert`s for shape preconditions (for example
`HIDDEN == THREADS * ITERS * 8`, `HD % 32 == 0`, `K % 8 == 0`).

#### Scenario: Different models, different code
- **WHEN** Llama 3.1 8B and Qwen3-30B-A3B are generated
- **THEN** their cache keys differ

### Requirement: Kernel status separates generated code from planned work

Each manifest entry SHALL carry entry name, file, grid, block, dynamic shared memory, argument names
and a status. A GEMV SHALL be `generated` only when its weights are BF16 (or the linear is `lm_head`
or `router`) and its staged activation vector (2*K bytes) fits the 101,376 B per-block
shared-memory limit; otherwise it SHALL be `planned`. Every prefill GEMM SHALL be `planned`.

#### Scenario: Shape too wide to stage
- **WHEN** a config gives `down_proj` K = 65536 (131072 B of staged activations)
- **THEN** `gemv_down_proj` is `planned`

### Requirement: Planned entries have no code and do not run

A `planned` entry SHALL keep its launch configuration in the manifest so later work can fill it in,
SHALL have no CUDA source, SHALL NOT be loaded by the runtime, and SHALL be listed as skipped by
verification. A planned prefill GEMM SHALL appear only as a tile struct in `gemm_tiles.cuh`.

#### Scenario: NVFP4 GEMVs are not claimed
- **WHEN** Qwen3-30B-A3B is generated with NVFP4 weights
- **THEN** `gemv_qkv_proj` is `planned` with no source in `model.cu`, and `gemv_lm_head` is `generated` with source

### Requirement: Generated launches stay within GB10 limits

Every block size SHALL be a multiple of 32 and at most 1024, and no `generated` kernel SHALL request
more dynamic shared memory than the 101,376 B per-block limit.

#### Scenario: Example models
- **WHEN** Llama 3.1 8B and Qwen3-30B-A3B are generated
- **THEN** every block size is a multiple of 32 and at most 1024

#### Scenario: Wide shape
- **WHEN** a config gives `down_proj` K = 65536
- **THEN** no `generated` kernel requests more than 101376 B of dynamic shared memory

### Requirement: RoPE rotates q and k and appends K/V at the position

`rope_kv_append` SHALL rotate q and k in the fused QKV buffer using the NeoX half-split layout, and
append the rotated k and the unmodified v to the caches at `pos`.

#### Scenario: Cache slot
- **WHEN** `rope_kv_append` runs at position p
- **THEN** cache slot p holds the rotated k and the unmodified v for every KV head

### Requirement: RoPE inverse frequencies match HF transformers

Inverse frequencies SHALL come from a host-computed `__constant__` table, computed in f32 exactly as
HF transformers does (`1 / theta^(2i/head_dim)`), with Llama 3 scaling applied when configured.

#### Scenario: Llama 3 scaling
- **WHEN** `inv_freq` is computed with llama3 scaling factor 8 and original context 8192
- **THEN** frequencies with wavelength under 2048 tokens are unchanged, those over 8192 are divided by 8, and those between are interpolated

### Requirement: RoPE matches the reference at every position

The angle SHALL be formed in f32 and its sine and cosine computed in double precision, because under
`--use_fast_math` `sincosf` becomes `__sincosf`, whose error grows with the angle.

#### Scenario: Long-context position
- **WHEN** Llama 3.1 8B RoPE runs at position 131071 on a GB10
- **THEN** its error against the f64 reference is at bf16 rounding level (measured 3.8e-3; it was 0.55 with fast-math `sincosf`)

### Requirement: Output is cached by content hash

The cache key SHALL be the first 8 bytes, as 16 hex characters, of SHA-256 over the target arch,
`model.cu` and `gemm_tiles.cuh`. `sparkforge emit` SHALL write `model.cu`, `gemm_tiles.cuh`,
`plan.json` and `manifest.json` into `<out>/<cache_key>/`.

#### Scenario: Emit layout
- **WHEN** `sparkforge emit configs/llama-3.1-8b.json --out build` runs
- **THEN** the four files exist under `build/<16-hex-char key>/`

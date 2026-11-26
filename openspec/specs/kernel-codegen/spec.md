# kernel-codegen Specification

## Purpose

Emits CUDA C++ in which every model dimension is a `constexpr` (`src/codegen.rs`), a kernel
manifest describing how to launch each entry point, and a cache key. Shapes baked in at compile
time let NVRTC unroll loops, keep per-lane state in registers, and drop bounds checks.

## Requirements

### Requirement: Decode kernels are generated with shapes as constexpr

Generation SHALL emit `model.cu` containing `rmsnorm`, `rope_kv_append`, `attn_decode_split`,
`attn_decode_combine`, `residual_add`, `embed_gather`, `argmax_advance`, `silu_mul`, and one
`gemv_<linear>` for each decode GEMV whose status is `generated` (no source is emitted for a
`planned` GEMV), each in its own namespace of `constexpr` dimensions, with `static_assert`s for shape
preconditions (for example `HIDDEN == THREADS * ITERS * 8`, `HD % 32 == 0`, `K % 8 == 0`).

#### Scenario: Different models, different code
- **WHEN** Llama 3.1 8B and Qwen3-30B-A3B are generated
- **THEN** their cache keys differ

### Requirement: Kernel status separates generated code from planned work

Each manifest entry SHALL carry entry name, file, grid, block, dynamic shared memory, argument names
and a status. A GEMV SHALL be `generated` when its weights are BF16 (or the linear is `lm_head` or
`router`) and its staged activation vector (2*K bytes) fits the 101,376 B per-block shared-memory
limit, or when its weights are NVFP4 or NVFP4 weight-only and N is a multiple of 64 and K a multiple
of 16; otherwise it SHALL be `planned`. Prefill GEMMs follow their own rule.

#### Scenario: Shape too wide to stage
- **WHEN** a config gives `down_proj` K = 65536 (131072 B of staged activations)
- **THEN** `gemv_down_proj` is `planned`

#### Scenario: NVFP4 GEMVs are generated
- **WHEN** Qwen3-30B-A3B is generated with NVFP4 weights
- **THEN** `gemv_qkv_proj`, `gemv_o_proj`, `gemv_expert_gate_up` and `gemv_expert_down` are `generated` with source in `model.cu`, and request no dynamic shared memory

#### Scenario: NVFP4 shape that does not tile
- **WHEN** an NVFP4 linear has N = 2016 (not a multiple of 64)
- **THEN** its GEMV is `planned` with no source

### Requirement: Planned entries have no code and do not run

A `planned` entry SHALL keep its launch configuration in the manifest so later work can fill it in,
SHALL have no CUDA source, SHALL NOT be loaded by the runtime, and SHALL be listed as skipped by
verification. A planned prefill GEMM SHALL appear only as a tile struct in `gemm_tiles.cuh`.

#### Scenario: NVFP4 GEMVs are not claimed
- **WHEN** a dense NVFP4 config has `hidden_size` 2016, so `o_proj` and `down_proj` have N = 2016
- **THEN** `gemv_o_proj` and `gemv_down_proj` are `planned` with no source in `model.cu`, and `gemv_lm_head` is `generated` with source

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
append the rotated k and the unmodified v to the caches at the position it reads from device memory.
`attn_decode_split` SHALL attend over positions 0 through that same position.

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

### Requirement: Hybrid plans generate no kernels until their op classes exist

For a hybrid plan, code generation SHALL emit source only for op classes whose kernel exists: the
NVFP4 decode GEMVs that tile. Every other op class SHALL be listed as `planned` with no source,
because the dense kernels assume full-head RoPE, dense attention and uniform weights. Its cache key
SHALL include the plan, so two hybrid plans never share an output directory.

#### Scenario: Flash Next emit
- **WHEN** `sparkforge emit configs/qwen3.8-flash-next-nvfp4.json` runs
- **THEN** `gemv_expert_gate_up` and `gemv_expert_down` are `generated`, every other op class is `planned`, and the source contains exactly those two `__global__` functions

#### Scenario: Hybrid without NVFP4 experts
- **WHEN** a hybrid config whose routed experts are MXFP8 is generated
- **THEN** every op class is `planned` and the source contains no `__global__` function

### Requirement: The NVFP4 GEMV computes a W4A16 product from the load-time tile layout

An NVFP4 GEMV SHALL read weights in the tile layout of the weight loader (16 K x 64 N tiles of 512 bytes) and per-16 scales as S0E5M3 `(K/16, N)` bytes. Each output row SHALL be the f32 sum over K of E2M1 value times its decoded scale and the row's global scale (applied before the activation multiply) times the BF16 activation, rounded once to BF16. Activations SHALL NOT be quantized and no shared memory SHALL be used.

#### Scenario: Zero and subnormal scales
- **WHEN** a weight's E4M3 scale bytes include 0x00 and the subnormals 0x01 and 0x07
- **THEN** those blocks contribute exactly 0, 2^-9 and 7 x 2^-9 times their codes, as the f64 reference does

#### Scenario: Wide K
- **WHEN** an NVFP4 linear has K = 65536
- **THEN** its GEMV is `generated` with no dynamic shared memory

### Requirement: Fused rows keep their own global scales

A GEMV over several stacked checkpoint tensors SHALL apply each tensor's global scale (`weight_scale_2`) to that tensor's rows only. Fused segments SHALL be q, k, v for `qkv_proj` and gate, up for `gate_up_proj`, `expert_gate_up` and `shared_gate_up`. A grouped expert GEMV SHALL read its global scales from an `[E, 3]` (gate, up, down) table.

#### Scenario: Different gate and up scales
- **WHEN** an expert's gate global scale is 0.01 and its up global scale is 0.02
- **THEN** rows below the expert intermediate size use 0.01 and the remaining rows use 0.02

#### Scenario: One stored expert
- **WHEN** a routed `expert_down` linear has one stored expert and one slot
- **THEN** it still reads the `[E, 3]` table, at column 2 (down), not column 0 (gate)

### Requirement: Split-K partials merge deterministically in one launch

When K is split across blocks, each split SHALL write f32 partials to a workspace, and the last split to arrive for an N tile SHALL sum all of that tile's partials in split order 0, 1, ... and write the output in the same launch. It SHALL then reset the tile's arrival counter to zero. Counters SHALL start zeroed when the workspace is allocated.

#### Scenario: Replay gives identical bits
- **WHEN** the same split-K NVFP4 GEMV is launched twice on the same inputs without clearing its counters
- **THEN** both outputs are bitwise identical

### Requirement: A grouped NVFP4 GEMV covers all active experts in one launch

For a routed-expert linear, one launch SHALL compute every active slot (the model's experts per token). It SHALL read slot s's expert id from a device array, its weights and scales at that expert's offset in the per-layer grouped buffers, and write the slot's N outputs at row s of a `[slots, N]` output. The gate/up GEMV SHALL read one shared activation vector; the down GEMV SHALL read one vector per slot.

#### Scenario: Unordered experts
- **WHEN** a Flash Next `gemv_expert_gate_up` runs with expert ids 17, 3, 511, ... (10 distinct ids in no order)
- **THEN** output row s equals expert id[s]'s product, for each of the 10 slots

### Requirement: RMSNorm and the residual add round like HF transformers

`rmsnorm` SHALL round `x * rsqrt(mean(x^2) + eps)` to bf16 before multiplying by the weight, as HF's
`LlamaRMSNorm` does. `residual_add` SHALL add a bf16 vector into the hidden state in place, summing
in f32 and rounding to bf16.

#### Scenario: Reference agreement
- **WHEN** `verify` checks `rmsnorm` against a reference that rounds before the weight
- **THEN** it passes (measured 3.80e-3 on Llama 3.1 8B), and on every element where rounding before the weight and rounding once give different bf16 values, at least 90% match the first (measured 4138 of 4138)

#### Scenario: Old rounding order
- **WHEN** `rmsnorm` rounds `x * r * w` once at the end
- **THEN** `verify` fails it (measured 0 of 4138 elements matching), although the tolerance alone would pass it

### Requirement: The token is looked up and chosen on the device

`embed_gather` SHALL copy the embedding row named by a token id in device memory. `argmax_advance`
SHALL write the index of the greatest logit (the lowest index on ties; NaN never wins, so negative
infinity can; all NaN gives 0) as the next token, record it at the current position of an output
buffer, and increment the position. A step whose token comes from the previous step SHALL need an
embedding row for every vocabulary index.

#### Scenario: Ties and NaN
- **WHEN** the greatest logit appears twice and NaNs come before it
- **THEN** the lower index wins, it is recorded at the position, and the position advances by one

#### Scenario: NaN and negative infinity
- **WHEN** index 0 is NaN and every other logit is negative infinity
- **THEN** index 1 wins

### Requirement: BF16 prefill GEMMs are generated

A prefill GEMM SHALL be `generated` when its linear's weights are BF16 and its tile divides N and K,
and `planned` otherwise.

#### Scenario: Prefill GEMMs by format
- **WHEN** Llama 3.1 8B (BF16) and Qwen3-30B-A3B with NVFP4 weights are generated
- **THEN** Llama's four `gemm_<linear>` entries are `generated` with source in `model.cu`, and every NVFP4 linear's GEMM is `planned` with no source

### Requirement: The prefill GEMM computes A times W transposed with mma.sync

`gemm_<linear>(A, W, C, M)` SHALL compute C[M][N] = A[M][K] · W[N][K]ᵀ in bf16 with f32 accumulation,
N, K and the tile `constexpr` and the prompt length M an argument: a `cp.async` pipeline of the
planned stages into 16-byte-padded shared memory, `ldmatrix`, `mma.sync m16n8k16`, tiles in the
planned `group_n` order, rows past M zero-filled on load and not stored.

#### Scenario: Ragged prompt
- **WHEN** a GEMM runs with M 37 rows short of the planned length
- **THEN** the rows below M match the reference and the rows past M are left as they were

### Requirement: Prefill GEMMs launch through a checked host call

`prefill::gemm` SHALL launch a generated prefill GEMM for M prompt rows with a grid of
ceil(M / BM) x N / BN, after checking that A holds M x K elements, W exactly N x K, C M x N, A and W
are 16-byte aligned and C 4-byte aligned. A launch with a larger grid, such as the manifest's for the
planned M, SHALL leave its extra blocks idle.

#### Scenario: Misaligned activations
- **WHEN** A is a view one element past an aligned allocation
- **THEN** `prefill::gemm` refuses it before launching

#### Scenario: Manifest grid for a shorter prompt
- **WHEN** a Llama GEMM is launched with the manifest's grid (for M = 2048) and M = 1024
- **THEN** the output matches the reference and rows past 1024 are untouched (without the in-kernel exit, the same launch faulted with an illegal address)

### Requirement: Prefill attention is causal FlashAttention-2 on mma.sync

`attn_prefill(qkv, k_cache, v_cache, out, M, start, scale_log2)` SHALL compute causal attention for M prompt rows, row r at position start + r over cache positions 0..start + r. It SHALL read Q from the qkv rows and K/V from the decode cache. It SHALL use the planned tiles, Q fragments in registers, a `cp.async` K/V pipeline, `mma.sync m16n8k16` for Q·Kᵀ and P·V, and an fp32 exp2 online softmax.

#### Scenario: Generated for dense models
- **WHEN** a dense BF16 plan is generated
- **THEN** `attn_prefill` is `generated` with the plan's block and dynamic shared memory

#### Scenario: Later chunk
- **WHEN** 128 rows run from position 1000 over a cache that holds positions 0 to 1127
- **THEN** each row matches f64 causal attention over its own prefix

### Requirement: Row-wise kernels run a whole prompt

`rope_kv_append`, `residual_add`, `silu_mul` and `embed_gather` SHALL process row `blockIdx.y`, and `rmsnorm` row `blockIdx.x`. `rope_kv_append` SHALL place row r at position `*pos_p` + r. A decode step launches one row; prefill launches one per prompt token.

#### Scenario: Prompt rows
- **WHEN** `rope_kv_append` runs over M rows with `*pos_p` = s
- **THEN** cache slots s to s + M - 1 hold each row's rotated k and its v, rotated as a decode step at that position rotates them

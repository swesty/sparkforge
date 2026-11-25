# Design

## Context

See proposal.md for the motivation. These parts of the current code shape the approach:

- **`src/repack.rs` (SHA-791)** already produces the operand layout.
  - Weights go into 16 K x 64 N tiles of 512 bytes, as `(K/16, N*2)` `u32`s. `tile_byte(i)` gives the lane order.
  - Scales go to S0E5M3 `(K/16, N)`.
  - E4M3 subnormals are renormalized: `e4m3_to_s0e5m3(0x01)` is FP16 2^-9. Zero stays zero, and negative or NaN scale bytes are refused.
  - The repack matches FlashInfer's own prep byte for byte (`tests/golden/fp4_pack_flashinfer_db23734.json`).
- **`load_plan::expert_buffers`** stores each layer's experts in expert order:
  - `gate_up.weight`: per expert, gate rows then up rows, one `2I` x `H` matrix;
  - `gate_up.scale`, `down.weight`, `down.scale`: the same order;
  - `weight_scale_2`: `[E, 3]` F32, in the order gate, up, down.
- **The BF16 GEMV** stages x in shared memory and puts one warp on 1, 2 or 4 rows. `plan_gemv` sizes its grid, and `codegen::generate` emits it only for BF16.
- **Hybrid plans:** `codegen::generate` returns early for them and emits nothing.
- **`verify::gemvs`** times a GEMV over rotated weight copies. `token_estimate_us` multiplies the expert GEMVs by `experts_per_tok`.
- **The instruction is supported:** CUDA 13.0.88 ptxas on spark-5 accepts `cvt.rn.f16x2.e2m1x2` for `sm_121a`, as one `F2FP.F16.E2M1.UNPACK_B` per byte (checked 2026-11-25 with a 20-line kernel and `cuobjdump -sass`).

## Goals / Non-Goals

**Goals:**
- **One template for two shapes.** One CUDA template serves both the dense and the grouped NVFP4 GEMV, with every shape, split and scale-segment boundary a `constexpr`.
- **Host-side check of the index map.** The kernel's index arithmetic has a host mirror that `cargo test` checks against the repack, so a lane-mapping mistake fails in CI rather than only on a GB10.

**Non-Goals:**
- Autotuning the split count. The rule is derived once from `target.rs`; SHA-752 owns autotuning.
- A fallback for shapes that do not tile. Those GEMVs stay `planned` with no source, as the spec requires.

## Decisions

### 1. W4A16 streaming kernel: no shared memory, no tensor cores

At M = 1 the GEMV reads 0.5625 bytes per weight and does two FMAs per weight, so DRAM sets the speed and math is nearly free. FlashInfer's comment on `GemvBf16Fp4Sm12x` states the same trade:
- A shared-memory pipeline feeding `mma.sync` costs shared memory and so caps resident warps.
- Streaming straight to registers hides latency with warp count alone.

The kernel converts each weight byte with `cvt.rn.f16x2.e2m1x2` to an f16 pair, widens it to f32, and accumulates in f32 against BF16 activations, also widened to f32. Activations are read with `__ldg`: 4 to 10 KB per vector, which stays in L1 and L2. Nothing is staged, so K has no shared-memory limit.

- **Alternative: W4A4 with `mma.sync kind::mxf4nvf4`, quantizing x per 16 using the checkpoint's `input_scale`.** Rejected for decode:
  - it adds an activation quantization step and its error;
  - tensor cores do not help a DRAM-bound M = 1 product.

  If the golden diff (SHA-773) shows the fleet engine's numerics need W4A4 to agree, the GEMV can add an activation-quantize step later without changing the layout.

### 2. The lane map is the inverse of `repack::tile_byte`

**Assignment.** A CTA is 8 warps, and warp w owns N tile `nt = blockIdx.x * 8 + w`. Per K tile, lane L makes one 16-byte load: `u32`s `4L .. 4L+3` of the tile's 128.

**Derivation from `tile_byte`.** `u32` number p belongs to pack lane `(p/2) % 32` and N-warp `p/64`.
- The rows: lane L's 4 `u32`s cover rows `base_n + {0, 16, 32, 48}`, with `base_n = (L/16)*8 + (L%16)/2`.
- Byte b of the lane's `u32` j:
  - holds packed K row `k_half = 2*(L%2) + (j>>1) + 4*(b&1)`, that is, K values `2*k_half` (low nibble) and `2*k_half + 1`;
  - belongs to row slot `p = 2*(j&1) + (b>>1)`.
- Each lane therefore sums 8 of a row's 16 K values per tile, and its xor-1 partner sums the other 8. One `__shfl_xor_sync(.., 1)` after the K loop completes each of the 4 rows.

`repack.rs` gains `lane_elem(lane, j, b) -> (row_slot, k_pair)`. A test asserts that, over every (lane, j, b), it visits every (row, K pair) of a tile exactly once and agrees with `tile_byte`. The CUDA template spells out the same three formulas, with comments naming the Rust function.

### 3. Scales: one decode per K tile and row, exact for every accepted byte

Each lane reads 4 S0E5M3 bytes per K tile, one per row: `SF[kt*N + nt*64 + base_n + 16p]`. It decodes each as `__half2float(__ushort_as_half(byte << 7))` and multiplies it into the tile's partial dot before adding to the row accumulator.

The decode is exact for every byte the repack emits:
- normal E4M3 values;
- renormalized subnormals;
- zero.

So the ticket's "reject subnormal or zero scale bytes" is unnecessary; SHA-791 already settled the subnormal case by renormalizing. The largest byte, 0xbe (448), shifted left by 7 is 0x5f00, so the f16 sign bit stays clear.

### 4. Global scales by constexpr row segment

**Per-row scale.** Each output row is multiplied by `gscale[slot_base + SEG_COL[seg(row)]]`, where `seg(row)` compares the row with constexpr segment ends.
- **Dense:** `slot_base` is 0, and the segments are the stacked tensors:
  - `qkv_proj`: q, k, v;
  - `gate_up_proj`, `shared_gate_up`: gate, up;
  - every other linear: one segment.
- **Grouped:** `slot_base` is `expert * 3`, which indexes SHA-791's `[E, 3]` table. `expert_gate_up` uses columns 0 and 1; `expert_down` uses column 2.

The planner derives segment ends from the model's own dimensions. Every boundary in both example models is a multiple of 64, so no warp's tile mixes rows of two tensors in a way that matters. The per-row compare also makes a boundary inside a tile correct, at no extra cost.

- **Alternative: one scale per output row in a separate array (N floats).** Rejected: it adds a buffer the loader does not have, for no gain.

### 5. Split-K with a last-arriving-warp merge in the same launch

**Split rule.** `grid.y` holds S splits; S follows the planning spec's rule.
- The rule is FlashInfer's `_bf16_fp4_gemv_knee_split`, re-derived for GB10: 20 warps per SM on 48 SMs, a cap of 0.95 of 6 CTAs of 256 threads per SM, and at least 4 K tiles per split.
- The slot count is added, so grouped launches count every slot's warps.
- S is then reduced so that `ceil(K_TILES / ceil(K_TILES / S))` splits all hold work.

**Merge.** A warp that finishes its split:
1. writes its 64 row partials to `partial[(slot*S + split)*N + row]`;
2. calls `__threadfence()`;
3. has lane 0 `atomicAdd` the tile's counter `count[slot*N_TILES + nt]`, broadcasting the old value.

The warp that sees `S - 1` is last. It calls `__threadfence()`, reads the S partials with `__ldcg` in split order 0..S-1, applies the global scale, writes BF16, and stores 0 to the counter.

**Determinism.** The sum order is fixed regardless of which warp arrives last, so results are bitwise reproducible. The counter returns to zero within the launch, so graph replay needs no memset. With S = 1, the warp writes the output directly and touches neither buffer.

- **Alternative: FlashInfer's separate fixed-order reduce kernel.** Rejected: it costs one launch per GEMV, and the inference-engine design budgets at most 370 launches per step.
- **Alternative: atomic f32 adds into the output.** Rejected: not deterministic.

A per-warp counter, rather than a per-CTA one, keeps the kernel free of `__syncthreads()`. A warp with no tile (N tiles not a multiple of 8) still returns early, as in FlashInfer.

### 6. Grouped variant: slots on grid.z, expert ids on the device

`grid.z` is the slot count (`experts_per_tok`), and slot s reads `experts[s]`.
- **Inputs:**
  - weights at `W + expert * (N*K/2)`;
  - scales at `SF + expert * (N*K/16)`;
  - x at `x + s * X_STRIDE`, where `X_STRIDE` is 0 for gate/up (one hidden vector) and K for down (one SiLU output per slot).
- **Output:** `y + s * N`.

The kernel trusts the ids: range-checking them is the router's contract (SHA-758), and the verify inputs are always in range.

**One template.** The dense kernel is the same template with one slot and `EXPERT_STRIDE` 0, and it ignores `experts`. Both kernels therefore share one argument list: `W, SF, gscale, experts, x, y, partial, count`. The manifest lists it, and `verify` passes null for the arguments a variant does not use.

### 7. Loads in flight

A lane's weight load is one 16-byte `__ldcs` per K tile; streaming loads keep weights out of L1, as in the BF16 GEMV. The K loop over the split's constexpr tile count is unrolled 4 deep, so each lane has 4 independent 16-byte loads in flight.

Little's law sizes it. At 20 warps per SM, 4 loads in flight per lane is 40 KB per SM. The survey estimates that 3 to 5 KB per SM saturates LPDDR5X, so the unroll gives margin against the long-latency tail. The unroll depth is a constexpr that `verify --bench` can test. If the 225 GB/s bar is missed, the first thing to try is changing the unroll, before changing the split rule.

**Measured on spark-5 (2026-11-25).**
- **Build:** ptxas gives every NVFP4 entry 40 registers and no spills, so all 48 warps per SM fit.
- **Sweep:** unroll 2, 4 and 8 against 20, 28 and 40 target warps per SM, two runs each.
  - Flash Next `gemv_expert_gate_up` measured 214-221 GB/s in every combination but one, an outlier at 200.
  - `gemv_expert_down` measured 212-221 GB/s, with one outlier at 186.
- **Scale experiments:** neither touches the result, so the split count, the unroll and the scale layout do not limit this shape.
  - A tile-major scale layout gave 217-219 GB/s.
  - Dropping the scale loads cut the time only in proportion to the bytes they read.
- **Defaults kept:** unroll 4, 20 warps per SM.
- **Against the BF16 GEMV:** the shape runs at about 91% of the bandwidth of the BF16 `gemv_lm_head` in the same session (238-239 GB/s).

### 8. Planner and codegen integration

- **Planner.** `GemvPlan` gains optional fields, which are skipped in JSON when `None`, so BF16 plans and the golden files are unchanged:
  - `splits`;
  - `slots`;
  - `grid_y`, `grid_z`;
  - `workspace_bytes`, `counter_bytes`;
  - `segments`, each as a row end and a global-scale column.
- **Plan builder.** For FP4 linears, `build` calls a new `plan_gemv_fp4`. The hybrid note changes from "every decode op is planned" to naming the generated NVFP4 GEMVs.
- **Codegen.**
  - **Dense path:** emits `GEMV_FP4` for each FP4 GEMV whose shape tiles.
  - **Hybrid path:** still lists every op class, but marks a tiling NVFP4 GEMV `generated` and appends its source.
  - **Manifest:** `grid` becomes `[x, y, z]` from the plan. The verifier and the runtime already take a 3-D grid.

### 9. Verification inputs and reference

**Inputs.** For an `n` x `k` linear, `verify` draws, in checkpoint format:
- E2M1 code bytes `[n, k/2]`, uniform over all 256 values; E2M1 has no NaN or Inf, so every code is valid;
- E4M3 scale bytes `[n, k/16]`: mostly normal values with exponent field 5 to 9, plus forced 0x00, 0x01 and 0x07 bytes at fixed positions;
- one f32 global scale per segment, with distinct values: 0.75, then 1.5, then 2.25.

**Packing.** `repack::pack_fp4` and `scales_s0e5m3` turn these into the GPU operands.

**Reference.** In f64, it decodes each code with a 16-entry E2M1 table and each scale with an E4M3 decoder written independently of the S0E5M3 path. It never reads repacked bytes, so the reference checks the repack and the kernel together.

**Grouped inputs.** The check uses 2 x slots experts, with ids taken from a seeded shuffle and kept unsorted. For timing, all `num_experts` copies are allocated. Each launch draws distinct random ids, written in advance into one device array per launch. For Flash Next, `expert_gate_up` takes 0.94 GB and `expert_down` 0.47 GB.

**Determinism check.** The kernel is launched, the output is overwritten with a 0xFFFF NaN sentinel, and the kernel is launched again. The two outputs must be bitwise equal.

**Host emulation in CI.** A host test runs a Rust emulation of the kernel on small shapes and compares it with the f64 reference. The emulation is a per-lane walk using `lane_elem`, with partials summed in split order. The shapes:
- N 128, K 64, S 3;
- grouped with 3 slots and gate/up segments.

## Risks / Trade-offs

- **[Split-K partial traffic]** At S = 29 (Qwen3 `o_proj`), the workspace reads and writes `29 x 2048 x 4 x 2` = 475 KB beside 4.7 MB of weights, a 10% overhead. → The cap of at least 4 K tiles per split bounds it. `verify --bench` measures it, and the split rule can be retuned in the plan without changing the kernel.
- **[Small grouped down GEMV]** Flash Next `expert_down` is 9.2 MB per layer. It may fall short of 225 GB/s because of launch and tail effects. → It is below the 10 MB bar M2 gates. Its number is recorded, and fusing it with SiLU is M5 work.
- **[W4A16 vs. the fleet's numerics]** W4A16 does not reproduce W4A4 rounding, if that is what the fleet uses. → This is noted for the golden diff (SHA-773), and the kernel layout does not depend on the answer.
- **[Memory ordering in the merge]** A missing fence would read stale partials on rare interleavings. → The fence-then-atomic and atomic-then-fence pattern is the CUDA programming guide's threadfence reduction. The NaN-sentinel double launch catches a counter that is not reset. Reads use `__ldcg`, which bypasses L1.
- **[Untrusted expert ids]** An out-of-range id reads outside the buffer. → This is the router's contract (SHA-758). The kernel stays free of per-slot checks, which would hide a router bug as silent zeros.

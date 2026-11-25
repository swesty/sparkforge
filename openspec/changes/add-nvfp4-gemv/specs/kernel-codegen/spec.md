## MODIFIED Requirements

### Requirement: Kernel status separates generated code from planned work

Each manifest entry SHALL carry entry name, file, grid, block, dynamic shared memory, argument names
and a status. A GEMV SHALL be `generated` when its weights are BF16 (or the linear is `lm_head` or
`router`) and its staged activation vector (2*K bytes) fits the 101,376 B per-block shared-memory
limit, or when its weights are NVFP4 or NVFP4 weight-only and N is a multiple of 64 and K a multiple
of 16; otherwise it SHALL be `planned`. Every prefill GEMM SHALL be `planned`.

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

## ADDED Requirements

### Requirement: The NVFP4 GEMV computes a W4A16 product from the load-time tile layout

An NVFP4 GEMV SHALL read weights in the tile layout of the weight loader (16 K x 64 N tiles of 512 bytes) and per-16 scales as S0E5M3 `(K/16, N)` bytes. Each output row SHALL be the f32 sum over K of E2M1 value times its decoded scale (an exact f16 product, formed before the activation multiply) times the BF16 activation, multiplied by the row's global scale and rounded once to BF16. Activations SHALL NOT be quantized and no shared memory SHALL be used.

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

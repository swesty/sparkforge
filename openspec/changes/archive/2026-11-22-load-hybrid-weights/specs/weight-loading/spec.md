## REMOVED Requirements

### Requirement: Only dense BF16 checkpoints load into GPU memory

**Reason**: Hybrid checkpoints now load into GPU memory through `sparkforge load-weights`; only the dense binding still refuses them.
**Migration**: Replaced by "The dense binding accepts only dense BF16 checkpoints", which keeps every refusal for the dense binding, and "Hybrid checkpoints load into GPU memory".

## ADDED Requirements

### Requirement: The dense binding accepts only dense BF16 checkpoints

Binding for the dense GEMV path MUST fail, before any GPU work, for a hybrid-layout checkpoint (pointing to `sparkforge load-weights`), for one whose config declares mixed precision, and for one whose planned tensors are not BF16, naming the architecture or the tensor. A format override used for planning MUST NOT lift this refusal.

#### Scenario: Flash Next checkpoint
- **WHEN** a Qwen3.8 Flash Next checkpoint is bound for the dense path
- **THEN** binding fails stating that hybrid checkpoints load with `sparkforge load-weights`

#### Scenario: Override used for planning
- **WHEN** a dense MIXED_PRECISION config is parsed with a BF16 override and its checkpoint is bound
- **THEN** binding fails stating that mixed-precision checkpoints do not load yet

#### Scenario: Quantized tensors behind a BF16 config
- **WHEN** a config without `quantization_config` is bound to a checkpoint whose `o_proj` is F8_E4M3 with a scale tensor
- **THEN** binding fails before GPU work, naming the tensor and its dtype

### Requirement: Hybrid checkpoints load into GPU memory

`sparkforge load-weights <dir> [--ple disk|resident]` (`--features cuda`) SHALL check every tensor of a hybrid checkpoint first, then copy the checked tensors into GPU memory, one allocation per buffer of the load plan. The copied bytes plus the PLE bytes left on disk MUST equal the checked bytes. The command SHALL report the bytes copied, the time, and `MemAvailable` before the copy and with the weights resident.

#### Scenario: Flash Next on GB10, PLE table on disk
- **WHEN** `load-weights --ple disk` loads the reference checkpoint (revision 6909a5bed089) on spark-5 with nothing else on the GPU
- **THEN** 76.60 GB (71.34 GiB) in 1,777 buffers are copied and read back matching the checkpoint, and 28.80 GB of table shards stay on disk

#### Scenario: Flash Next on GB10, PLE table resident
- **WHEN** `load-weights --ple resident` loads the same checkpoint
- **THEN** 105.40 GB (98.16 GiB) in 2,033 buffers are copied and read back matching the checkpoint

### Requirement: Hybrid loads are admitted before CUDA starts

A hybrid load's admission (lock and memory guard) MUST cover the copied bytes plus staging, fill, read-back and gather scratch, MUST be taken before the CUDA context is created, and MUST be held until every allocation of the run is made.

#### Scenario: Not enough memory
- **WHEN** the load would leave less than the 4 GiB headroom, as with the spark-5 vLLM lane running (25.45 GiB available)
- **THEN** it is refused before any CUDA call, stating the GiB to copy and `MemAvailable`

### Requirement: Routed experts are repacked for the decode GEMV

Each layer's NVFP4 routed experts SHALL load as six buffers indexed by expert: `gate_up.weight` (gate rows then up rows, `2I` x `H`), `gate_up.scale`, `down.weight` (`H` x `I`), `down.scale`, `weight_scale_2` (`[E, 3]` F32: gate, up, down), and `input_scale` (`[E, 3]` F32) when activations are quantized.

#### Scenario: One loaded expert
- **WHEN** the tiny checkpoint is loaded into host memory
- **THEN** expert 2's `gate_up.weight` block unpacks to its gate rows followed by its up rows, and each scale equals the S0E5M3 conversion of the matching checkpoint byte

#### Scenario: Weight-only MTP experts
- **WHEN** the MTP layer's experts are W4A16 NVFP4
- **THEN** they load without an `input_scale` buffer


### Requirement: Expert weights use FlashInfer's SM12x GEMV layout

Packed expert weights SHALL be tiled 16 K by 64 N and lane-ordered, and their per-16 scales stored as `(K/16, N)` S0E5M3, matching FlashInfer `db2373490172`'s SM12x BF16 x FP4 GEMV preparation byte for byte. Shapes that do not tile, and negative or NaN scales, MUST fail naming the buffer.

#### Scenario: FlashInfer golden bytes
- **WHEN** the 128 x 64 and 64 x 32 golden matrices are repacked
- **THEN** the weights and scales equal the bytes FlashInfer's own functions produced, and unpacking gives back the input

#### Scenario: Every scale byte
- **WHEN** each of the 126 finite non-negative E4M3 bytes is converted
- **THEN** the result equals bits 14..7 of the value's FP16 encoding

#### Scenario: Negative scale
- **WHEN** an expert's scale byte has its sign bit set
- **THEN** the load fails naming `gate_up.scale` and saying the scale is negative

### Requirement: The PLE table is resident or read from disk

In resident mode the PLE table shards SHALL be copied as stored. In disk mode they MUST NOT be copied, while the table's global scale and size tensors stay resident. Global row `r` SHALL be read from shard `r / rows_per_shard`, at row `r % rows_per_shard`, into one region per shard tensor. A row beyond the table MUST fail. Disk mode SHALL time a gather of the rows of `--gather-tokens` tokens (one row per n-gram head) into pinned memory, and their copy to the GPU.

#### Scenario: Tiny table on disk
- **WHEN** the tiny checkpoint is planned in disk mode
- **THEN** 540 bytes of shards (4 x 15 rows x 9 bytes) stay on disk, and rows 0, 14, 15, 59 and 31 gather to the bytes of their shards

#### Scenario: Out of range
- **WHEN** row 60 of the 60-row table is gathered
- **THEN** the gather fails naming the row and the table size

### Requirement: Loaded hybrid weights are read back and compared

After the copy, every buffer up to 64 MiB SHALL be read back whole. Each larger buffer SHALL be read back at its first and last piece, with a stored tensor over 8 MiB read at its first and last 4 MiB. Stored tensors MUST match byte for byte. Packed weights MUST unpack to the checkpoint's rows, and scales MUST equal the conversion of the checkpoint's bytes. A mismatch MUST fail naming the buffer and piece.

#### Scenario: One flipped byte
- **WHEN** one byte of the last expert's `down.weight` differs on the device
- **THEN** read-back fails naming `layers.3.mlp.experts.down.weight` piece 3

#### Scenario: Sampling large buffers
- **WHEN** the thresholds are 1 KiB
- **THEN** an expert buffer is read back at experts 0 and 3 only, and the 16 KiB embedding at its first and last KiB

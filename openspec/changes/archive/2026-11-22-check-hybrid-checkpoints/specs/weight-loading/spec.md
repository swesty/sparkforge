## REMOVED Requirements

### Requirement: Hybrid and mixed-precision checkpoints are refused

**Reason**: Hybrid checkpoints are now checked tensor by tensor by `sparkforge weights`; only loading them into GPU memory is still refused.
**Migration**: Replaced by "Hybrid checkpoints are checked tensor by tensor" and "Only dense BF16 checkpoints load into GPU memory", which keeps this requirement's refusals for GPU binding.

## ADDED Requirements

### Requirement: Hybrid checkpoints are checked tensor by tensor

For a hybrid checkpoint, `sparkforge weights` SHALL check every tensor before any GPU work: each tensor the plan needs exists once, with the dtype and shape of the format its module is priced at, and every other tensor belongs to the skipped vision tower. PLE table shard rows SHALL come from the checkpoint's per-head table sizes. Any mismatch MUST fail naming the tensor and the expected and found values.

#### Scenario: Flash Next checkpoint
- **WHEN** `sparkforge weights` opens the Qwen3.8 Flash Next NVFP4 checkpoint (revision 6909a5bed089) on spark-5
- **THEN** 301,260 tensors (105.40 GB) match the plan, the 333 vision-tower tensors are skipped, the PLE shards hold 2,500,012 rows, and the plan prices 105.40 GB of resident text weights

#### Scenario: Wrong dtype or shape
- **WHEN** a GDN projection's MXFP8 scale is F8_E4M3 and an expert's packed weight has twice its columns
- **THEN** the check fails naming both tensors with the expected and found dtype and shape

#### Scenario: Missing scale and unexpected tensor
- **WHEN** an NVFP4 expert lacks its `input_scale` and a text layer holds a tensor the plan doesn't use
- **THEN** the check fails naming both, while vision-tower tensors are not reported

#### Scenario: Two spellings of one tensor
- **WHEN** a tensor exists under both `model.language_model.` and `model.`
- **THEN** the check fails naming both names

#### Scenario: Format map disagrees with the tensors
- **WHEN** the map lists a projection as NVFP4 and the checkpoint holds MXFP8 tensors for it
- **THEN** the check fails naming the NVFP4 tensors that are missing or mismatched

#### Scenario: PLE table sizes
- **WHEN** the per-head table sizes imply a different shard row count, a head below `ngram_vocab_size_base`, or a padded total that doesn't split into `split_ngram_parts`
- **THEN** the check fails saying which

### Requirement: Only dense BF16 checkpoints load into GPU memory

Binding for the GPU MUST fail, before any GPU work, for a hybrid-layout checkpoint (stating that it is checked but not loaded yet), for one whose config declares mixed precision, and for one whose planned tensors are not BF16, naming the architecture or the tensor. A format override used for planning MUST NOT lift this refusal.

#### Scenario: Flash Next checkpoint
- **WHEN** a Qwen3.8 Flash Next checkpoint is bound for the GPU
- **THEN** binding fails stating that hybrid checkpoints are checked tensor by tensor but do not load into GPU memory yet

#### Scenario: Override used for planning
- **WHEN** a dense MIXED_PRECISION config is parsed with a BF16 override and its checkpoint is bound
- **THEN** binding fails stating that mixed-precision checkpoints do not load yet

#### Scenario: Quantized tensors behind a BF16 config
- **WHEN** a config without `quantization_config` is bound to a checkpoint whose `o_proj` is F8_E4M3 with a scale tensor
- **THEN** binding fails before GPU work, naming the tensor and its dtype

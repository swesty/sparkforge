# Spec Delta

## MODIFIED Requirements

### Requirement: The shared-expert width follows the config

The shared-expert width SHALL be `shared_expert_intermediate_size` when present, otherwise
`n_shared_experts` (default 0) times the expert width; a model has shared experts when that width
is above 0.

#### Scenario: Flash Next shared expert
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is parsed
- **THEN** the MoE has 512 experts, top-10, expert width 640 and a shared expert of width 640

#### Scenario: No shared-expert fields
- **WHEN** an MoE config with 128 experts, top-8 and expert width 768 omits `n_shared_experts` and `shared_expert_intermediate_size`
- **THEN** it parses with zero shared experts

### Requirement: Multimodal configs are read from text_config

When `config.json` has a `text_config` object, the language model SHALL be parsed from it, taking
`quantization_config` and `architectures` from the top level when `text_config` lacks them or sets
them to null. Different non-null values at both levels MUST be rejected, except that a
`quantization_config` conflict is ignored when a format override is given. A missing nested
`model_type` SHALL come from the top level.

#### Scenario: Flash Next nesting
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is parsed
- **THEN** the spec has 48 layers, hidden 2560 and vocab 248,320 from `text_config`, and the top-level mixed-precision map

#### Scenario: Null nested quantization config
- **WHEN** `text_config.quantization_config` is null and the top level carries the mixed-precision map
- **THEN** the map is used

#### Scenario: Family named only at the top level
- **WHEN** `text_config.model_type` is absent and the top-level `model_type` is `qwen4_exp`
- **THEN** the model is parsed as the qwen4_exp hybrid

#### Scenario: Nested quantization conflict with an override
- **WHEN** the nested `quantization_config` has `"bits": "4"`, the top level has `"bits": 4`, and the caller passes BF16
- **THEN** parsing succeeds with BF16 weights, and without the override it fails reporting the conflict

### Requirement: An explicit format override replaces detection

An explicit format override (CLI `--format`) SHALL replace detection entirely, so an unrecognized,
malformed or conflicting `quantization_config` (any field, any level, including a `quantized_layers`
map this parser cannot read) is not an error when an override is given. A mixed-precision
declaration at either level is still recorded.

#### Scenario: Override wins
- **WHEN** `quantization_config` names modelopt FP8 and the caller passes BF16 as the format
- **THEN** parsing succeeds with BF16 weights

#### Scenario: Malformed field with an override
- **WHEN** `quantization_config` is `{"quant_method": "awq", "bits": "4", "group_size": 128}` and the caller passes BF16
- **THEN** parsing succeeds with BF16 weights, and without the override it fails naming `quantization_config`

#### Scenario: Malformed layer map with an override
- **WHEN** a MIXED_PRECISION `quantized_layers` entry is null and the caller passes BF16
- **THEN** parsing succeeds with BF16 weights, and without the override it fails naming the entry

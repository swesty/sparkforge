# Spec Delta

## MODIFIED Requirements

### Requirement: Mixture-of-experts is detected across family naming conventions

The parser SHALL accept `num_experts`, `num_local_experts` (Mixtral) and `n_routed_experts` as the
expert count. A model SHALL be treated as MoE only when that count is greater than 1 and
`num_experts_per_tok` is present, which MUST be between 1 and the count. The expert width SHALL be
`moe_intermediate_size`, falling back to `intermediate_size`; if neither exists parsing MUST fail.

#### Scenario: Qwen3-30B-A3B
- **WHEN** `configs/qwen3-30b-a3b.json` is parsed
- **THEN** the spec is MoE with 128 experts, top-8, expert width 768

#### Scenario: Top-k out of range
- **WHEN** a 512-expert config sets `num_experts_per_tok` to 0 or 513
- **THEN** parsing fails naming `num_experts_per_tok`

### Requirement: Multimodal configs are read from text_config

When `config.json` has a `text_config` object, the language model SHALL be parsed from it, taking
`quantization_config` and `architectures` from the top level when `text_config` lacks them or sets
them to null; different non-null values at both levels MUST be rejected. A missing nested
`model_type` SHALL come from the top level. The vision configuration SHALL be ignored.

#### Scenario: Flash Next nesting
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is parsed
- **THEN** the spec has 48 layers, hidden 2560 and vocab 248,320 from `text_config`, and the top-level mixed-precision map

#### Scenario: Null nested quantization config
- **WHEN** `text_config.quantization_config` is null and the top level carries the mixed-precision map
- **THEN** the map is used

#### Scenario: Family named only at the top level
- **WHEN** `text_config.model_type` is absent and the top-level `model_type` is `qwen4_exp`
- **THEN** the model is parsed as the qwen4_exp hybrid

### Requirement: An explicit format override replaces detection

An explicit format override (CLI `--format`) SHALL replace detection entirely, so an unrecognized
or malformed `quantization_config`, including a `quantized_layers` map this parser cannot read, is
not an error when an override is given.

#### Scenario: Override wins
- **WHEN** `quantization_config` names modelopt FP8 and the caller passes BF16 as the format
- **THEN** parsing succeeds with BF16 weights

#### Scenario: Malformed layer map with an override
- **WHEN** a MIXED_PRECISION `quantized_layers` entry is null and the caller passes BF16
- **THEN** parsing succeeds with BF16 weights, and without the override it fails naming the entry

# model-config Specification

## Purpose

Turns a Hugging Face `config.json` into a normalized `ModelSpec` (`src/config.rs`). Everything
downstream (planning, codegen) reads only the `ModelSpec`. The central rule: a config that the
engine cannot compute correctly is rejected or flagged, never silently mapped to something close.

## Requirements

### Requirement: Dense attention shapes are derived from the config with documented defaults

The parser SHALL read `hidden_size`, `num_hidden_layers`, `num_attention_heads` and `vocab_size`
as required fields, and SHALL apply these defaults when a field is absent:
`num_key_value_heads` = `num_attention_heads`, `head_dim` = `hidden_size / num_attention_heads`,
`intermediate_size` = `4 * hidden_size`, `rms_norm_eps` = 1e-6, `rope_theta` = 10000,
`max_position_embeddings` = 8192. It MUST reject a config whose `num_attention_heads` is not
divisible by `num_key_value_heads`.

#### Scenario: Llama 3.1 8B
- **WHEN** `configs/llama-3.1-8b.json` is parsed
- **THEN** the fused QKV width is 32*128 + 2*8*128 = 6144 and the GQA group is 4

#### Scenario: Indivisible GQA
- **WHEN** a config has 32 attention heads and 6 key/value heads
- **THEN** parsing fails with an error naming both counts

### Requirement: Mixture-of-experts is detected across family naming conventions

The parser SHALL accept `num_experts`, `num_local_experts` (Mixtral) and `n_routed_experts` as the
expert count. A model SHALL be treated as MoE only when that count is greater than 1 and
`num_experts_per_tok` is present. The expert width SHALL be `moe_intermediate_size`, falling back
to `intermediate_size`; if neither exists parsing MUST fail. `n_shared_experts` defaults to 0.

#### Scenario: Qwen3-30B-A3B
- **WHEN** `configs/qwen3-30b-a3b.json` is parsed
- **THEN** the spec is MoE with 128 experts, top-8, expert width 768

### Requirement: Weight format comes from quantization_config, and unknown schemes are rejected

With no `quantization_config` the weight format SHALL be BF16. `quant_method` `nvfp4`, or
`modelopt` with `quant_algo` `NVFP4`, SHALL map to NVFP4. `gptq` or `awq` with 4 bits and a
positive `group_size` (default 128) SHALL map to W4A16 with that group. `modelopt`
`MIXED_PRECISION` SHALL map each `quantized_layers` module to its own format, with BF16 for the
rest. Rejection of everything else is specified in "Unknown or unsupported quantization is
rejected".

#### Scenario: AWQ
- **WHEN** `quantization_config` is `{"quant_method": "awq", "bits": 4, "group_size": 128}`
- **THEN** the weight format is W4A16 with group 128

#### Scenario: Flash Next mixed precision
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is parsed
- **THEN** 435 modules are mapped: routed experts NVFP4, MTP experts weight-only NVFP4, attention and shared-expert projections MXFP8, and the default format is BF16

### Requirement: Unknown or unsupported quantization is rejected

Without a format override, parsing MUST fail for GPTQ/AWQ whose `bits` is anything other than 4 or is absent, for a GPTQ/AWQ `group_size` of zero or less (per-channel scales), for `modelopt` with a `quant_algo` other than NVFP4 or MIXED_PRECISION, for MIXED_PRECISION without a `quantized_layers` map or with a scheme other than NVFP4/16, W4A16_NVFP4/16 or MXFP8/32, and for any other `quant_method`.

#### Scenario: Per-channel GPTQ
- **WHEN** `quantization_config` is `{"quant_method": "gptq", "bits": 4, "group_size": -1}`
- **THEN** parsing fails and the message suggests `--format`

#### Scenario: 8-bit GPTQ
- **WHEN** `quantization_config` is `{"quant_method": "gptq", "bits": 8, "group_size": 128}` and no format override is given
- **THEN** parsing fails and the message names the bit width and suggests `--format`

#### Scenario: Override wins over an unsupported bit width
- **WHEN** `quantization_config` is `{"quant_method": "awq", "bits": 8, "group_size": 128}` and the caller passes BF16 as the format
- **THEN** parsing succeeds with BF16 weights

#### Scenario: Unknown scheme in a mixed-precision map
- **WHEN** a MIXED_PRECISION map lists a module with `quant_algo` `FP8`
- **THEN** parsing fails naming that module

### Requirement: An explicit format override replaces detection

An explicit format override (CLI `--format`) SHALL replace detection entirely, so an unrecognized
`quantization_config` is not an error when an override is given.

#### Scenario: Override wins
- **WHEN** `quantization_config` names modelopt FP8 and the caller passes BF16 as the format
- **THEN** parsing succeeds with BF16 weights

### Requirement: RoPE scaling is either implemented exactly or rejected

An absent or null `rope_scaling`, or one whose `rope_type`/`type` is `default`, SHALL mean no
scaling. `llama3` SHALL be parsed with its `factor`, `low_freq_factor`, `high_freq_factor` and
`original_max_position_embeddings`, failing if any is missing. Every other type (for example
`yarn`, `dynamic`, `linear`, `longrope`) MUST be rejected.

#### Scenario: YaRN
- **WHEN** `rope_scaling` is `{"rope_type": "yarn", "factor": 4.0}`
- **THEN** parsing fails naming the unsupported type

### Requirement: Unsupported attention variants are rejected or flagged

A config containing `kv_lora_rank` (MLA, DeepSeek V2/V3) MUST be rejected. Architectures whose name
starts with `Qwen3` SHALL set `qk_norm`, and `attention_bias: true` or a `Qwen2` architecture SHALL
set `qkv_bias`. These two flags mark features not yet generated; the planner reports them (see
kernel-planning).

#### Scenario: DeepSeek V3
- **WHEN** a config contains `"kv_lora_rank": 512`
- **THEN** parsing fails with an MLA error

#### Scenario: Qwen3 is flagged, not rejected
- **WHEN** `configs/qwen3-30b-a3b.json` is parsed
- **THEN** parsing succeeds and `qk_norm` is true

### Requirement: Multimodal configs are read from text_config

When `config.json` has a `text_config` object, the language model SHALL be parsed from it, taking
`quantization_config` and `architectures` from the top level when `text_config` lacks them or sets
them to null. Different non-null values at both levels MUST be rejected. The vision configuration
SHALL be ignored.

#### Scenario: Flash Next nesting
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is parsed
- **THEN** the spec has 48 layers, hidden 2560 and vocab 248,320 from `text_config`, and the top-level mixed-precision map

#### Scenario: Null nested quantization config
- **WHEN** `text_config.quantization_config` is null and the top level carries the mixed-precision map
- **THEN** the map is used

### Requirement: The qwen4_exp hybrid layout is parsed into a layer schedule

A `qwen4_exp` config SHALL always be parsed as hybrid and MUST have `layer_types` matching
`num_hidden_layers` and `full_attention_interval`; `linear_attention` layers are rejected for other
model types. The parser SHALL record the Gated DeltaNet, indexer, HyperConnection, PLE, partial
interleaved mRoPE and MTP parameters, failing on a missing field or an `mrope_section` that does
not cover the rotated dims.

#### Scenario: Flash Next schedule
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is parsed
- **THEN** layers 3, 7, …, 47 are full attention and the other 36 are linear attention, with 4 HyperConnection streams, 16 PLE heads of 160, 64 rotated dims and 1 MTP layer

#### Scenario: Another hybrid family
- **WHEN** the same schedule appears with `model_type` `qwen3_next`
- **THEN** parsing fails naming the model type

#### Scenario: Only full attention
- **WHEN** a non-qwen4_exp config has `layer_types` that are all `full_attention`
- **THEN** the model is parsed as an ordinary dense model

#### Scenario: qwen4_exp without linear attention
- **WHEN** a qwen4_exp config sets every layer to `full_attention`
- **THEN** it is still parsed as hybrid, so no dense kernel is generated for it

### Requirement: Hybrid configs are validated against what the planner models

Every qwen4_exp dimension and count MUST be positive, `ple_layer_ids` MUST have exactly one entry,
and at most one MTP layer is accepted. The MTP layer count SHALL come from `mtp_num_hidden_layers`
or `mtp.num_hidden_layers`, and parsing MUST fail when both are present and differ.

#### Scenario: Zero dimension
- **WHEN** `full_attention_interval` or `indexer_compress_ratio` is 0
- **THEN** parsing fails naming the field instead of panicking later

#### Scenario: No PLE placement
- **WHEN** `ple_layer_ids` is empty
- **THEN** parsing fails stating that exactly one PLE layer is supported

#### Scenario: Nested MTP count only
- **WHEN** `mtp_num_hidden_layers` is absent and `mtp.num_hidden_layers` is 1
- **THEN** the spec has 1 MTP layer and the resident budget still includes it

### Requirement: Mixed precision is accepted only for the hybrid layout

A dense model with a MIXED_PRECISION map MUST be rejected unless a format override is given,
because the dense planner does not read per-module formats. Two map paths for one module (with and
without a language-model prefix, or two indices of the single MTP layer) MUST agree. Quantized
HyperConnection weights MUST be rejected.

#### Scenario: Conflicting aliases
- **WHEN** the map lists `layers.0.mlp.experts` as MXFP8 beside `model.language_model.layers.0.mlp.experts` as NVFP4
- **THEN** parsing fails reporting conflicting formats for one module

#### Scenario: Dense mixed-precision config
- **WHEN** a dense config's MIXED_PRECISION map lists `model.layers.0.self_attn.o_proj` as MXFP8
- **THEN** parsing fails stating that mixed precision is supported only for qwen4_exp

### Requirement: RoPE parameters may come from rope_parameters

`rope_theta`, the RoPE type and `partial_rotary_factor` SHALL also be read from `rope_parameters`.
A dense model with a `partial_rotary_factor` other than 1 MUST be rejected, because the generated
RoPE rotates the whole head.

#### Scenario: Partial rotary on a dense model
- **WHEN** a dense config sets `partial_rotary_factor` 0.5
- **THEN** parsing fails naming `partial_rotary_factor`

### Requirement: The shared-expert width follows the config

The shared-expert width SHALL be `shared_expert_intermediate_size` when present, otherwise
`n_shared_experts` times the expert width; a model has shared experts when that width is above 0.

#### Scenario: Flash Next shared expert
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is parsed
- **THEN** the MoE has 512 experts, top-10, expert width 640 and a shared expert of width 640

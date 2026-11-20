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
positive `group_size` (default 128) SHALL map to W4A16 with that group. Rejection of everything
else is specified in "Unknown or unsupported quantization is rejected".

#### Scenario: AWQ
- **WHEN** `quantization_config` is `{"quant_method": "awq", "bits": 4, "group_size": 128}`
- **THEN** the weight format is W4A16 with group 128

### Requirement: Unknown or unsupported quantization is rejected

Parsing MUST fail for a GPTQ/AWQ `group_size` of zero or less (per-channel scales), for `modelopt`
with any `quant_algo` other than NVFP4, and for any other `quant_method`.

#### Scenario: Per-channel GPTQ
- **WHEN** `quantization_config` is `{"quant_method": "gptq", "bits": 4, "group_size": -1}`
- **THEN** parsing fails and the message suggests `--format`

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

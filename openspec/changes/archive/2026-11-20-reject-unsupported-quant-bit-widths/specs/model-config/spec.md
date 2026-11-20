## MODIFIED Requirements

### Requirement: Unknown or unsupported quantization is rejected

Without a format override, parsing MUST fail for GPTQ/AWQ whose `bits` is anything other than 4 or is absent, for a GPTQ/AWQ `group_size` of zero or less (per-channel scales), for `modelopt` with any `quant_algo` other than NVFP4, and for any other `quant_method`.

#### Scenario: Per-channel GPTQ
- **WHEN** `quantization_config` is `{"quant_method": "gptq", "bits": 4, "group_size": -1}`
- **THEN** parsing fails and the message suggests `--format`

#### Scenario: 8-bit GPTQ
- **WHEN** `quantization_config` is `{"quant_method": "gptq", "bits": 8, "group_size": 128}` and no format override is given
- **THEN** parsing fails and the message names the bit width and suggests `--format`

#### Scenario: Override wins over an unsupported bit width
- **WHEN** `quantization_config` is `{"quant_method": "awq", "bits": 8, "group_size": 128}` and the caller passes BF16 as the format
- **THEN** parsing succeeds with BF16 weights

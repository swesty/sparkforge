use sparkforge::{codegen, config::ModelSpec, plan, target::GB10};

fn load(p: &str) -> plan::Plan {
    let s = ModelSpec::from_json(&std::fs::read_to_string(p).unwrap(), None).unwrap();
    plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 32768 })
}

#[test]
fn llama_shapes_are_derived_from_config() {
    let p = load("configs/llama-3.1-8b.json");
    let n = |name: &str| p.linears.iter().find(|l| l.name == name).unwrap().n;
    assert_eq!(n("qkv_proj"), 32 * 128 + 2 * 8 * 128);
    assert_eq!(n("gate_up_proj"), 2 * 14336);
    assert_eq!(p.attn.group, 4);
    assert!(p.budget.fits);
}

#[test]
fn every_kernel_respects_gb10_limits() {
    for cfg in ["configs/llama-3.1-8b.json", "configs/qwen3-30b-a3b.json"] {
        let p = load(cfg);
        for t in &p.gemm { assert!(t.smem_bytes <= GB10.smem_per_block_optin, "{t:?}"); }
        assert!(p.attn.smem_bytes <= GB10.smem_per_block_optin);
        assert!(p.attn.grid[0] * p.attn.grid[1] >= GB10.sm_count, "attention must fill all 48 SMs");
        for k in codegen::generate(&p).kernels { assert!(k.block <= 1024 && k.block % 32 == 0, "{k:?}"); }
    }
}

#[test]
fn moe_is_detected_and_planned() {
    let p = load("configs/qwen3-30b-a3b.json");
    let e = p.model.moe.as_ref().unwrap();
    assert_eq!((e.num_experts, e.experts_per_tok, e.expert_intermediate), (128, 8, 768));
    assert!(p.gemm.iter().any(|t| t.linear == "expert_gate_up" && t.m == 128));
}

#[test]
fn different_models_get_different_kernels() {
    let a = codegen::generate(&load("configs/llama-3.1-8b.json"));
    let b = codegen::generate(&load("configs/qwen3-30b-a3b.json"));
    assert_ne!(a.cache_key, b.cache_key);
}

fn spec(json: &str) -> anyhow::Result<ModelSpec> {
    ModelSpec::from_json(json, None)
}

const LLAMA31_ROPE: &str = r#"{"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
  "num_key_value_heads": 8, "vocab_size": 128256, "rope_theta": 500000.0,
  "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                   "original_max_position_embeddings": 8192, "rope_type": "llama3"}}"#;

#[test]
fn llama3_rope_scaling_keeps_high_freqs_and_divides_low_freqs() {
    let scaled = codegen::rope_inv_freq(&spec(LLAMA31_ROPE).unwrap());
    let plain = codegen::rope_inv_freq(&spec(&LLAMA31_ROPE.replace("llama3", "default")).unwrap());
    assert_eq!(scaled.len(), 64);
    // wavelength 2*pi/f below 8192/4 = 2048 tokens: untouched
    assert_eq!(scaled[0], plain[0]);
    assert_eq!(scaled[0], 1.0);
    // wavelength above 8192 tokens: divided by the factor
    assert_eq!(scaled[63], plain[63] / 8.0);
    // in between: strictly between the two
    let mid = (0..64).find(|&i| { let w = 2.0 * std::f32::consts::PI / plain[i]; w > 2048.0 && w < 8192.0 }).unwrap();
    assert!(scaled[mid] < plain[mid] && scaled[mid] > plain[mid] / 8.0);
}

#[test]
fn unsupported_configs_fail_loudly_instead_of_planning_wrong_math() {
    let base = r#"{"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000 EXTRA}"#;
    let with = |extra: &str| spec(&base.replace(" EXTRA", extra));
    assert!(with(r#", "rope_scaling": {"rope_type": "yarn", "factor": 4.0}"#).is_err());
    assert!(with(r#", "kv_lora_rank": 512"#).is_err());
    assert!(with(r#", "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": -1}"#).is_err());
    assert!(with(r#", "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}"#).is_err());
    assert!(with(r#", "quantization_config": {"quant_method": "compressed-tensors"}"#).is_err());
    // an explicit --format overrides an unrecognized quantization_config
    let fp8 = base.replace(" EXTRA", r#", "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}"#);
    assert!(ModelSpec::from_json(&fp8, Some(sparkforge::config::WeightFormat::Bf16)).is_ok());
    // supported paths still parse
    assert!(matches!(with(r#", "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}"#).unwrap().weights, sparkforge::config::WeightFormat::Nvfp4));
    assert!(matches!(with(r#", "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}"#).unwrap().weights, sparkforge::config::WeightFormat::W4A16 { group: 128 }));
    assert!(with(r#", "rope_scaling": null"#).unwrap().rope_scaling.is_none());
}

#[test]
fn unimplemented_attention_features_are_flagged_in_the_plan() {
    let p = load("configs/qwen3-30b-a3b.json");
    assert!(p.model.qk_norm);
    assert!(p.notes.iter().any(|n| n.contains("q_norm")));
    assert!(!load("configs/llama-3.1-8b.json").notes.iter().any(|n| n.contains("q_norm") || n.contains("bias")));
}

#[test]
fn attention_splits_stay_valid_below_at_and_above_256_tokens() {
    let s = ModelSpec::from_json(&std::fs::read_to_string("configs/llama-3.1-8b.json").unwrap(), None).unwrap();
    let splits = |ctx| plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx }).attn.splits;
    assert_eq!((splits(128), splits(255), splits(256), splits(512), splits(32768)), (1, 1, 1, 2, 12));
}

#[test]
fn gemv_whose_staged_activations_exceed_smem_is_planned_not_generated() {
    // down_proj K = 4 * 16384 = 65536 -> 131072 B of staged activations, above 101376 B.
    let wide = r#"{"hidden_size": 16384, "num_hidden_layers": 1, "num_attention_heads": 128,
                   "num_key_value_heads": 8, "vocab_size": 4096}"#;
    let p = plan::build(&spec(wide).unwrap(), &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 });
    let g = codegen::generate(&p);
    let down = g.kernels.iter().find(|k| k.entry == "gemv_down_proj").unwrap();
    assert_eq!(down.status, "planned");
    assert!(p.notes.iter().any(|n| n.contains("gemv down_proj")));
    assert!(!g.files[0].1.contains("gemv_down_proj("));
    for k in g.kernels.iter().filter(|k| k.status == "generated") {
        assert!(k.dyn_smem <= GB10.smem_per_block_optin, "{k:?}");
    }
}

#[test]
fn only_generated_gemvs_get_cuda_source() {
    let s = ModelSpec::from_json(&std::fs::read_to_string("configs/qwen3-30b-a3b.json").unwrap(),
        Some(sparkforge::config::WeightFormat::Nvfp4)).unwrap();
    let g = codegen::generate(&plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 32768 }));
    let src = &g.files.iter().find(|(n, _)| n == "model.cu").unwrap().1;
    let status = |e: &str| g.kernels.iter().find(|k| k.entry == e).unwrap().status;
    assert_eq!(status("gemv_qkv_proj"), "planned");
    assert!(!src.contains("gemv_qkv_proj("));
    assert_eq!(status("gemv_lm_head"), "generated");
    assert!(src.contains("gemv_lm_head("));
}

#[test]
fn tile_choice_weighs_wave_efficiency_not_just_intensity() {
    // 128x128 has the higher (capped) intensity, but for M=128, N=2048 it makes only 16 tiles for
    // 48 SMs; the scored search must prefer the 64x64 tile that keeps more SMs busy.
    let p = load("configs/qwen3-30b-a3b.json");
    let t = p.gemm.iter().find(|t| t.linear == "expert_down").unwrap();
    assert_eq!((t.m, t.n, t.bm, t.bn), (128, 2048, 64, 64), "{t:?}");
    assert!(t.wave_efficiency > 16.0 / 48.0, "{t:?}");
}

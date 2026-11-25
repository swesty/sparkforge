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
fn gptq_and_awq_at_unsupported_bit_widths_are_rejected_unless_overridden() {
    let base = r#"{"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000,
                   "quantization_config": QUANT}"#;
    let cfg = |q: &str| base.replace("QUANT", q);
    let gptq8 = cfg(r#"{"quant_method": "gptq", "bits": 8, "group_size": 128}"#);
    let awq8 = cfg(r#"{"quant_method": "awq", "bits": 8, "group_size": 128}"#);
    // No override: only 4-bit maps to W4A16, so 8 bits must not be read as 4-bit weights.
    let msg = format!("{:#}", spec(&gptq8).unwrap_err());
    assert!(msg.contains("bits Some(8)") && msg.contains("--format"), "{msg}");
    assert!(spec(&awq8).is_err());
    assert!(spec(&cfg(r#"{"quant_method": "gptq", "bits": 3, "group_size": 128}"#)).is_err());
    assert!(spec(&cfg(r#"{"quant_method": "awq", "group_size": 128}"#)).is_err(), "missing bits");
    // An explicit --format still replaces detection.
    let s = ModelSpec::from_json(&awq8, Some(sparkforge::config::WeightFormat::Bf16)).unwrap();
    assert!(matches!(s.weights, sparkforge::config::WeightFormat::Bf16));
    let s = ModelSpec::from_json(&gptq8, Some(sparkforge::config::WeightFormat::Bf16)).unwrap();
    assert!(matches!(s.weights, sparkforge::config::WeightFormat::Bf16));
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
fn nvfp4_gemvs_are_generated_without_shared_memory() {
    let p = qwen3_nvfp4();
    let g = codegen::generate(&p);
    let src = &g.files.iter().find(|(n, _)| n == "model.cu").unwrap().1;
    for e in ["gemv_qkv_proj", "gemv_o_proj", "gemv_expert_gate_up", "gemv_expert_down"] {
        let k = g.kernels.iter().find(|k| k.entry == e).unwrap();
        assert_eq!((k.status, k.dyn_smem, k.block), ("generated", 0, 256), "{e}");
        assert_eq!(k.args, codegen::FP4_ARGS.to_vec(), "{e}");
        assert_eq!(k.grid, p.gemv.iter().find(|g| format!("gemv_{}", g.linear) == e).unwrap().fp4.as_ref().unwrap().grid, "{e}");
        assert!(src.contains(&format!("{e}(")), "{e} has source");
    }
    assert_eq!(src.matches("cvt.rn.f16x2.e2m1x2 %0").count(), 1, "the FP4 helpers are emitted once");
    // Grouped GEMVs read expert ids; the dense ones do not.
    assert!(src.contains("namespace gemv_expert_down_ns {\nconstexpr int N = 2048, K = 768, SPLITS = 4, KT_PER_SPLIT = 12;"), "{src}");
    assert!(src.contains("constexpr bool GROUPED = true") && src.contains("constexpr bool GROUPED = false"));
    // qkv: q rows, then k, then v, each with its own global scale.
    assert!(src.contains("seg_col(int row) { return row < 4096 ? 0 : row < 4608 ? 1 : 2; }"));
    let status = |e: &str| g.kernels.iter().find(|k| k.entry == e).unwrap().status;
    assert_eq!(status("gemv_lm_head"), "generated");
    assert_eq!(status("gemv_router"), "generated");
    for k in &g.kernels {
        assert!(k.block % 32 == 0 && k.block <= 1024, "{k:?}");
    }
}

#[test]
fn nvfp4_gemvs_that_do_not_tile_stay_planned() {
    // hidden 2016: o_proj and down_proj have N = 2016, not a multiple of 64.
    let s = spec(r#"{"hidden_size": 2016, "head_dim": 128, "intermediate_size": 4096, "num_hidden_layers": 2, "num_attention_heads": 16,
                     "num_key_value_heads": 4, "vocab_size": 4096, "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}"#).unwrap();
    let g = codegen::generate(&plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 }));
    let src = &g.files.iter().find(|(n, _)| n == "model.cu").unwrap().1;
    let status = |e: &str| g.kernels.iter().find(|k| k.entry == e).unwrap().status;
    for e in ["gemv_o_proj", "gemv_down_proj"] {
        assert_eq!(status(e), "planned", "{e}");
        assert!(!src.contains(&format!("{e}(")), "{e} has no source");
    }
    assert_eq!(status("gemv_qkv_proj"), "generated");
    assert_eq!(status("gemv_lm_head"), "generated");
    assert!(src.contains("gemv_lm_head("));
}

#[test]
fn a_wide_nvfp4_gemv_needs_no_shared_memory() {
    // down_proj K = 65536: the BF16 GEMV could not stage it, the NVFP4 one stages nothing.
    let s = spec(r#"{"hidden_size": 16384, "num_hidden_layers": 1, "num_attention_heads": 128, "num_key_value_heads": 8,
                     "vocab_size": 4096, "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}"#).unwrap();
    let p = plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 });
    let g = codegen::generate(&p);
    let down = g.kernels.iter().find(|k| k.entry == "gemv_down_proj").unwrap();
    assert_eq!((down.status, down.dyn_smem), ("generated", 0));
    assert!(!p.notes.iter().any(|n| n.contains("gemv down_proj")), "no staging note for an NVFP4 GEMV");
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

#[test]
fn a_tied_dense_head_takes_the_resolved_embedding_format() {
    // codex-review fp:509d200bd957: a dense spec resolved with an MXFP8 entry for its tied
    // embedding priced the head BF16 and the embedding at 2 bytes per element. A tied head is the
    // embedding matrix, so both take the resolved format, counted once.
    let mut s = ModelSpec::from_json(&std::fs::read_to_string("configs/llama-3.1-8b.json").unwrap(), None).unwrap();
    s.tie_embeddings = true;
    let opts = plan::PlanOpts { prefill_tokens: 2048, ctx: 32768 };
    let bf16 = plan::build(&s, &GB10, &opts);
    s.formats.modules.insert("model.embed_tokens".into(), sparkforge::config::WeightFormat::Mxfp8);
    s.resolve_modules(["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]).unwrap();
    let p = plan::build(&s, &GB10, &opts);
    let saved = (bf16.budget.weight_gib - p.budget.weight_gib) * (1u64 << 30) as f64;
    let once = 128_256.0 * 4096.0 * (2.0 - 1.03125);
    assert!((saved - once).abs() < 1e3, "resident once at MXFP8: saved {saved} vs {once}");
    let head = |p: &plan::Plan| p.gemv.iter().find(|g| g.linear == "lm_head").unwrap().est_us;
    assert!((head(&p) / head(&bf16) - 1.03125 / 2.0).abs() < 1e-9, "the head GEMV reads MXFP8 bytes");
}

fn qwen3_nvfp4() -> plan::Plan {
    let s = ModelSpec::from_json(&std::fs::read_to_string("configs/qwen3-30b-a3b.json").unwrap(),
        Some(sparkforge::config::WeightFormat::Nvfp4)).unwrap();
    plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 32768 })
}

fn fp4_of<'a>(p: &'a plan::Plan, linear: &str) -> &'a plan::Fp4Gemv {
    p.gemv.iter().find(|g| g.linear == linear).and_then(|g| g.fp4.as_ref()).unwrap_or_else(|| panic!("{linear} has no NVFP4 plan"))
}

#[test]
fn nvfp4_gemvs_split_k_to_about_20_warps_per_sm() {
    let p = qwen3_nvfp4();
    let f = |l| { let g = fp4_of(&p, l); (g.splits, g.grid) };
    assert_eq!(f("qkv_proj"), (12, [10, 12, 1]));
    assert_eq!(f("o_proj").0, 29);
    assert_eq!(f("expert_gate_up"), (5, [3, 5, 8]));
    assert_eq!(f("expert_down"), (4, [4, 4, 8]));
    for g in p.gemv.iter().filter(|g| g.fp4.is_some()) {
        assert_eq!((g.smem_bytes, g.warps_per_block), (0, 8), "{}: nothing staged, 8 warps", g.linear);
        let f = g.fp4.as_ref().unwrap();
        let k_tiles = g.k / 16;
        assert!(f.splits * f.kt_per_split >= k_tiles && (f.splits - 1) * f.kt_per_split < k_tiles, "{}: no empty split", g.linear);
    }
    // BF16 linears of the same plan keep the staged GEMV.
    assert!(p.gemv.iter().find(|g| g.linear == "lm_head").unwrap().fp4.is_none());
    assert!(p.gemv.iter().find(|g| g.linear == "router").unwrap().fp4.is_none());
}

#[test]
fn nvfp4_segments_follow_the_stacked_tensors() {
    use plan::Segment;
    let p = qwen3_nvfp4();
    // q is 32 x 128 = 4096 rows, k and v 4 x 128 = 512 each.
    assert_eq!(fp4_of(&p, "qkv_proj").segments, vec![Segment { end: 4096, col: 0 }, Segment { end: 4608, col: 1 }, Segment { end: 5120, col: 2 }]);
    let gu = fp4_of(&p, "expert_gate_up");
    assert_eq!((gu.segments.clone(), gu.gscale_stride, gu.x_stride, gu.experts), (vec![Segment { end: 768, col: 0 }, Segment { end: 1536, col: 1 }], 3, 0, 128));
    let d = fp4_of(&p, "expert_down");
    assert_eq!((d.segments.clone(), d.gscale_stride, d.x_stride), (vec![Segment { end: 2048, col: 2 }], 3, 768));
    let o = fp4_of(&p, "o_proj");
    assert_eq!((o.segments.clone(), o.gscale_stride, o.experts), (vec![Segment { end: 2048, col: 0 }], 0, 0));
}

#[test]
fn the_token_estimate_counts_a_grouped_expert_gemv_once_per_layer() {
    use sparkforge::estimate::{token_estimate_us, Row};
    let us = [("rmsnorm", 1.0), ("gemv_qkv_proj", 10.0), ("rope_kv_append", 2.0), ("attn_decode (split+combine)", 20.0),
        ("gemv_o_proj", 8.0), ("gemv_router", 3.0), ("gemv_expert_gate_up", 30.0), ("silu_mul", 0.5), ("gemv_expert_down", 15.0),
        ("gemv_lm_head", 600.0)];
    let rows: Vec<Row> = us.iter().map(|&(k, t)| Row { kernel: k.into(), case: String::new(), err: 0.0, us: Some(t), gbs: None }).collect();
    // NVFP4: the expert GEMVs are grouped (8 slots in one launch): once per layer; SiLU 8 times.
    let layer = 2.0 + 10.0 + 2.0 + 20.0 + 8.0 + 3.0 + 30.0 + 8.0 * 0.5 + 15.0;
    assert_eq!(token_estimate_us(&qwen3_nvfp4(), &rows), Some(48.0 * layer + 1.0 + 600.0));
    // BF16: one launch per active expert.
    let layer = 2.0 + 10.0 + 2.0 + 20.0 + 8.0 + 3.0 + 8.0 * (30.0 + 0.5 + 15.0);
    assert_eq!(token_estimate_us(&load("configs/qwen3-30b-a3b.json"), &rows), Some(48.0 * layer + 1.0 + 600.0));
    // Missing a kernel the token needs: no estimate.
    assert_eq!(token_estimate_us(&qwen3_nvfp4(), &rows[1..]), None);
}

#[test]
fn the_token_estimate_counts_the_shared_expert_activation() {
    use sparkforge::estimate::{token_estimate_us, Row};
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("configs/qwen3-30b-a3b.json").unwrap()).unwrap();
    v["shared_expert_intermediate_size"] = 4096.into();
    let p = plan::build(&spec(&v.to_string()).unwrap(), &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 32768 });
    assert!(p.model.moe.as_ref().unwrap().shared_experts > 0);
    let us = [("rmsnorm", 1.0), ("gemv_qkv_proj", 10.0), ("rope_kv_append", 2.0), ("attn_decode (split+combine)", 20.0),
        ("gemv_o_proj", 8.0), ("gemv_router", 3.0), ("gemv_expert_gate_up", 30.0), ("silu_mul", 0.5), ("gemv_expert_down", 15.0),
        ("gemv_shared_gate_up", 40.0), ("gemv_shared_down", 20.0), ("gemv_lm_head", 600.0)];
    let rows = |skip: &str| us.iter().filter(|(k, _)| *k != skip).map(|&(k, t)| Row { kernel: k.into(), case: String::new(), err: 0.0, us: Some(t), gbs: None }).collect::<Vec<_>>();
    // BF16 experts: 8 launches each, routed SiLU 8 times, plus the shared GEMVs and their SiLU.
    let layer = 2.0 + 10.0 + 2.0 + 20.0 + 8.0 + 3.0 + 8.0 * (30.0 + 0.5 + 15.0) + 40.0 + 0.5 + 20.0;
    assert_eq!(token_estimate_us(&p, &rows("")), Some(48.0 * layer + 1.0 + 600.0));
    // Without a shared GEMV's timing, no estimate.
    assert_eq!(token_estimate_us(&p, &rows("gemv_shared_down")), None);
}

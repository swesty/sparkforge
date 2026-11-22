//! Qwen3.8 Flash Next (`qwen4_exp`): parsing, planning and the decode byte budget.
//!
//! `configs/qwen3.8-flash-next-nvfp4.json` is the checkpoint's config.json verbatim, from
//! `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at revision 6909a5bed089a48fa07e956d3915af2537de9368.
//! The byte figures below come from summing that revision's safetensors headers (Linear plan
//! "Qwen3.8 Flash Next engine plan (2026-11-20)", research section 2.2).
use serde_json::{json, Value};
use sparkforge::config::{LayerKind, ModelSpec, WeightFormat};
use sparkforge::{codegen, plan, target::GB10};

const CFG: &str = "configs/qwen3.8-flash-next-nvfp4.json";

fn raw() -> Value {
    serde_json::from_str(&std::fs::read_to_string(CFG).unwrap()).unwrap()
}

fn spec_of(v: &Value) -> anyhow::Result<ModelSpec> {
    ModelSpec::from_json(&v.to_string(), None)
}

fn plan_of(s: &ModelSpec) -> plan::Plan {
    plan::build(s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 32768 })
}

/// The quantized_layers entries a plan reports as unused.
fn unused_entries(p: &plan::Plan) -> Vec<String> {
    p.notes.iter().find_map(|n| n.split_once("not used by the plan (no module of this model at that path): "))
        .map_or(vec![], |(_, list)| list.split(", ").map(str::to_owned).collect())
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want
}

#[test]
fn flash_next_config_parses_into_a_hybrid_spec() {
    let s = spec_of(&raw()).unwrap();
    let h = s.hybrid.as_ref().expect("qwen4_exp is hybrid");
    assert_eq!((s.layers, s.hidden, s.heads, s.kv_heads, s.head_dim, s.vocab), (48, 2560, 24, 2, 256, 248_320));
    assert_eq!((h.count(LayerKind::LinearAttention), h.count(LayerKind::FullAttention)), (36, 12));
    for (i, k) in h.schedule.iter().enumerate() {
        assert_eq!(*k == LayerKind::FullAttention, i % 4 == 3, "layer {i}");
    }
    assert_eq!((h.gdn.qkv_dim(), h.gdn.value_dim(), h.gdn.state_bytes()), (10_240, 6144, 48 * 128 * 128 * 4));
    assert_eq!((h.qsa.selected_positions(), h.qsa.index_qk_dim()), (2051, 640));
    assert_eq!((h.hc.streams, h.hc.lowrank, h.hc_mixers()), (4, 320, 97));
    assert_eq!((h.ple.heads(), h.ple.head_dim(), h.ple.table_rows(), h.ple.layer_ids.clone()), (16, 160, 320_000_000, vec![2]));
    assert_eq!((h.mrope.sections.clone(), h.mrope.interleaved, h.mtp_layers), (vec![11, 11, 10], true, 1));
    assert_eq!((s.rope_theta, s.max_pos, s.tie_embeddings), (1e7, 262_144, false));
    let e = s.moe.as_ref().unwrap();
    assert_eq!((e.num_experts, e.experts_per_tok, e.expert_intermediate, e.shared_intermediate), (512, 10, 640, 640));
    // MIXED_PRECISION: everything unlisted is BF16; the map holds 435 modules in three schemes.
    assert_eq!((s.weights, s.formats.modules.len()), (WeightFormat::Bf16, 435));
    assert_eq!(s.formats.get("model.language_model.layers.0.mlp.experts"), Some(WeightFormat::Nvfp4));
    assert_eq!(s.formats.get("model.language_model.layers.0.linear_attn.in_proj_qkv"), Some(WeightFormat::Mxfp8));
    assert_eq!(s.formats.get("mtp.layers.0.mlp.experts"), Some(WeightFormat::Nvfp4WeightOnly));
    assert!(!s.qk_norm && !s.qkv_bias, "dense-path flags stay off; the sparse-attention norm is part of the hybrid plan");
}

#[test]
fn flash_next_decode_bytes_match_the_checkpoint_headers() {
    let p = plan_of(&spec_of(&raw()).unwrap());
    // Context-free decode bytes per token from the safetensors headers: 7,338 MB.
    let free = p.budget.decode_bytes_per_token_ctx_free_gb.unwrap();
    assert!(close(free, 7.338, 0.01), "{free}");
    // At 32K: plus FP8 KV for 2,051 selected positions and the indexer scan, in 12 layers.
    let ctx_mb = (p.budget.decode_bytes_per_token_gb - free) * 1e3;
    assert!(close(ctx_mb, 2051.0 * 1024.0 * 12.0 / 1e6 + 8192.0 * 128.0 * 12.0 / 1e6, 0.001), "{ctx_mb}");
    // Per-group figures from the headers, in MB per token.
    let g = plan::bytes_by_group(&p);
    let mb = |name: &str| g.iter().find(|(k, _)| *k == name).map(|(_, b)| b / 1e6).unwrap();
    for (name, want, tol) in [("gdn_projections", 2153.0, 0.01), ("routed_experts", 1327.0, 0.01), ("hyperconnection", 1283.0, 0.01),
        ("lm_head", 1271.0, 0.01), ("qsa_projections", 637.0, 0.01), ("shared_experts", 244.0, 0.01), ("routers", 126.0, 0.01),
        ("ple", 66.0, 0.02), ("gdn_state", 231.0, 0.02)] {
        assert!(close(mb(name), want, tol), "{name}: {} MB, headers {want} MB", mb(name));
    }
    // Resident text weights with the PLE table: 105.4 GB from the headers.
    assert!(close(p.budget.weight_gib * (1u64 << 30) as f64 / 1e9, 105.4, 0.01), "{}", p.budget.weight_gib);
    assert!(p.budget.fits);
}

#[test]
fn flash_next_plans_one_op_per_distinct_kernel() {
    let p = plan_of(&spec_of(&raw()).unwrap());
    let names: std::collections::BTreeSet<&str> = p.ops.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names.len(), p.ops.len(), "op names are unique");
    assert!((25..=40).contains(&p.ops.len()), "about 30 distinct kernels, not one per layer: {}", p.ops.len());
    let op = |n: &str| p.ops.iter().find(|o| o.name == n).unwrap_or_else(|| panic!("missing op {n}"));
    assert_eq!((op("gdn_in_proj_qkv").per_token, op("gdn_in_proj_qkv").format), (36, Some(WeightFormat::Mxfp8)));
    assert_eq!((op("qsa_q_proj").per_token, op("qsa_q_proj").shape.as_str()), (12, "N=12288 K=2560"));
    assert_eq!(op("expert_gate_up").format, Some(WeightFormat::Nvfp4));
    assert_eq!((op("hc_down_inject").per_token, op("hc_down_final").per_token, op("hc_up").per_token), (96, 1, 97));
    assert_eq!((op("lm_head").per_token, op("lm_head").format), (1, Some(WeightFormat::Bf16)));
    for n in ["gdn_recurrence", "gdn_conv1d", "qsa_sparse_attention", "qsa_indexer_score_topk", "ple_ngram_gather", "hc_combine_norm"] {
        op(n);
    }
    // Sparse attention splits are sized for the selected positions, not the whole context.
    assert_eq!(p.attn.ref_ctx, 2051);
    assert!(p.gemm.is_empty(), "prefill tiles for MXFP8/NVFP4 are not planned yet");
}

#[test]
fn flash_next_generates_no_kernels_yet() {
    let p = plan_of(&spec_of(&raw()).unwrap());
    let g = codegen::generate(&p);
    assert_eq!(g.kernels.len(), p.ops.len());
    assert!(g.kernels.iter().all(|k| k.status == "planned"), "nothing may be generated for the hybrid layout yet");
    assert!(g.files.iter().all(|(_, src)| !src.contains("__global__")), "planned entries have no CUDA source");
    assert!(p.notes.iter().any(|n| n.contains("planned, not generated")));
}

#[test]
fn mixed_precision_needs_a_layer_map_with_known_schemes() {
    let base = |quant: Value| json!({"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000, "quantization_config": quant});
    let q = |layers: Value| json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers});
    let err = |v: Value| format!("{:#}", spec_of(&v).unwrap_err());
    assert!(err(base(json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}))).contains("quantized_layers"));
    assert!(err(base(q(json!({"model.layers.0.mlp": {"quant_algo": "FP8"}})))).contains("model.layers.0.mlp"));
    assert!(err(base(q(json!({"model.layers.0.mlp": {"quant_algo": "MXFP8", "group_size": 16}})))).contains("group_size 16"));
    // A valid map on a dense model is refused: the dense planner does not read per-module formats,
    // so planning it would report every listed module at the default format.
    assert!(err(base(q(json!({"model.layers.0.self_attn.o_proj": {"quant_algo": "MXFP8", "group_size": 32}})))).contains("only for qwen4_exp"));
    // An explicit --format replaces detection, map included.
    let over = ModelSpec::from_json(&base(q(json!({"model.layers.0.mlp": {"quant_algo": "FP8"}}))).to_string(), Some(WeightFormat::Bf16)).unwrap();
    assert!(over.formats.is_empty());
    // Other modelopt algorithms are still rejected.
    assert!(spec_of(&base(json!({"quant_method": "modelopt", "quant_algo": "FP8"}))).is_err());
}

#[test]
fn other_hybrid_layouts_and_inconsistent_schedules_are_rejected() {
    let mut v = raw();
    v["text_config"]["model_type"] = json!("qwen3_next");
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("qwen3_next"));

    let mut v = raw();
    v["text_config"]["layer_types"].as_array_mut().unwrap().pop();
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("47 entries"));

    let mut v = raw();
    v["text_config"]["layer_types"][0] = json!("full_attention");
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("full_attention_interval"));

    let mut v = raw();
    v["text_config"]["layer_types"][1] = json!("sliding_attention");
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("sliding_attention"));

    let mut v = raw();
    v["text_config"].as_object_mut().unwrap().remove("hc_count");
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("hc_count"));

    let mut v = raw();
    v["text_config"]["full_attention_interval"] = json!(0);
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("full_attention_interval"));

    for ids in [json!([]), json!([2, 5])] {
        let mut v = raw();
        v["text_config"]["ple_layer_ids"] = ids;
        assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("exactly one PLE layer"));
    }

    let mut v = raw();
    v["text_config"]["mtp_num_hidden_layers"] = json!(2);
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("disagrees"));

    let mut v = raw();
    v["text_config"]["rope_parameters"]["mrope_section"] = json!([16, 16, 16]);
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("mrope_section"));

    // A schedule of only full attention is an ordinary dense model.
    let dense = json!({"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000,
                       "layer_types": ["full_attention", "full_attention"]});
    assert!(spec_of(&dense).unwrap().hybrid.is_none());
}

#[test]
fn partial_rotary_is_rejected_on_dense_models() {
    let base = |extra: Value| {
        let mut v = json!({"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000});
        v.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        v
    };
    assert!(format!("{:#}", spec_of(&base(json!({"partial_rotary_factor": 0.5}))).unwrap_err()).contains("partial_rotary_factor"));
    assert!(spec_of(&base(json!({"rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.25, "rope_theta": 1e6}}))).is_err());
    let full = spec_of(&base(json!({"rope_parameters": {"rope_type": "default", "partial_rotary_factor": 1.0, "rope_theta": 1e6}}))).unwrap();
    assert_eq!(full.rope_theta, 1e6, "rope_theta is read from rope_parameters");
}

#[test]
fn unused_format_map_entries_are_reported() {
    let mut v = raw();
    for i in 0..4 {
        v["quantization_config"]["quantized_layers"][format!("model.decoder.blocks.{i}.ffn")] = json!({"quant_algo": "MXFP8", "group_size": 32});
    }
    let p = plan_of(&spec_of(&v).unwrap());
    let note = p.notes.iter().find(|n| n.contains("not used by the plan")).expect("unused entries noted");
    for i in 0..4 {
        assert!(note.contains(&format!("model.decoder.blocks.{i}.ffn")), "every unused path is named: {note}");
    }
    // The checkpoint's own map is consumed except its second MTP expert entry: the experts are
    // listed under `mtp.layers.0` and `mtp.layers.48`, and the one MTP layer is index 0.
    assert_eq!(unused_entries(&plan_of(&spec_of(&raw()).unwrap())), vec!["mtp.layers.48.mlp.experts".to_string()]);
}

#[test]
fn the_mtp_count_can_come_from_the_nested_mtp_config() {
    let mut v = raw();
    v["text_config"].as_object_mut().unwrap().remove("mtp_num_hidden_layers");
    let s = spec_of(&v).unwrap();
    assert_eq!(s.hybrid.as_ref().unwrap().mtp_layers, 1);
    let with = plan_of(&spec_of(&raw()).unwrap()).budget.weight_gib;
    assert_eq!(plan_of(&s).budget.weight_gib, with, "draft weights stay in the resident budget");
}

#[test]
fn layers_with_different_formats_are_counted_at_their_own_size() {
    let mut v = raw();
    v["quantization_config"]["quantized_layers"].as_object_mut().unwrap().remove("model.language_model.layers.0.linear_attn.in_proj_qkv");
    let p = plan_of(&spec_of(&v).unwrap());
    let op = |n: &str| p.ops.iter().find(|o| o.name == n).unwrap_or_else(|| panic!("missing {n}"));
    assert_eq!((op("gdn_in_proj_qkv").per_token, op("gdn_in_proj_qkv").format), (35, Some(WeightFormat::Mxfp8)));
    assert_eq!((op("gdn_in_proj_qkv_bf16").per_token, op("gdn_in_proj_qkv_bf16").format), (1, Some(WeightFormat::Bf16)));
    // One BF16 instance adds (2 - 1.03125) bytes per element of one 10240 x 2560 matrix, nothing more.
    let base = plan_of(&spec_of(&raw()).unwrap()).budget.decode_bytes_per_token_ctx_free_gb.unwrap();
    let delta = p.budget.decode_bytes_per_token_ctx_free_gb.unwrap() - base;
    assert!(close(delta, 10240.0 * 2560.0 * (2.0 - 1.03125) / 1e9, 1e-6), "{delta}");
}

#[test]
fn different_hybrid_plans_emit_to_different_directories() {
    let mut v = raw();
    v.as_object_mut().unwrap().remove("quantization_config");
    let a = codegen::generate(&plan_of(&spec_of(&raw()).unwrap()));
    let b = codegen::generate(&plan_of(&spec_of(&v).unwrap()));
    assert_ne!(a.cache_key, b.cache_key, "the same op names with different formats must not share an artifact directory");
}

#[test]
fn dense_plan_json_is_unchanged_from_the_base_revision() {
    // Golden files: `sparkforge plan <config> --json` from main at 8aa1a73, before this change.
    // Update them only for an intentional change to the dense plan.
    for c in ["llama-3.1-8b", "qwen3-30b-a3b"] {
        let s = ModelSpec::from_json(&std::fs::read_to_string(format!("configs/{c}.json")).unwrap(), None).unwrap();
        // Both sides go through the same text round trip: serde_json's default float parser can
        // differ from the in-memory value by one ulp, so parse the current plan's JSON text too.
        let got: Value = serde_json::from_str(&serde_json::to_string_pretty(&plan_of(&s)).unwrap()).unwrap();
        let want: Value = serde_json::from_str(&std::fs::read_to_string(format!("tests/golden/{c}.plan.json")).unwrap()).unwrap();
        assert_eq!(got, want, "{c}: dense plan JSON changed");
    }
}

fn plan_at(s: &ModelSpec, ctx: u32) -> plan::Plan {
    plan::build(s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx })
}

fn ql(v: &mut Value) -> &mut serde_json::Map<String, Value> {
    v["quantization_config"]["quantized_layers"].as_object_mut().unwrap()
}

#[test]
fn shared_gate_and_up_in_different_formats_are_planned_separately() {
    let mut v = raw();
    for l in 0..48 {
        ql(&mut v).remove(&format!("model.language_model.layers.{l}.mlp.shared_expert.up_proj")).unwrap();
    }
    let p = plan_of(&spec_of(&v).unwrap());
    let op = |n: &str| p.ops.iter().find(|o| o.name == n);
    assert!(op("shared_gate_up").is_none(), "incompatible projections are not fused");
    assert_eq!((op("shared_gate").unwrap().per_token, op("shared_gate").unwrap().format), (48, Some(WeightFormat::Mxfp8)));
    assert_eq!((op("shared_up").unwrap().per_token, op("shared_up").unwrap().format), (48, Some(WeightFormat::Bf16)));
    // Each BF16 up projection costs (2 - 1.03125) bytes per element more: 48 x 640 x 2560 elements.
    let base = plan_of(&spec_of(&raw()).unwrap());
    let want = 48.0 * 640.0 * 2560.0 * (2.0 - 1.03125);
    assert_eq!(want, 76_185_600.0);
    let gib = (1u64 << 30) as f64;
    assert!(((p.budget.weight_gib - base.budget.weight_gib) * gib - want).abs() < 1.0, "resident grows by the up matrices");
    let free = |p: &plan::Plan| p.budget.decode_bytes_per_token_ctx_free_gb.unwrap() * 1e9;
    assert!((free(&p) - free(&base) - want).abs() < 1.0, "decode reads grow by the same amount");
}

#[test]
fn a_null_nested_quantization_config_falls_back_to_the_top_level() {
    let mut v = raw();
    v["text_config"]["quantization_config"] = Value::Null;
    assert_eq!(spec_of(&v).unwrap().formats.modules.len(), 435);
    let mut v = raw();
    v["text_config"]["quantization_config"] = json!({"quant_method": "modelopt", "quant_algo": "NVFP4"});
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("different quantization_config"));
}

#[test]
fn other_towers_never_match_text_layer_lookups() {
    let base = plan_of(&spec_of(&raw()).unwrap());
    let mut v = raw();
    ql(&mut v).insert("model.visual.layers.0.mlp.experts".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&spec_of(&v).unwrap());
    let op = |p: &plan::Plan, n: &str| p.ops.iter().find(|o| o.name == n).unwrap().clone();
    assert_eq!(op(&p, "expert_gate_up").format, Some(WeightFormat::Nvfp4), "the vision entry does not change the text experts");
    assert_eq!(p.budget.weight_gib, base.budget.weight_gib);
    assert!(p.notes.iter().any(|n| n.contains("not used by the plan") && n.contains("model.visual.layers.0.mlp.experts")), "and is not consumed");
    // Two paths for one text module that disagree are rejected at parse time.
    let mut v = raw();
    ql(&mut v).insert("layers.0.mlp.experts".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("conflicting formats"));
}

#[test]
fn unplanned_hybrid_shapes_are_rejected_instead_of_mis_budgeted() {
    let mut v = raw();
    v["text_config"]["mtp_num_hidden_layers"] = json!(2);
    v["text_config"]["mtp"]["num_hidden_layers"] = json!(2);
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("at most one MTP layer"));
    for field in ["indexer_compress_ratio", "linear_conv_kernel_dim", "heads_per_ngram", "hc_lowrank"] {
        let mut v = raw();
        v["text_config"][field] = json!(0);
        assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains(field), "{field} = 0 must be a config error");
    }
}

#[test]
fn a_qwen4_exp_variant_without_linear_attention_stays_planned_only() {
    let mut v = raw();
    let n = v["text_config"]["layer_types"].as_array().unwrap().len();
    v["text_config"]["layer_types"] = json!(vec!["full_attention"; n]);
    v["text_config"]["full_attention_interval"] = json!(1);
    let p = plan_of(&spec_of(&v).unwrap());
    assert!(p.model.hybrid.is_some(), "qwen4_exp keeps its HyperConnection, PLE and gated attention");
    let g = codegen::generate(&p);
    assert!(g.kernels.iter().all(|k| k.status == "planned") && g.files.iter().all(|(_, s)| !s.contains("__global__")));
    let mut v = raw();
    v["text_config"].as_object_mut().unwrap().remove("layer_types");
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("missing layer_types"));
}

#[test]
fn tied_embeddings_are_resident_once() {
    let base = plan_of(&spec_of(&raw()).unwrap());
    let mut v = raw();
    v["text_config"]["tie_word_embeddings"] = json!(true);
    v["tie_word_embeddings"] = json!(true);
    let p = plan_of(&spec_of(&v).unwrap());
    let gib = (1u64 << 30) as f64;
    let lm_head = 248_320.0 * 2560.0 * 2.0;
    assert!(((base.budget.weight_gib - p.budget.weight_gib) * gib - lm_head).abs() < 1.0, "one matrix fewer resident");
    assert_eq!(p.budget.decode_bytes_per_token_gb, base.budget.decode_bytes_per_token_gb, "lm_head is still read every token");
}

#[test]
fn resident_parameters_and_fixed_state_are_counted_separately() {
    let s = spec_of(&raw()).unwrap();
    let h = s.hybrid.as_ref().unwrap();
    let params = plan::hybrid_param_bytes(&s, h);
    let get = |n: &str| params.iter().find(|(k, _)| *k == n).unwrap().1;
    assert_eq!(get("gdn_conv_filters"), 36.0 * 10_240.0 * 4.0 * 2.0); // 2,949,120 B
    assert_eq!(get("hc_norms"), 97.0 * 10_240.0 * 4.0);
    // With no context, KV/state is exactly the fixed recurrent state: FP32 state + BF16 conv state.
    let p0 = plan_at(&s, 0);
    let fixed = 36.0 * (48.0 * 128.0 * 128.0 * 4.0 + 10_240.0 * 3.0 * 2.0);
    assert!((p0.budget.kv_gib_at_ctx * (1u64 << 30) as f64 - fixed).abs() < 1.0, "{}", p0.budget.kv_gib_at_ctx);
    // Decode traffic is unchanged by the residency accounting.
    assert!(close(p0.budget.decode_bytes_per_token_ctx_free_gb.unwrap(), 7.338, 0.01));
}

#[test]
fn formats_the_planner_cannot_honor_are_rejected() {
    let mut v = raw();
    ql(&mut v).insert("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("HyperConnection"));
    let mut v = raw();
    ql(&mut v).insert("mtp.layers.48.mlp.experts".into(), json!({"quant_algo": "NVFP4", "group_size": 16}));
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("one MTP layer"));
}

// ---- SHA-783: codex-review round 3 follow-ups on the qwen4_exp plan ----

#[test]
fn a_planned_only_model_is_refused_before_any_cuda_call() {
    // `require_generated` is host-only; `verify` calls it before creating the CUDA engine.
    let g = codegen::generate(&plan_of(&spec_of(&raw()).unwrap()));
    let err = format!("{:#}", codegen::require_generated(&g, "Qwen4ExpForConditionalGeneration").unwrap_err());
    assert!(err.contains("Qwen4ExpForConditionalGeneration") && err.contains("no kernels are generated"), "{err}");
    let llama = ModelSpec::from_json(&std::fs::read_to_string("configs/llama-3.1-8b.json").unwrap(), None).unwrap();
    assert!(codegen::require_generated(&codegen::generate(&plan_of(&llama)), "llama").is_ok());
}

#[test]
fn only_the_model_level_lm_head_sets_the_head_format() {
    let base = plan_of(&spec_of(&raw()).unwrap());
    let mut v = raw();
    ql(&mut v).insert("model.language_model.layers.0.aux_lm_head".into(), json!({"quant_algo": "NVFP4", "group_size": 16}));
    let p = plan_of(&spec_of(&v).unwrap());
    let head = p.ops.iter().find(|o| o.name == "lm_head").unwrap();
    assert_eq!(head.format, Some(WeightFormat::Bf16));
    assert_eq!((p.budget.weight_gib, p.budget.decode_bytes_per_token_gb), (base.budget.weight_gib, base.budget.decode_bytes_per_token_gb));
    assert!(p.notes.iter().any(|n| n.contains("not used by the plan") && n.contains("aux_lm_head")), "{:#?}", p.notes);
    // A real head entry is honored.
    let mut v = raw();
    ql(&mut v).insert("lm_head".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&spec_of(&v).unwrap());
    assert_eq!(p.ops.iter().find(|o| o.name == "lm_head").unwrap().format, Some(WeightFormat::Mxfp8));
}

#[test]
fn mtp_key_and_value_projections_are_priced_separately() {
    let base = plan_of(&spec_of(&raw()).unwrap());
    let mut v = raw();
    ql(&mut v).insert("mtp.layers.0.self_attn.v_proj".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&spec_of(&v).unwrap());
    // v_proj is 512 x 2560: BF16 (2 B) -> MXFP8 (1.03125 B) saves 1,269,760 B; k_proj stays BF16.
    let delta = (base.budget.weight_gib - p.budget.weight_gib) * (1u64 << 30) as f64;
    assert!((delta - 512.0 * 2560.0 * (2.0 - 1.03125)).abs() < 1.0, "{delta}");
    assert!(!unused_entries(&p).contains(&"mtp.layers.0.self_attn.v_proj".to_string()), "the v_proj entry is consumed: {:#?}", p.notes);
}

#[test]
fn the_top_level_model_type_identifies_qwen4_exp() {
    let mut v = raw();
    v["text_config"].as_object_mut().unwrap().remove("model_type");
    let p = plan_of(&spec_of(&v).unwrap());
    assert!(p.model.hybrid.is_some());
    assert!(codegen::generate(&p).kernels.iter().all(|k| k.status == "planned"));
}

#[test]
fn moe_top_k_must_be_between_one_and_the_expert_count() {
    for k in [0, 513] {
        let mut v = raw();
        v["text_config"]["num_experts_per_tok"] = json!(k);
        assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("num_experts_per_tok"), "top-k {k}");
    }
}

#[test]
fn a_format_override_can_skip_a_malformed_layer_map() {
    let cfg = json!({"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {"layers.0.self_attn.o_proj": null}}});
    let s = ModelSpec::from_json(&cfg.to_string(), Some(WeightFormat::Bf16)).unwrap();
    assert_eq!((s.weights, s.formats.is_empty(), s.declared_mixed), (WeightFormat::Bf16, true, true));
    assert!(format!("{:#}", spec_of(&cfg).unwrap_err()).contains("layers.0.self_attn.o_proj"));
}

// ---- SHA-783 round 1 follow-ups ----

#[test]
fn a_format_override_skips_malformed_quantization_fields() {
    let cfg = json!({"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000,
        "quantization_config": {"quant_method": "awq", "bits": "4", "group_size": 128}});
    assert_eq!(ModelSpec::from_json(&cfg.to_string(), Some(WeightFormat::Bf16)).unwrap().weights, WeightFormat::Bf16);
    assert!(format!("{:#}", spec_of(&cfg).unwrap_err()).contains("quantization_config"));
    // A mixed checkpoint with a malformed field is still recognized as mixed under an override.
    let mixed = json!({"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "group_size": "x"}});
    assert!(ModelSpec::from_json(&mixed.to_string(), Some(WeightFormat::Bf16)).unwrap().declared_mixed);
}

#[test]
fn each_mtp_shared_expert_projection_is_priced_at_its_own_format() {
    let base = plan_of(&spec_of(&raw()).unwrap()).budget.weight_gib;
    let one = 640.0 * 2560.0 * (2.0 - 1.03125); // one 640 x 2560 matrix, BF16 -> MXFP8
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let mut v = raw();
        ql(&mut v).insert(format!("mtp.layers.0.mlp.shared_expert.{proj}"), json!({"quant_algo": "MXFP8", "group_size": 32}));
        let p = plan_of(&spec_of(&v).unwrap());
        let delta = (base - p.budget.weight_gib) * (1u64 << 30) as f64;
        assert!((delta - one).abs() < 1.0, "{proj}: {delta}");
        assert!(!unused_entries(&p).contains(&format!("mtp.layers.0.mlp.shared_expert.{proj}")), "{proj} entry consumed");
    }
}

#[test]
fn auxiliary_paths_never_stand_in_for_ple_or_mtp_modules() {
    let base = plan_of(&spec_of(&raw()).unwrap());
    for aux in ["model.language_model.layers.0.aux.ple.key_proj", "mtp.layers.0.aux.self_attn.v_proj"] {
        let mut v = raw();
        ql(&mut v).insert(aux.into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
        let p = plan_of(&spec_of(&v).unwrap());
        assert_eq!((p.budget.weight_gib, p.budget.decode_bytes_per_token_gb), (base.budget.weight_gib, base.budget.decode_bytes_per_token_gb), "{aux}");
        assert!(p.notes.iter().any(|n| n.contains("not used by the plan") && n.contains(aux)), "{aux} reported unused");
    }
    // The genuine paths are honored.
    let mut v = raw();
    ql(&mut v).insert("model.language_model.layers.1.ple.key_proj".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&spec_of(&v).unwrap());
    assert_eq!(p.ops.iter().find(|o| o.name == "ple_key_proj").unwrap().format, Some(WeightFormat::Mxfp8));
}

// ---- SHA-783 round 2 follow-ups ----

#[test]
fn the_ple_table_is_recognized_at_its_model_level_path() {
    let mut v = raw();
    ql(&mut v).remove("model.language_model.layers.1.ple.ple_embedding.ngram_embedding").unwrap();
    let bf16 = plan_of(&spec_of(&v).unwrap());
    ql(&mut v).insert("model.language_model.ple_embedding.ngram_embedding".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&spec_of(&v).unwrap());
    assert!(!unused_entries(&p).contains(&"model.language_model.ple_embedding.ngram_embedding".to_string()), "the model-level table entry is consumed: {:#?}", p.notes);
    // 320,000,000 rows x 160 elements, BF16 (2 B) -> MXFP8 (1.03125 B), resident; 16 rows per token decoded.
    let gib = (1u64 << 30) as f64;
    let resident = (bf16.budget.weight_gib - p.budget.weight_gib) * gib;
    assert!((resident - 320_000_000.0 * 160.0 * (2.0 - 1.03125)).abs() < 1e3, "{resident}");
    let decode = (bf16.budget.decode_bytes_per_token_gb - p.budget.decode_bytes_per_token_gb) * 1e9;
    assert!((decode - 16.0 * 160.0 * (2.0 - 1.03125)).abs() < 1e-3, "{decode}");
}

#[test]
fn ple_entries_only_count_at_the_configured_ple_layer() {
    let base = plan_of(&spec_of(&raw()).unwrap());
    let mut v = raw();
    ql(&mut v).insert("model.language_model.layers.999.ple.key_proj".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&spec_of(&v).unwrap());
    assert_eq!((p.budget.weight_gib, p.budget.decode_bytes_per_token_gb), (base.budget.weight_gib, base.budget.decode_bytes_per_token_gb));
    assert!(p.notes.iter().any(|n| n.contains("not used by the plan") && n.contains("layers.999.ple.key_proj")));
    // The configured placement (ple_layer_ids [2]) and the layer the checkpoint files it under (1) both count.
    for l in [1, 2] {
        let mut v = raw();
        ql(&mut v).insert(format!("model.language_model.layers.{l}.ple.key_proj"), json!({"quant_algo": "MXFP8", "group_size": 32}));
        let p = plan_of(&spec_of(&v).unwrap());
        assert_eq!(p.ops.iter().find(|o| o.name == "ple_key_proj").unwrap().format, Some(WeightFormat::Mxfp8), "layer {l}");
    }
}

#[test]
fn an_absent_mtp_shared_expert_leaves_its_entries_unused() {
    let mut v = raw();
    v["text_config"]["shared_expert_intermediate_size"] = json!(0);
    for l in 0..48 {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            ql(&mut v).remove(&format!("model.language_model.layers.{l}.mlp.shared_expert.{proj}"));
        }
    }
    ql(&mut v).insert("mtp.layers.0.mlp.shared_expert.up_proj".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&spec_of(&v).unwrap());
    assert!(p.notes.iter().any(|n| n.contains("not used by the plan") && n.contains("mtp.layers.0.mlp.shared_expert.up_proj")), "{:#?}", p.notes);
}

// ---- archive review (swesty/sparkforge#12) ----

#[test]
fn moe_without_n_shared_experts_has_no_shared_expert() {
    let s = ModelSpec::from_json(&json!({"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000,
        "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768}).to_string(), None).unwrap();
    let e = s.moe.unwrap();
    assert_eq!((e.shared_experts, e.shared_intermediate), (0, 0));
}

#[test]
fn an_override_also_skips_a_nested_quantization_conflict() {
    let mut v = json!({"text_config": {"hidden_size": 2048, "num_hidden_layers": 2, "num_attention_heads": 16, "vocab_size": 1000,
        "quantization_config": {"quant_method": "awq", "bits": "4", "group_size": 128}},
        "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}});
    assert_eq!(ModelSpec::from_json(&v.to_string(), Some(WeightFormat::Bf16)).unwrap().weights, WeightFormat::Bf16);
    assert!(format!("{:#}", spec_of(&v).unwrap_err()).contains("different quantization_config"));
    // A mixed-precision declaration at either level survives the override.
    v["quantization_config"] = json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"});
    assert!(ModelSpec::from_json(&v.to_string(), Some(WeightFormat::Bf16)).unwrap().declared_mixed);
}

// ---- SHA-785: formats from the checkpoint's module identities ----

/// Tensor names laid out as in the Flash Next checkpoint (one `.weight` per module is enough for
/// identities; the real checkpoint adds scales beside them): every text layer by kind, all routed
/// experts, the PLE at `layers.1`, one MTP layer at index 0, and `lm_head` unless tied.
fn checkpoint_names(s: &ModelSpec) -> Vec<String> {
    let h = s.hybrid.as_ref().unwrap();
    let experts = s.moe.as_ref().unwrap().num_experts;
    let t = |rest: &str| format!("model.language_model.{rest}.weight");
    let mut v = vec![t("embed_tokens")];
    if !s.tie_embeddings {
        v.push("lm_head.weight".into());
    }
    let block = |v: &mut Vec<String>, pre: &str, kind: LayerKind| {
        let attn: &[&str] = match kind {
            LayerKind::LinearAttention => &["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj"],
            LayerKind::FullAttention => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.indexer.index_qk_proj"],
        };
        for m in attn.iter().chain(&["mlp.gate", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]) {
            v.push(format!("{pre}.{m}.weight"));
        }
        for e in 0..experts {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                v.push(format!("{pre}.mlp.experts.{e}.{p}.weight"));
            }
        }
    };
    for (i, &k) in h.schedule.iter().enumerate() {
        block(&mut v, &format!("model.language_model.layers.{i}"), k);
    }
    for &p in &h.ple.layer_ids {
        for m in ["ple.key_proj", "ple.value_proj", "ple.ple_embedding.ngram_embedding.shard_0"] {
            v.push(t(&format!("layers.{}.{m}", p - 1)));
        }
    }
    for i in 0..h.mtp_layers {
        block(&mut v, &format!("mtp.layers.{i}"), LayerKind::FullAttention);
    }
    v
}

fn resolved(v: &Value) -> anyhow::Result<ModelSpec> {
    let mut s = spec_of(v)?;
    let names = checkpoint_names(&s);
    s.resolve_modules(names.iter().map(String::as_str))?;
    Ok(s)
}

#[test]
fn a_checkpoint_resolved_plan_matches_the_config_plan_for_the_real_map() {
    let by_path = plan_of(&spec_of(&raw()).unwrap());
    let exact = plan_of(&resolved(&raw()).unwrap());
    assert_eq!(exact.budget.weight_gib, by_path.budget.weight_gib);
    assert_eq!(exact.budget.decode_bytes_per_token_gb, by_path.budget.decode_bytes_per_token_gb);
    // The second MTP expert entry names index 48; the checkpoint's one MTP layer is index 0.
    assert_eq!(unused_entries(&exact), vec!["mtp.layers.48.mlp.experts".to_string()]);
}

#[test]
fn overlapping_entries_must_agree_on_every_module_they_cover() {
    // Two entries that cover the same table modules: agreeing is fine, disagreeing is an error
    // naming both entries and the module.
    let mut v = raw();
    ql(&mut v).insert("model.language_model.layers.1.ple.ple_embedding".into(), json!({"quant_algo": "NVFP4", "group_size": 16}));
    let p = plan_of(&resolved(&v).unwrap());
    assert!(!unused_entries(&p).contains(&"model.language_model.layers.1.ple.ple_embedding".to_string()));
    ql(&mut v).insert("model.language_model.layers.1.ple.ple_embedding".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let err = format!("{:#}", resolved(&v).unwrap_err());
    assert!(err.contains("model.language_model.layers.1.ple.ple_embedding ")
        && err.contains("model.language_model.layers.1.ple.ple_embedding.ngram_embedding")
        && err.contains("conflicting formats"), "{err}");
}

#[test]
fn an_alias_that_names_no_checkpoint_module_is_unused_not_honored() {
    // codex-review fp:198fb71ba126: the bare `ple_embedding.ngram_embedding` was accepted beside
    // the real `layers.1.ple...` entry without the two being compared. In the checkpoint the
    // table lives under `layers.1.ple`, so the bare path covers nothing: it is reported, and the
    // table keeps the real entry's NVFP4 price.
    let base = plan_of(&resolved(&raw()).unwrap());
    let mut v = raw();
    ql(&mut v).insert("model.language_model.ple_embedding.ngram_embedding".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let p = plan_of(&resolved(&v).unwrap());
    assert!(unused_entries(&p).contains(&"model.language_model.ple_embedding.ngram_embedding".to_string()), "{:#?}", p.notes);
    assert_eq!(p.budget.weight_gib, base.budget.weight_gib);
    let table = |p: &plan::Plan| p.ops.iter().find(|o| o.name == "ple_ngram_gather").unwrap().format;
    assert_eq!(table(&p), table(&base));
    assert_ne!(table(&p), Some(WeightFormat::Mxfp8));
}

#[test]
fn a_stale_mtp_index_never_prices_the_draft_layer() {
    // codex-review fp:8b1d8ca98265: `mtp.layers.999.self_attn.v_proj` used to price the real draft
    // projection because any index matched. With the checkpoint, index 999 holds no module; from
    // the config alone, it is beyond the one configured MTP layer. Either way it is unused.
    let mut v = raw();
    ql(&mut v).insert("mtp.layers.999.self_attn.v_proj".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    for (label, base, p) in [
        ("checkpoint", plan_of(&resolved(&raw()).unwrap()), plan_of(&resolved(&v).unwrap())),
        ("config", plan_of(&spec_of(&raw()).unwrap()), plan_of(&spec_of(&v).unwrap())),
    ] {
        assert_eq!(p.budget.weight_gib, base.budget.weight_gib, "{label}");
        assert!(unused_entries(&p).contains(&"mtp.layers.999.self_attn.v_proj".to_string()), "{label}: {:#?}", p.notes);
    }
}

#[test]
fn a_tied_head_has_the_embedding_format_and_one_resident_copy() {
    // codex-review fp:794525984166: with tied weights, lm_head ignored a format declared under
    // embed_tokens. Tied, the head is the embedding matrix, so an MXFP8 embedding makes an MXFP8
    // head, counted once.
    let mut v = raw();
    v["tie_word_embeddings"] = json!(true);
    if let Some(t) = v.get_mut("text_config") { t["tie_word_embeddings"] = json!(true); }
    let bf16 = plan_of(&spec_of(&v).unwrap());
    ql(&mut v).insert("model.language_model.embed_tokens".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let (s_cfg, s_ck) = (spec_of(&v).unwrap(), resolved(&v).unwrap());
    assert!(s_cfg.tie_embeddings);
    let gib = (1u64 << 30) as f64;
    let once = 248_320.0 * 2560.0 * (2.0 - 1.03125);
    for (label, p) in [("config", plan_of(&s_cfg)), ("checkpoint", plan_of(&s_ck))] {
        assert_eq!(p.ops.iter().find(|o| o.name == "lm_head").unwrap().format, Some(WeightFormat::Mxfp8), "{label}");
        let saved = (bf16.budget.weight_gib - p.budget.weight_gib) * gib;
        assert!((saved - once).abs() < 1e3, "{label}: resident once, saved {saved} vs {once}");
        assert!(!unused_entries(&p).contains(&"model.language_model.embed_tokens".to_string()), "{label}");
    }
}

// ---- SHA-785: codex-review cycle 1 follow-ups ----

fn resolved_with(v: &Value, extra: &[&str]) -> anyhow::Result<ModelSpec> {
    let mut s = spec_of(v)?;
    let mut names = checkpoint_names(&s);
    names.extend(extra.iter().map(|e| e.to_string()));
    s.resolve_modules(names.iter().map(String::as_str))?;
    Ok(s)
}

#[test]
fn a_tied_head_entry_that_disagrees_with_the_embedding_is_rejected() {
    let mut v = raw();
    v["tie_word_embeddings"] = json!(true);
    if let Some(t) = v.get_mut("text_config") { t["tie_word_embeddings"] = json!(true); }
    ql(&mut v).insert("model.language_model.embed_tokens".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    ql(&mut v).insert("lm_head".into(), json!({"quant_algo": "NVFP4", "group_size": 16}));
    let err = format!("{:#}", spec_of(&v).unwrap_err());
    assert!(err.contains("tied") && err.contains("lm_head") && err.contains("embed_tokens"), "config: {err}");
    // From a checkpoint that also holds an lm_head tensor, the same conflict is caught on resolve.
    ql(&mut v).remove("lm_head");
    let mut s = spec_of(&v).unwrap();
    let mut names = checkpoint_names(&s);
    names.push("lm_head.weight".into());
    s.formats.modules.insert("lm_head".into(), WeightFormat::Nvfp4);
    let err = format!("{:#}", s.resolve_modules(names.iter().map(String::as_str)).unwrap_err());
    assert!(err.contains("tied") && err.contains("one format"), "checkpoint: {err}");
}

#[test]
fn a_ple_table_split_across_formats_is_rejected() {
    // The planner prices the table as one format. One shard declared MXFP8 and another left at
    // the default is a table it can't price, so resolution refuses it instead of collapsing it.
    let mut v = raw();
    ql(&mut v).remove("model.language_model.layers.1.ple.ple_embedding.ngram_embedding").unwrap();
    ql(&mut v).insert("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let err = format!("{:#}", resolved_with(&v, &["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight"]).unwrap_err());
    assert!(err.contains("ngram_embedding") && err.contains("mixed formats"), "{err}");
}

#[test]
fn checkpoint_mtp_layers_must_be_the_configured_ones() {
    // An extra live index (mtp.layers.48 tensors beside the one configured layer) is diagnosed
    // instead of letting its default format reprice layer 0.
    let mut v = raw();
    ql(&mut v).insert("mtp.layers.0.self_attn.v_proj".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let err = format!("{:#}", resolved_with(&v, &["mtp.layers.48.self_attn.v_proj.weight"]).unwrap_err());
    assert!(err.contains("MTP layer") && err.contains("48"), "{err}");
    assert!(resolved_with(&v, &[]).is_ok());
}

#[test]
fn an_entry_whose_modules_the_plan_never_prices_is_reported() {
    // The vision tower's experts exist in the checkpoint, so the entry covers modules, but no
    // text op uses them: it is still reported as unused, and text layer 0's experts keep NVFP4.
    let mut v = raw();
    ql(&mut v).insert("model.visual.layers.0.mlp.experts".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let base = plan_of(&resolved(&raw()).unwrap());
    let p = plan_of(&resolved_with(&v, &["model.visual.layers.0.mlp.experts.0.up_proj.weight"]).unwrap());
    assert!(unused_entries(&p).contains(&"model.visual.layers.0.mlp.experts".to_string()), "{:#?}", p.notes);
    assert_eq!(p.budget.weight_gib, base.budget.weight_gib);
}

#[test]
fn the_embedding_stays_bf16_whatever_the_general_weight_format() {
    // `--format nvfp4` sets the linears' format; the embedding table stays BF16 unless an entry
    // quantizes it, in its resident bytes and its decode gather.
    let s = ModelSpec::from_json(&raw().to_string(), Some(WeightFormat::Nvfp4)).unwrap();
    assert_eq!(s.weights, WeightFormat::Nvfp4);
    let p = plan_of(&s);
    let gather = p.ops.iter().find(|o| o.name == "embed_gather").unwrap();
    assert_eq!(gather.format, Some(WeightFormat::Bf16));
    assert!((gather.bytes_per_token - 2560.0 * 2.0).abs() < 1e-9, "{}", gather.bytes_per_token);
}

// ---- SHA-785: codex-review cycle 2 follow-ups ----

#[test]
fn a_model_level_ple_table_split_across_formats_is_rejected() {
    let mut v = raw();
    ql(&mut v).remove("model.language_model.layers.1.ple.ple_embedding.ngram_embedding").unwrap();
    ql(&mut v).insert("model.language_model.ple_embedding.ngram_embedding.shard_0".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let err = format!("{:#}", resolved_with(&v, &[
        "model.language_model.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.ple_embedding.ngram_embedding.shard_1.weight",
    ]).unwrap_err());
    assert!(err.contains("ple_embedding.ngram_embedding") && err.contains("mixed formats"), "{err}");
}

#[test]
fn more_than_one_mtp_layer_is_refused_before_formats_matter() {
    // codex-review fp:752cfe280501 (two MTP layers in different formats priced at one) can't
    // reach the planner: a config with more than one MTP layer is refused when parsed.
    let mut v = raw();
    fn set_mtp(v: &mut Value) {
        if let Some(o) = v.as_object_mut() {
            for (k, x) in o.iter_mut() {
                if k == "mtp_num_hidden_layers" { *x = json!(2); }
                if k == "mtp" { if let Some(n) = x.get_mut("num_hidden_layers") { *n = json!(2); } }
                set_mtp(x);
            }
        }
    }
    set_mtp(&mut v);
    let err = format!("{:#}", spec_of(&v).unwrap_err());
    assert!(err.contains("at most one MTP layer"), "{err}");
}

#[test]
fn mtp_layer_indices_must_be_canonical() {
    let err = format!("{:#}", resolved_with(&raw(), &["mtp.layers.00.self_attn.v_proj.weight"]).unwrap_err());
    assert!(err.contains("mtp.layers.00") && err.contains("canonical"), "{err}");
}

#[test]
fn every_spelling_of_a_tied_head_must_agree() {
    let mut v = raw();
    v["tie_word_embeddings"] = json!(true);
    if let Some(t) = v.get_mut("text_config") { t["tie_word_embeddings"] = json!(true); }
    for k in ["model.language_model.embed_tokens", "lm_head"] {
        ql(&mut v).insert(k.into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    }
    spec_of(&v).unwrap(); // agreeing spellings
    ql(&mut v).insert("model.lm_head".into(), json!({"quant_algo": "NVFP4", "group_size": 16}));
    let err = format!("{:#}", spec_of(&v).unwrap_err());
    assert!(err.contains("model.lm_head") && (err.contains("tied") || err.contains("conflicting")), "{err}");
}

#[test]
fn tensors_kept_under_a_projection_never_reprice_it() {
    // codex-review fp:708c7a7b8993: layer 3's q_proj has no entry (BF16 default) and only an
    // observer's tensor under it is declared MXFP8. Pricing q_proj looks at q_proj itself, so the
    // projection stays BF16 and the observer entry is reported unused.
    let mut v = raw();
    ql(&mut v).remove("model.language_model.layers.3.self_attn.q_proj").unwrap();
    let base = plan_of(&resolved(&v).unwrap());
    ql(&mut v).insert("model.language_model.layers.3.self_attn.q_proj.observer".into(), json!({"quant_algo": "MXFP8", "group_size": 32}));
    let s = resolved_with(&v, &["model.language_model.layers.3.self_attn.q_proj.observer.weight"]).unwrap();
    assert!(s.hybrid.as_ref().unwrap().schedule[3] == LayerKind::FullAttention, "layer 3 is a sparse-attention layer");
    let p = plan_of(&s);
    assert_eq!(p.budget.weight_gib, base.budget.weight_gib);
    assert!(unused_entries(&p).contains(&"model.language_model.layers.3.self_attn.q_proj.observer".to_string()), "{:#?}", p.notes);
}

#[test]
fn agreeing_overlapping_entries_are_both_used() {
    let mut v = raw();
    ql(&mut v).insert("model.language_model.layers.1.ple.ple_embedding".into(), json!({"quant_algo": "NVFP4", "group_size": 16}));
    let p = plan_of(&resolved(&v).unwrap());
    let unused = unused_entries(&p);
    for e in ["model.language_model.layers.1.ple.ple_embedding", "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"] {
        assert!(!unused.contains(&e.to_string()), "{e}: {unused:?}");
    }
}

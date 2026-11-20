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
    // The checkpoint's own map is fully consumed, including its duplicate MTP expert entry.
    assert!(!plan_of(&spec_of(&raw()).unwrap()).notes.iter().any(|n| n.contains("not used by the plan")));
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

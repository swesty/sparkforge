//! Hybrid (`qwen4_exp`) checkpoint fixtures shared by the planning and loading tests.
//!
//! `hybrid_tensors` writes out, by hand, the tensor layout of the reference checkpoint
//! (`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision 6909a5bed089, from its 40 shard
//! headers): names, dtypes and shapes, with the reference's formats. It deliberately doesn't use
//! `sparkforge::inventory`, so the loader's inventory is checked against an independent copy.
#![allow(dead_code)]
use serde_json::{json, Value};
use sparkforge::config::{LayerKind, ModelSpec};

pub const CFG: &str = "configs/qwen3.8-flash-next-nvfp4.json";

pub fn raw() -> Value {
    serde_json::from_str(&std::fs::read_to_string(CFG).unwrap()).unwrap()
}

/// One tensor: name, dtype, shape.
pub type Tensor = (String, &'static str, Vec<usize>);

/// Every tensor of a checkpoint for `s` laid out like the reference: the text model under
/// `model.language_model.`, the PLE under `ple` (a canonical path such as `layers.1.ple`), PLE
/// table shards of `table_rows` rows, the MTP layer under `mtp.`, and two vision-tower tensors.
/// Formats are the reference's: MXFP8 attention, GDN and shared-expert projections, NVFP4 routed
/// experts and PLE table, W4A16 NVFP4 MTP experts, and BF16 for everything else.
pub fn hybrid_tensors(s: &ModelSpec, ple: &str, table_rows: usize) -> Vec<Tensor> {
    let h = s.hybrid.as_ref().unwrap();
    let e = s.moe.as_ref().unwrap();
    let (hd, res, lr) = (s.hidden as usize, (h.hc.streams * s.hidden) as usize, h.hc.lowrank as usize);
    let mut v: Vec<Tensor> = vec![];
    let t = |rest: &str| format!("model.language_model.{rest}");
    let bf16 = |v: &mut Vec<Tensor>, name: String, shape: Vec<usize>| v.push((format!("{name}.weight"), "BF16", shape));
    let mxfp8 = |v: &mut Vec<Tensor>, name: String, n: usize, k: usize| {
        v.push((format!("{name}.weight"), "F8_E4M3", vec![n, k]));
        v.push((format!("{name}.weight_scale"), "U8", vec![n, k / 32]));
    };
    let nvfp4 = |v: &mut Vec<Tensor>, name: String, n: usize, k: usize, input_scale: bool| {
        v.push((format!("{name}.weight"), "U8", vec![n, k / 2]));
        v.push((format!("{name}.weight_scale"), "F8_E4M3", vec![n, k / 16]));
        v.push((format!("{name}.weight_scale_2"), "F32", vec![]));
        if input_scale {
            v.push((format!("{name}.input_scale"), "F32", vec![]));
        }
    };
    let mixer = |v: &mut Vec<Tensor>, p: String, inject: bool, norm: &'static str| {
        if inject {
            v.push((format!("{p}.block_inject_weight.weight"), "BF16", vec![h.hc.streams as usize, res]));
        }
        v.push((format!("{p}.hc_norm.weight"), norm, vec![res]));
        v.push((format!("{p}.input_mix_weight_down.weight"), "BF16", vec![lr, res]));
        v.push((format!("{p}.input_mix_weight_up.weight"), "BF16", vec![res, lr]));
    };
    let (q, kv, ei, si, ne) = (s.q_dim() as usize, s.kv_dim() as usize, e.expert_intermediate as usize, e.shared_intermediate as usize, e.num_experts as usize);
    let iqk = h.qsa.index_qk_dim() as usize;
    // A layer: attention (MXFP8 in the main model, BF16 in the MTP layer), router, experts, shared
    // expert, and two mixers.
    let block = |v: &mut Vec<Tensor>, p: String, kind: LayerKind, mtp: bool| {
        let quant = |v: &mut Vec<Tensor>, name: String, n: usize, k: usize| if mtp { bf16(v, name, vec![n, k]) } else { mxfp8(v, name, n, k) };
        match kind {
            LayerKind::LinearAttention => {
                let g = &h.gdn;
                let a = format!("{p}.linear_attn");
                let (qkv, vd, vh) = (g.qkv_dim() as usize, g.value_dim() as usize, g.value_heads as usize);
                for (rel, n, k) in [("in_proj_qkv", qkv, hd), ("in_proj_z", vd, hd), ("in_proj_a", vh, hd), ("in_proj_b", vh, hd), ("out_proj", hd, vd)] {
                    quant(v, format!("{a}.{rel}"), n, k);
                }
                v.push((format!("{a}.A_log"), "BF16", vec![vh]));
                v.push((format!("{a}.dt_bias"), "BF16", vec![vh]));
                bf16(v, format!("{a}.conv1d"), vec![qkv, 1, g.conv_kernel as usize]);
                bf16(v, format!("{a}.norm"), vec![g.value_head_dim as usize]);
            }
            LayerKind::FullAttention => {
                let a = format!("{p}.self_attn");
                for (rel, n, k) in [("q_proj", 2 * q, hd), ("k_proj", kv, hd), ("v_proj", kv, hd), ("o_proj", hd, q), ("indexer.index_qk_proj", iqk, hd)] {
                    quant(v, format!("{a}.{rel}"), n, k);
                }
                bf16(v, format!("{a}.q_norm"), vec![s.head_dim as usize]);
                bf16(v, format!("{a}.k_norm"), vec![s.head_dim as usize]);
                bf16(v, format!("{a}.indexer.q_layernorm"), vec![h.qsa.indexer_head_dim as usize]);
                bf16(v, format!("{a}.indexer.k_layernorm"), vec![h.qsa.indexer_head_dim as usize]);
            }
        }
        bf16(v, format!("{p}.mlp.gate"), vec![ne, hd]);
        for x in 0..ne {
            for (rel, n, k) in [("gate_proj", ei, hd), ("up_proj", ei, hd), ("down_proj", hd, ei)] {
                nvfp4(v, format!("{p}.mlp.experts.{x}.{rel}"), n, k, !mtp);
            }
        }
        for (rel, n, k) in [("gate_proj", si, hd), ("up_proj", si, hd), ("down_proj", hd, si)] {
            quant(v, format!("{p}.mlp.shared_expert.{rel}"), n, k);
        }
        bf16(v, format!("{p}.mlp.shared_expert_gate"), vec![1, hd]);
        let norm = if mtp { "BF16" } else { "F32" };
        mixer(v, format!("{p}.attn_hyper_connection"), true, norm);
        mixer(v, format!("{p}.mlp_hyper_connection"), true, norm);
    };
    bf16(&mut v, t("embed_tokens"), vec![s.vocab as usize, hd]);
    if !s.tie_embeddings {
        bf16(&mut v, "lm_head".into(), vec![s.vocab as usize, hd]);
    }
    mixer(&mut v, t("hyper_connection_mixer"), false, "F32");
    for (i, &k) in h.schedule.iter().enumerate() {
        block(&mut v, t(&format!("layers.{i}")), k, false);
    }
    let p = &h.ple;
    let ple = t(ple);
    bf16(&mut v, format!("{ple}.key_proj"), vec![res, p.embed_dim as usize]);
    bf16(&mut v, format!("{ple}.value_proj"), vec![hd, p.embed_dim as usize]);
    bf16(&mut v, format!("{ple}.conv1d"), vec![res, 1, p.conv_kernel as usize]);
    for n in ["norm_conv", "norm_key", "norm_query"] {
        v.push((format!("{ple}.{n}.weight"), "F32", vec![res]));
    }
    let heads = p.heads() as usize;
    v.push((format!("{ple}.ple_embedding.layer_multipliers"), "I64", vec![p.ngram_size as usize]));
    v.push((format!("{ple}.ple_embedding.ngram_heads_offsets"), "I64", vec![heads]));
    v.push((format!("{ple}.ple_embedding.ngram_heads_vocab_sizes"), "I64", vec![heads]));
    let table = format!("{ple}.ple_embedding.ngram_embedding");
    v.push((format!("{table}.weight_scale_2"), "F32", vec![1]));
    let hdim = p.head_dim() as usize;
    for j in 0..p.split_parts.unwrap() as usize {
        v.push((format!("{table}.shard_{j}.weight"), "U8", vec![table_rows, hdim / 2]));
        v.push((format!("{table}.shard_{j}.weight_scale"), "F8_E4M3", vec![table_rows, hdim / 16]));
    }
    if h.mtp_layers > 0 {
        bf16(&mut v, "mtp.fc_embedding".into(), vec![hd, hd]);
        bf16(&mut v, "mtp.fc_hidden".into(), vec![hd, hd]);
        bf16(&mut v, "mtp.pre_fc_norm_embedding".into(), vec![hd]);
        bf16(&mut v, "mtp.pre_fc_norm_hidden".into(), vec![res]);
        mixer(&mut v, "mtp.hyper_connection_mixer".into(), false, "BF16");
    }
    for i in 0..h.mtp_layers {
        block(&mut v, format!("mtp.layers.{i}"), LayerKind::FullAttention, true);
    }
    v.push(("model.visual.blocks.0.attn.qkv.weight".into(), "BF16", vec![12, 4]));
    v.push(("model.visual.blocks.0.attn.qkv.bias".into(), "BF16", vec![12]));
    v
}

/// A Flash Next config scaled down to a checkpoint a test can write: 4 layers (3 GDN, 1 sparse
/// attention), hidden 64, 4 experts, a 4-head PLE table in 4 shards, one MTP layer, and the
/// reference's `quantized_layers` scheme for those modules.
pub fn tiny_config() -> Value {
    let mut v = raw();
    let t = v["text_config"].as_object_mut().unwrap();
    for (k, x) in [
        ("hidden_size", json!(64)), ("num_hidden_layers", json!(4)), ("num_attention_heads", json!(2)), ("num_key_value_heads", json!(1)),
        ("head_dim", json!(32)), ("vocab_size", json!(128)), ("num_experts", json!(4)), ("num_experts_per_tok", json!(2)),
        ("moe_intermediate_size", json!(32)), ("shared_expert_intermediate_size", json!(32)), ("linear_num_key_heads", json!(2)),
        ("linear_num_value_heads", json!(2)), ("linear_key_head_dim", json!(16)), ("linear_value_head_dim", json!(16)),
        ("indexer_n_heads", json!(1)), ("indexer_head_dim", json!(16)), ("hc_lowrank", json!(8)), ("ple_embed_dim", json!(64)),
        ("heads_per_ngram", json!(2)), ("ngram_vocab_size_base", json!(10)), ("split_ngram_parts", json!(4)),
        ("make_ngram_vocab_size_divisible_by", json!(4)),
        ("layer_types", json!(["linear_attention", "linear_attention", "linear_attention", "full_attention"])),
    ] {
        t.insert(k.into(), x);
    }
    t.get_mut("rope_parameters").unwrap()["mrope_section"] = json!([2, 1, 1]);
    let mut q = serde_json::Map::new();
    let fmt = |algo: &str, g: u32| json!({"quant_algo": algo, "group_size": g});
    for l in 0..4 {
        let p = format!("model.language_model.layers.{l}");
        q.insert(format!("{p}.mlp.experts"), fmt("NVFP4", 16));
        let attn: &[&str] = if l == 3 {
            &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.indexer.index_qk_proj"]
        } else {
            &["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj"]
        };
        for m in attn.iter().chain(&["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]) {
            q.insert(format!("{p}.{m}"), fmt("MXFP8", 32));
        }
    }
    q.insert("model.language_model.layers.1.ple.ple_embedding.ngram_embedding".into(), fmt("NVFP4", 16));
    q.insert("mtp.layers.0.mlp.experts".into(), fmt("W4A16_NVFP4", 16));
    v["quantization_config"]["quantized_layers"] = Value::Object(q);
    v
}

/// The tiny config's per-head PLE table sizes (each at least `ngram_vocab_size_base` 10): 60 rows,
/// already a multiple of 4, in 4 shards of 15 rows.
pub const TINY_VOCAB_SIZES: [i64; 4] = [11, 13, 17, 19];
pub const TINY_TABLE_ROWS: usize = 15;

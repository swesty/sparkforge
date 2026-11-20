use sparkforge::config::{ModelSpec, WeightFormat};
use sparkforge::weights::{bind, check_headroom, Checkpoint, HEADROOM_BYTES};
use std::path::PathBuf;

// Tiny Llama-shaped model: hidden 16, 2 layers, 2 heads / 1 kv head of 8, inter 32, vocab 64.
const CFG: &str = r#"{"architectures": ["LlamaForCausalLM"], "hidden_size": 16, "intermediate_size": 32,
  "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
  "vocab_size": 64, "tie_word_embeddings": TIE}"#;

fn spec(tie: bool) -> ModelSpec {
    ModelSpec::from_json(&CFG.replace("TIE", if tie { "true" } else { "false" }), None).unwrap()
}

/// Every tensor the tiny model needs, as (name, shape).
fn tensors(tie: bool) -> Vec<(String, Vec<usize>)> {
    let mut v = vec![];
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        for (n, s) in [("self_attn.q_proj", vec![16, 16]), ("self_attn.k_proj", vec![8, 16]), ("self_attn.v_proj", vec![8, 16]),
            ("self_attn.o_proj", vec![16, 16]), ("mlp.gate_proj", vec![32, 16]), ("mlp.up_proj", vec![32, 16]), ("mlp.down_proj", vec![16, 32])] {
            v.push((format!("{p}.{n}.weight"), s));
        }
        v.push((format!("{p}.input_layernorm.weight"), vec![16]));
        v.push((format!("{p}.post_attention_layernorm.weight"), vec![16]));
    }
    v.push(("model.embed_tokens.weight".into(), vec![64, 16]));
    v.push(("model.norm.weight".into(), vec![16]));
    if !tie {
        v.push(("lm_head.weight".into(), vec![64, 16]));
    }
    v
}

/// Writes a safetensors file; `dtype` per tensor defaults to BF16. `truncate` drops trailing bytes.
fn write_shard(path: &PathBuf, ts: &[(String, Vec<usize>, &str)], truncate: usize) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = vec![];
    for (name, shape, dtype) in ts {
        let elem = if *dtype == "F32" { 4 } else { 2 };
        let bytes = shape.iter().product::<usize>() * elem;
        let b = data.len();
        data.extend((0..bytes).map(|i| (i % 251) as u8));
        header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [b, b + bytes]}));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out.truncate(out.len() - truncate);
    std::fs::write(path, out).unwrap();
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sparkforge-weights-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn single(name: &str, ts: Vec<(String, Vec<usize>, &str)>) -> PathBuf {
    let d = dir(name);
    write_shard(&d.join("model.safetensors"), &ts, 0);
    d
}

fn bf16(ts: Vec<(String, Vec<usize>)>) -> Vec<(String, Vec<usize>, &'static str)> {
    ts.into_iter().map(|(n, s)| (n, s, "BF16")).collect()
}

#[test]
fn single_file_checkpoint_binds_every_linear_and_reports_extras() {
    let mut ts = bf16(tensors(false));
    ts.push(("model.layers.0.self_attn.rotary_emb.inv_freq".into(), vec![4], "F32"));
    let d = single("single", ts);
    let ck = Checkpoint::open(&d).unwrap();
    let b = bind(&ck, &spec(false)).unwrap();
    assert_eq!(b.linears.len(), 2 * 4 + 1);
    let qkv = b.linears.iter().find(|l| l.key() == "layers.1.qkv_proj").unwrap();
    assert_eq!((qkv.n, qkv.k, qkv.parts.len()), (32, 16, 3));
    assert!(qkv.parts[0].ends_with("q_proj.weight") && qkv.parts[2].ends_with("v_proj.weight"));
    let gu = b.linears.iter().find(|l| l.key() == "layers.0.gate_up_proj").unwrap();
    assert!(gu.parts[0].ends_with("gate_proj.weight") && gu.parts[1].ends_with("up_proj.weight"));
    assert_eq!(b.unexpected, vec!["model.layers.0.self_attn.rotary_emb.inv_freq".to_string()]);
}

#[test]
fn sharded_checkpoint_resolves_through_the_index() {
    let d = dir("sharded");
    let ts = bf16(tensors(false));
    let (a, b) = ts.split_at(ts.len() / 2);
    write_shard(&d.join("model-00001-of-00002.safetensors"), a, 0);
    write_shard(&d.join("model-00002-of-00002.safetensors"), b, 0);
    let mut map = serde_json::Map::new();
    for (n, _, _) in a { map.insert(n.clone(), "model-00001-of-00002.safetensors".into()); }
    for (n, _, _) in b { map.insert(n.clone(), "model-00002-of-00002.safetensors".into()); }
    std::fs::write(d.join("model.safetensors.index.json"), serde_json::json!({"metadata": {}, "weight_map": map}).to_string()).unwrap();
    let ck = Checkpoint::open(&d).unwrap();
    assert_eq!(ck.shards.len(), 2);
    bind(&ck, &spec(false)).unwrap();
}

#[test]
fn truncated_shard_is_rejected_naming_the_tensor() {
    let d = dir("trunc");
    write_shard(&d.join("model.safetensors"), &bf16(tensors(false)), 10);
    let err = Checkpoint::open(&d).err().expect("must fail").to_string();
    assert!(err.contains("outside the file"), "{err}");
}

#[test]
fn wrong_shape_and_dtype_fail_with_names_and_values() {
    let mut ts = bf16(tensors(false));
    for t in ts.iter_mut() {
        if t.0 == "model.layers.0.self_attn.o_proj.weight" { t.1 = vec![32, 16]; }
        if t.0 == "model.norm.weight" { t.2 = "F32"; }
    }
    let d = single("shape", ts);
    let err = bind(&Checkpoint::open(&d).unwrap(), &spec(false)).unwrap_err().to_string();
    assert!(err.contains("model.layers.0.self_attn.o_proj.weight: expected shape [16, 16], found [32, 16]"), "{err}");
    assert!(err.contains("model.norm.weight: expected dtype BF16, found F32"), "{err}");
}

#[test]
fn missing_tensor_is_named() {
    let ts: Vec<_> = bf16(tensors(false)).into_iter().filter(|t| t.0 != "model.layers.1.mlp.down_proj.weight").collect();
    let d = single("missing", ts);
    let err = bind(&Checkpoint::open(&d).unwrap(), &spec(false)).unwrap_err().to_string();
    assert!(err.contains("missing model.layers.1.mlp.down_proj.weight (expected BF16 [16, 32])"), "{err}");
}

#[test]
fn tied_embeddings_bind_lm_head_to_the_embedding() {
    let d = single("tied", bf16(tensors(true)));
    let b = bind(&Checkpoint::open(&d).unwrap(), &spec(true)).unwrap();
    let head = b.linears.iter().find(|l| l.linear == "lm_head").unwrap();
    assert_eq!(head.parts, vec!["model.embed_tokens.weight".to_string()]);
}

#[test]
fn qwen3_requires_qk_norm_tensors() {
    let qwen = CFG.replace("LlamaForCausalLM", "Qwen3ForCausalLM").replace("TIE", "false");
    let m = ModelSpec::from_json(&qwen, None).unwrap();
    let d = single("qwen3", bf16(tensors(false)));
    let err = bind(&Checkpoint::open(&d).unwrap(), &m).unwrap_err().to_string();
    assert!(err.contains("q_norm.weight") && err.contains("k_norm.weight"), "{err}");
}

#[test]
fn moe_and_quantized_checkpoints_are_refused() {
    let d = single("refuse", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let moe = ModelSpec::from_json(&std::fs::read_to_string("configs/qwen3-30b-a3b.json").unwrap(), None).unwrap();
    assert!(bind(&ck, &moe).unwrap_err().to_string().contains("MoE weight loading is not supported yet"));
    let fp4 = ModelSpec::from_json(&CFG.replace("TIE", "false"), Some(WeightFormat::Nvfp4)).unwrap();
    assert!(bind(&ck, &fp4).unwrap_err().to_string().contains("not supported yet"));
}

#[test]
fn memory_guard_keeps_four_gib_free() {
    let gib = 1u64 << 30;
    assert!(check_headroom(10 * gib, 14 * gib).is_ok());
    let err = check_headroom(10 * gib + 1, 14 * gib).unwrap_err().to_string();
    assert!(err.contains("MemAvailable is 14.00 GiB"), "{err}");
    assert_eq!(HEADROOM_BYTES, 4 * gib);
}

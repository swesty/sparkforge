use sparkforge::config::{ModelSpec, WeightFormat};
use sparkforge::weights::{bind, check_headroom, Checkpoint, HEADROOM_BYTES};
use std::path::PathBuf;

mod common;
use common::{dir, fresh_private_dir, write_hybrid};

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
        let elem = match *dtype { "F32" => 4, "F8_E4M3" | "U8" => 1, _ => 2 };
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

#[test]
fn fixture_dirs_refuse_planted_directories_and_symlinks() {
    let root = dir("planted");
    let victim = root.join("victim.txt");
    std::fs::write(&victim, b"keep").unwrap();
    // A pre-created directory at the fixture path is not reused.
    let planted = root.join("planted-dir");
    std::fs::create_dir(&planted).unwrap();
    assert_eq!(fresh_private_dir(&planted).unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
    // Nor is a symlink, even one pointing at a writable file.
    #[cfg(unix)]
    {
        let link = root.join("planted-link");
        std::os::unix::fs::symlink(&victim, &link).unwrap();
        assert_eq!(fresh_private_dir(&link).unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(dir("mode")).unwrap().permissions().mode() & 0o777, 0o700);
    }
    assert_eq!(std::fs::read(&victim).unwrap(), b"keep");
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

/// Writes a safetensors file from an explicit header (for malformed-metadata cases).
fn write_raw(path: &PathBuf, header: serde_json::Value, data_len: usize, header_len_override: Option<u64>) {
    let mut h = serde_json::to_vec(&header).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = header_len_override.unwrap_or(h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend((0..data_len).map(|i| (i % 251) as u8));
    std::fs::write(path, out).unwrap();
}

fn sharded(name: &str) -> (PathBuf, Vec<(String, Vec<usize>, &'static str)>, Vec<(String, Vec<usize>, &'static str)>) {
    let d = dir(name);
    let ts = bf16(tensors(false));
    let (a, b) = ts.split_at(ts.len() / 2);
    write_shard(&d.join("model-00001-of-00002.safetensors"), a, 0);
    write_shard(&d.join("model-00002-of-00002.safetensors"), b, 0);
    (d, a.to_vec(), b.to_vec())
}

fn write_index(d: &PathBuf, map: serde_json::Map<String, serde_json::Value>) {
    std::fs::write(d.join("model.safetensors.index.json"), serde_json::json!({"weight_map": map}).to_string()).unwrap();
}

#[test]
fn index_entries_must_name_the_shard_that_holds_the_tensor() {
    let (d, a, b) = sharded("swap");
    let mut map = serde_json::Map::new();
    for (n, _, _) in &a { map.insert(n.clone(), "model-00001-of-00002.safetensors".into()); }
    for (n, _, _) in &b { map.insert(n.clone(), "model-00002-of-00002.safetensors".into()); }
    // Swap one assignment from each shard.
    map.insert(a[0].0.clone(), "model-00002-of-00002.safetensors".into());
    map.insert(b[0].0.clone(), "model-00001-of-00002.safetensors".into());
    write_index(&d, map);
    let err = Checkpoint::open(&d).err().expect("must fail").to_string();
    assert!(err.contains("index lists") && (err.contains(&a[0].0) || err.contains(&b[0].0)), "{err}");
}

#[test]
fn index_entry_without_a_tensor_is_rejected() {
    let (d, a, b) = sharded("ghost");
    let mut map = serde_json::Map::new();
    for (n, _, _) in a.iter().chain(&b) {
        let f = if a.iter().any(|t| &t.0 == n) { "model-00001-of-00002.safetensors" } else { "model-00002-of-00002.safetensors" };
        map.insert(n.clone(), f.into());
    }
    map.insert("model.layers.9.mlp.down_proj.weight".into(), "model-00001-of-00002.safetensors".into());
    write_index(&d, map);
    let err = Checkpoint::open(&d).err().expect("must fail").to_string();
    assert!(err.contains("model.layers.9.mlp.down_proj.weight") && err.contains("no such tensor"), "{err}");
}

#[test]
fn wrapping_offsets_overflowing_header_and_shape_are_errors_not_header_reads() {
    let d = dir("wrap");
    let p = d.join("model.safetensors");
    // Offsets that would wrap into the header if added to the data start unchecked.
    write_raw(&p, serde_json::json!({"model.norm.weight": {"dtype": "BF16", "shape": [16],
        "data_offsets": [18446744073709551552u64, 18446744073709551584u64]}}), 32, None);
    let err = Checkpoint::open(&d).err().expect("wrapped offsets must fail").to_string();
    assert!(err.contains("outside the file"), "{err}");
    // A header length that would overflow 8 + N.
    write_raw(&p, serde_json::json!({}), 0, Some(u64::MAX));
    assert!(Checkpoint::open(&d).is_err());
    // A shape whose byte count overflows u64.
    write_raw(&p, serde_json::json!({"x": {"dtype": "BF16", "shape": [4294967296u64, 4294967296u64], "data_offsets": [0, 0]}}), 0, None);
    let err = Checkpoint::open(&d).err().expect("overflowing shape must fail").to_string();
    assert!(err.contains("overflows"), "{err}");
}

#[test]
fn overlapping_or_gapped_tensor_ranges_are_rejected() {
    let d = dir("overlap");
    let p = d.join("model.safetensors");
    // up_proj points at gate_proj's bytes; up_proj's own bytes are left unaccounted.
    write_raw(&p, serde_json::json!({
        "g": {"dtype": "BF16", "shape": [32, 16], "data_offsets": [0, 1024]},
        "u": {"dtype": "BF16", "shape": [32, 16], "data_offsets": [0, 1024]}}), 2048, None);
    let err = Checkpoint::open(&d).err().expect("overlap must fail").to_string();
    assert!(err.contains("overlaps"), "{err}");
    // A hole between two tensors.
    write_raw(&p, serde_json::json!({
        "g": {"dtype": "BF16", "shape": [32, 16], "data_offsets": [0, 1024]},
        "u": {"dtype": "BF16", "shape": [32, 16], "data_offsets": [1040, 2064]}}), 2064, None);
    let err = Checkpoint::open(&d).err().expect("gap must fail").to_string();
    assert!(err.contains("unaccounted"), "{err}");
    // Empty tensors take no space and are fine.
    write_raw(&p, serde_json::json!({
        "g": {"dtype": "BF16", "shape": [32, 16], "data_offsets": [0, 1024]},
        "e": {"dtype": "BF16", "shape": [0], "data_offsets": [1024, 1024]}}), 1024, None);
    assert!(Checkpoint::open(&d).is_ok());
}

#[test]
fn truncating_a_shard_after_open_is_an_error_not_a_crash() {
    let d = single("truncate-after", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    std::fs::OpenOptions::new().write(true).open(d.join("model.safetensors")).unwrap().set_len(0).unwrap();
    let t = &ck.tensors["model.layers.0.self_attn.o_proj.weight"];
    let mut buf = vec![];
    let err = ck.read_into(t, &mut buf).unwrap_err().to_string();
    assert!(err.contains("changed size"), "{err}");
}

#[test]
fn streaming_reference_matches_a_naive_gemv_on_fused_parts() {
    let d = single("reference", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let b = bind(&ck, &spec(false)).unwrap();
    let qkv = b.linears.iter().find(|l| l.key() == "layers.0.qkv_proj").unwrap();
    let x: Vec<f64> = (0..qkv.k).map(|i| (i as f64 - 7.5) / 8.0).collect();
    let got = sparkforge::weights::reference_gemv(&ck, qkv, &x).unwrap();
    let mut want = vec![];
    let mut buf = vec![];
    for name in &qkv.parts {
        ck.read_into(&ck.tensors[name], &mut buf).unwrap();
        for row in buf.chunks_exact(qkv.k * 2) {
            want.push(row.chunks_exact(2).zip(&x).map(|(c, xv)| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64 * xv).sum::<f64>());
        }
    }
    assert_eq!(got.len(), qkv.n);
    assert_eq!(got, want);
}

#[test]
fn memory_guard_budgets_staging_and_verification_scratch() {
    use sparkforge::weights::{load_overhead_bytes, REF_CHUNK_BYTES, REF_THREADS, STAGING_BYTES};
    let d = single("overhead", bf16(tensors(false)));
    let b = bind(&Checkpoint::open(&d).unwrap(), &spec(false)).unwrap();
    assert_eq!(load_overhead_bytes(&b, false), STAGING_BYTES);
    assert_eq!(load_overhead_bytes(&b, true), STAGING_BYTES + (REF_THREADS * REF_CHUNK_BYTES) as u64 + 64 * 8);
}

#[test]
fn concurrent_loads_cannot_both_pass_the_memory_guard() {
    use sparkforge::weights::{admit, ADMISSION_WAIT};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    const GIB: u64 = 1 << 30;
    let lock = dir("admission").join("admission.lock");
    // 20 GiB available; each load needs 14 GiB, so either fits alone but not both.
    let avail = Arc::new(AtomicU64::new(20 * GIB));
    let first = admit(&user_lock(&lock), 14 * GIB, 0, || Some(avail.load(Ordering::SeqCst)), ADMISSION_WAIT).expect("first load fits");
    let second = {
        let (lock, avail) = (lock.clone(), avail.clone());
        std::thread::spawn(move || admit(&user_lock(&lock), 14 * GIB, 0, || Some(avail.load(Ordering::SeqCst)), ADMISSION_WAIT).map(|_| ()))
    };
    // The second load is blocked on the lock while the first allocates its 14 GiB.
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(!second.is_finished(), "second admission must wait for the first load's allocations");
    avail.fetch_sub(14 * GIB, Ordering::SeqCst);
    drop(first);
    let err = second.join().unwrap().expect_err("combined loads exceed memory").to_string();
    assert!(err.contains("refusing to copy"), "{err}");
}

#[test]
fn same_length_rewrite_between_reads_is_an_error_not_a_mixed_load() {
    let d = single("rewrite", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let t = &ck.tensors["model.layers.1.mlp.down_proj.weight"];
    let mut first = vec![0u8; 64];
    ck.read_range(t, 0, &mut first).unwrap();
    // Coarse filesystem clocks: make sure the write lands in a later timestamp tick than the open.
    std::thread::sleep(std::time::Duration::from_millis(50));
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(d.join("model.safetensors")).unwrap();
        f.seek(SeekFrom::Start(t.start + 64)).unwrap();
        f.write_all(&[0xAB; 64]).unwrap();
    }
    assert_eq!(std::fs::metadata(d.join("model.safetensors")).unwrap().len(), ck.shards[0].len, "rewrite must keep the length");
    let mut second = vec![0u8; 64];
    let err = ck.read_range(t, 64, &mut second).unwrap_err().to_string();
    assert!(err.contains("modified since it was opened"), "{err}");
}

#[test]
fn duplicate_json_keys_are_rejected_in_the_index_and_in_headers() {
    // Index: an earlier entry for model.norm.weight points at a shard that does not exist. Literal
    // JSON, because building it through a JSON map would drop the duplicate before writing.
    let (d, a, b) = sharded("dup-index");
    let mut entries = vec![r#""model.norm.weight": "missing.safetensors""#.to_string()];
    for (n, _, _) in &a { entries.push(format!("{n:?}: \"model-00001-of-00002.safetensors\"")); }
    for (n, _, _) in &b { entries.push(format!("{n:?}: \"model-00002-of-00002.safetensors\"")); }
    std::fs::write(d.join("model.safetensors.index.json"), format!(r#"{{"weight_map": {{{}}}}}"#, entries.join(", "))).unwrap();
    let err = format!("{:#}", Checkpoint::open(&d).err().expect("duplicate index key must fail"));
    assert!(err.contains("duplicate key \"model.norm.weight\""), "{err}");

    // Header: the same tensor name twice.
    let d = dir("dup-header");
    let h = br#"{"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}, "w": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]}}"#;
    let mut header = h.to_vec();
    while header.len() % 8 != 0 { header.push(b' '); }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(&header);
    out.extend([0u8; 8]);
    std::fs::write(d.join("model.safetensors"), out).unwrap();
    let err = format!("{:#}", Checkpoint::open(&d).err().expect("duplicate header key must fail"));
    assert!(err.contains("duplicate tensor name \"w\""), "{err}");
}

#[test]
fn index_shard_names_cannot_leave_the_checkpoint_directory() {
    let root = dir("escape");
    let prev = root.join("previous");
    std::fs::create_dir(&prev).unwrap();
    write_shard(&prev.join("model.safetensors"), &bf16(tensors(false)), 0);
    let sel = root.join("selected");
    std::fs::create_dir(&sel).unwrap();
    for target in ["../previous/model.safetensors".to_string(), prev.join("model.safetensors").display().to_string()] {
        let mut map = serde_json::Map::new();
        for (n, _) in tensors(false) { map.insert(n, target.clone().into()); }
        write_index(&sel, map);
        let err = format!("{:#}", Checkpoint::open(&sel).err().expect("escaping index must fail"));
        assert!(err.contains("not a plain relative path"), "{target}: {err}");
    }
    // A Hugging Face cache snapshot links each shard to ../../blobs/<hash>; that still opens.
    #[cfg(unix)]
    {
        let blobs = root.join("blobs");
        std::fs::create_dir(&blobs).unwrap();
        let snap = root.join("snapshots").join("abc123");
        std::fs::create_dir_all(&snap).unwrap();
        write_shard(&blobs.join(hf_blob()), &bf16(tensors(false)), 0);
        std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), snap.join("model.safetensors")).unwrap();
        let ck = Checkpoint::open(&snap).expect("symlinked HF cache snapshot opens");
        bind(&ck, &spec(false)).unwrap();
    }
}

#[cfg(unix)]
#[test]
fn a_fifo_shard_fails_promptly_instead_of_hanging() {
    let d = dir("fifo");
    let path = d.join("model.safetensors");
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0, "mkfifo");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(Checkpoint::open(&d).err().map(|e| format!("{e:#}")));
    });
    let err = rx.recv_timeout(std::time::Duration::from_secs(5)).expect("opening a FIFO shard must not block").expect("must fail");
    assert!(err.contains("not a regular file"), "{err}");
}

#[test]
fn a_format_override_does_not_let_a_mixed_precision_checkpoint_load() {
    // SHA-783: --format lets `plan` run on a MIXED_PRECISION config, but the checkpoint itself is
    // still mixed precision, so binding refuses it.
    let cfg = CFG.replace("TIE", "false").replace("\"vocab_size\": 64,",
        "\"vocab_size\": 64, \"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\", \"quantized_layers\": {\"model.layers.0.self_attn.o_proj\": {\"quant_algo\": \"MXFP8\", \"group_size\": 32}}},");
    assert!(ModelSpec::from_json(&cfg, None).is_err(), "dense mixed precision needs --format");
    let s = ModelSpec::from_json(&cfg, Some(WeightFormat::Bf16)).unwrap();
    assert!(s.declared_mixed && s.formats.is_empty());
    let d = single("mixed-override", bf16(tensors(false)));
    let err = bind(&Checkpoint::open(&d).unwrap(), &s).err().expect("binding must refuse").to_string();
    assert!(err.contains("mixed-precision"), "{err}");
}

#[test]
fn a_quantized_checkpoint_behind_a_bf16_config_is_refused_before_gpu_work() {
    // The config omits quantization_config, but the checkpoint holds an FP8 projection and its scale.
    let mut ts = bf16(tensors(false));
    for t in ts.iter_mut() {
        if t.0 == "model.layers.0.self_attn.o_proj.weight" { t.2 = "F8_E4M3"; }
    }
    ts.push(("model.layers.0.self_attn.o_proj.weight_scale".into(), vec![16, 1], "U8"));
    let d = single("undeclared-fp8", ts);
    let err = bind(&Checkpoint::open(&d).unwrap(), &spec(false)).err().expect("binding must refuse").to_string();
    assert!(err.contains("model.layers.0.self_attn.o_proj.weight: expected dtype BF16, found F8_E4M3"), "{err}");
}

/// The Hugging Face cache blob name of the standard untied fixture shard: the SHA256 of its bytes,
/// as the cache names LFS files.
fn hf_blob() -> &'static str {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        use sha2::Digest;
        let d = dir("hf-blob-name");
        write_shard(&d.join("w"), &bf16(tensors(false)), 0);
        sha2::Sha256::digest(std::fs::read(d.join("w")).unwrap()).iter().map(|b| format!("{b:02x}")).collect()
    })
}

// ---- SHA-765: loader hardening ----

#[cfg(unix)]
fn my_uid() -> u32 { unsafe { libc::geteuid() } }

/// The admission lock at `p`, as `admit` takes it.
fn user_lock(p: &std::path::Path) -> PathBuf {
    p.to_path_buf()
}

#[cfg(unix)]
fn chmod(p: &std::path::Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)).unwrap();
}

#[cfg(unix)]
#[test]
fn a_holder_cannot_stall_admission_forever() {
    use sparkforge::weights::admit;
    let lock = dir("wait-limit").join("admission.lock");
    let _held = admit(&user_lock(&lock), 0, 0, || Some(64 << 30), std::time::Duration::from_secs(1)).unwrap();
    let t0 = std::time::Instant::now();
    let err = format!("{:#}", admit(&user_lock(&lock), 0, 0, || Some(64 << 30), std::time::Duration::from_millis(200)).err().expect("must time out"));
    assert!(err.contains("gave up after"), "{err}");
    assert!(t0.elapsed() < std::time::Duration::from_secs(5));
}

#[cfg(unix)]
#[test]
fn the_admission_lock_comes_from_ownership_not_the_environment() {
    use sparkforge::weights::admission_lock_in;
    let root = dir("user-lock");
    let home = root.join("home");
    std::fs::create_dir(&home).unwrap();
    // The function reads no environment: the home directory comes from the caller (getpwuid_r in production).
    let p = admission_lock_in(my_uid(), || Some(home.clone())).unwrap();
    assert_eq!(p.parent().unwrap(), home.join(".local/state/sparkforge"));
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(std::fs::metadata(p.parent().unwrap()).unwrap().permissions().mode() & 0o777, 0o700);
    // The old shared fallback (<tmp>/sparkforge-<uid>) is never used, so a pre-claimed entry there cannot block loads.
    assert!(!p.starts_with(std::env::temp_dir().join(format!("sparkforge-{}", my_uid()))));
    // No password entry: nowhere to coordinate, so refuse.
    let err = format!("{:#}", admission_lock_in(my_uid(), || None).unwrap_err());
    assert!(err.contains("password database"), "{err}");
    // A lock directory others can write is refused.
    chmod(p.parent().unwrap(), 0o777);
    let err = format!("{:#}", admission_lock_in(my_uid(), || Some(home.clone())).unwrap_err());
    assert!(err.contains("not writable by others"), "{err}");
}

#[cfg(unix)]
#[test]
fn a_login_does_not_split_one_users_admission() {
    // codex-review P2: the user lock used to move to /run/user/<uid> once a login created it. It
    // depends only on the home directory, so a load started before the login still blocks one
    // started after it.
    use sparkforge::weights::{admission_lock_in, admit};
    let root = dir("login-split");
    let home = root.join("home");
    std::fs::create_dir(&home).unwrap();
    let before = admission_lock_in(my_uid(), || Some(home.clone())).unwrap();
    let _first = admit(&before, 0, 0, || Some(64 << 30), std::time::Duration::from_secs(1)).unwrap();
    std::fs::create_dir(root.join("run-user")).unwrap(); // the login's runtime directory appears
    let after = admission_lock_in(my_uid(), || Some(home.clone())).unwrap();
    assert_eq!(before, after);
    let err = format!("{:#}", admit(&after, 0, 0, || Some(64 << 30), std::time::Duration::from_millis(200)).err().expect("must wait"));
    assert!(err.contains("gave up after"), "{err}");
}

#[cfg(unix)]
#[test]
fn a_fifo_admission_lock_fails_promptly() {
    use sparkforge::weights::admit;
    let lock = dir("fifo-lock").join("sparkforge-gpu-admission.lock");
    let c = std::ffi::CString::new(lock.as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = admit(&user_lock(&lock), 0, 0, || Some(64 << 30), std::time::Duration::from_secs(1));
        let _ = tx.send(r.err().map(|e| format!("{e:#}")));
    });
    let err = rx.recv_timeout(std::time::Duration::from_secs(5)).expect("a FIFO lock must not block").expect("must fail");
    assert!(err.contains("not a regular file"), "{err}");
}

#[test]
fn an_admission_only_covers_the_workload_it_checked() {
    // codex-review P2: upload confirms its admission against the bytes it actually copies.
    use sparkforge::weights::admit;
    const GIB: u64 = 1 << 30;
    let lock = dir("admission-scope").join("admission.lock");
    let small = admit(&user_lock(&lock), GIB, 0, || Some(64 * GIB), std::time::Duration::from_secs(1)).unwrap();
    assert!(small.confirm(GIB, 0, || Some(64 * GIB)).is_ok());
    let err = small.confirm(2 * GIB, 0, || Some(64 * GIB)).unwrap_err().to_string();
    assert!(err.contains("admission covers only"), "{err}");
}

#[test]
fn memory_used_after_admission_is_rechecked_before_allocating() {
    // codex-review P2: CUDA startup between admission and upload can eat the margin.
    use sparkforge::weights::admit;
    const GIB: u64 = 1 << 30;
    let lock = dir("admission-recheck").join("admission.lock");
    let adm = admit(&user_lock(&lock), 14 * GIB, 0, || Some(20 * GIB), std::time::Duration::from_secs(1)).expect("fits at admission");
    let err = adm.confirm(14 * GIB, 0, || Some(17 * GIB)).unwrap_err().to_string();
    assert!(err.contains("refusing to copy"), "{err}");
}

#[test]
fn the_reported_baseline_is_the_memory_admission_checked_before_allocating() {
    // codex-review fp:69a474a7ea8f on #19: `load-weights` read its "before" figure ahead of
    // admission, so memory another process freed while this load waited showed up as memory the
    // load used. `confirm` returns the reading it checked, under the lock right before allocating.
    use sparkforge::weights::admit;
    use std::cell::Cell;
    const GIB: u64 = 1 << 30;
    let lock = dir("admission-baseline").join("admission.lock");
    let avail = Cell::new(20 * GIB);
    let adm = admit(&user_lock(&lock), 14 * GIB, 0, || Some(avail.get()), std::time::Duration::from_secs(1)).expect("fits at admission");
    // Another process frees 4 GiB between admission and the allocation.
    avail.set(24 * GIB);
    let baseline = adm.confirm(14 * GIB, 0, || Some(avail.get())).unwrap();
    assert_eq!(baseline, 24 * GIB);
    // fp:08579c39429a on #22: a change after the final check doesn't move the saved baseline. It
    // shows only in the reading taken after the copy.
    avail.set(28 * GIB);
    assert_eq!(baseline, 24 * GIB, "the baseline is the checked reading, not a later one");
    assert_eq!(avail.get() - baseline, 4 * GIB, "a later reading sees the change");
}

#[test]
fn a_metadata_file_that_grows_past_the_cap_is_refused_not_truncated() {
    // codex-review P2: the first `cap` bytes parse as JSON; the byte after them must still fail the read.
    use sparkforge::weights::read_capped;
    let cap = 64u64;
    let mut doc = b"{}".to_vec();
    doc.resize(cap as usize, b' ');
    assert!(read_capped(std::io::Cursor::new(doc.clone()), cap, "config.json").is_ok());
    doc.push(b'x');
    let err = read_capped(std::io::Cursor::new(doc), cap, "config.json").unwrap_err().to_string();
    assert!(err.contains("config.json") && err.contains("grew past"), "{err}");
}

#[test]
fn admission_is_decided_on_the_host_before_any_cuda_call() {
    use sparkforge::weights::admit_load_with;
    let d = single("admit-host", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let b = bind(&ck, &spec(false)).unwrap();
    let lock = dir("admit-host-lock").join("admission.lock");
    // A machine with 1 GiB available refuses before anything touches the GPU.
    let err = admit_load_with(&ck, &b, true, &user_lock(&lock), || Some(1 << 30)).err().expect("over budget").to_string();
    assert!(err.contains("refusing to copy"), "{err}");
    assert!(admit_load_with(&ck, &b, true, &user_lock(&lock), || Some(64 << 30)).is_ok());
}

#[cfg(unix)]
#[test]
fn a_checkpoint_is_read_from_one_generation_even_if_its_symlink_moves() {
    let root = dir("generations");
    let (a, b) = (root.join("gen-a"), root.join("gen-b"));
    for g in [&a, &b] {
        std::fs::create_dir(g).unwrap();
        write_shard(&g.join("model.safetensors"), &bf16(tensors(false)), 0);
    }
    // Make generation B's bytes differ.
    let mut bytes = std::fs::read(b.join("model.safetensors")).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 0xff;
    std::fs::write(b.join("model.safetensors"), bytes).unwrap();
    let current = root.join("current");
    std::os::unix::fs::symlink(&a, &current).unwrap();
    let ck = Checkpoint::open(&current).unwrap();
    assert_eq!(ck.dir, std::fs::canonicalize(&a).unwrap(), "the directory is resolved once, to generation A");
    // Switch the symlink to generation B after opening: reads still come from A.
    std::fs::remove_file(&current).unwrap();
    std::os::unix::fs::symlink(&b, &current).unwrap();
    let t = ck.tensors.values().max_by_key(|t| t.end).unwrap();
    let mut got = vec![0u8; 1];
    ck.read_range(t, t.len() - 1, &mut got).unwrap();
    let want = std::fs::read(a.join("model.safetensors")).unwrap();
    assert_eq!(got[0], want[want.len() - 1]);
    assert_eq!(sparkforge::weights::resolve_checkpoint_dir(&current).unwrap().path, std::fs::canonicalize(&b).unwrap());
}

#[cfg(unix)]
#[test]
fn replacing_the_canonical_directory_after_the_config_read_cannot_mix_generations() {
    // codex-review P2: the config and the shards come from the directory that was opened, even if
    // its canonical path is replaced by another generation between the two.
    use sparkforge::weights::{read_config, resolve_checkpoint_dir};
    let root = dir("dir-swap");
    let (a, b) = (root.join("current"), root.join("gen-b"));
    for (g, tag) in [(&a, "A"), (&b, "B")] {
        std::fs::create_dir(g).unwrap();
        std::fs::write(g.join("config.json"), format!("{{\"gen\": \"{tag}\"}}")).unwrap();
        write_shard(&g.join("model.safetensors"), &bf16(tensors(false)), 0);
    }
    let mut bytes = std::fs::read(b.join("model.safetensors")).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 0xff;
    std::fs::write(b.join("model.safetensors"), bytes).unwrap();
    let opened = resolve_checkpoint_dir(&a).unwrap();
    assert!(read_config(&opened).unwrap().contains("\"A\""));
    // Move generation A aside and put B at the canonical path.
    std::fs::rename(&a, root.join("gen-a-old")).unwrap();
    std::os::unix::fs::symlink(&b, &a).unwrap();
    let ck = Checkpoint::open_dir(&opened).unwrap();
    let t = ck.tensors.values().max_by_key(|t| t.end).unwrap();
    let mut got = vec![0u8; 1];
    ck.read_range(t, t.len() - 1, &mut got).unwrap();
    let want = std::fs::read(root.join("gen-a-old/model.safetensors")).unwrap();
    assert_eq!(got[0], want[want.len() - 1], "shards come from generation A, like the config");
}

#[cfg(unix)]
#[test]
fn a_fifo_config_or_index_fails_promptly() {
    for name in ["config.json", "model.safetensors.index.json"] {
        let d = dir("fifo-meta");
        write_shard(&d.join("model.safetensors"), &bf16(tensors(false)), 0);
        let c = std::ffi::CString::new(d.join(name).as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let d2 = d.clone();
        let is_config = name == "config.json";
        std::thread::spawn(move || {
            let r = if is_config {
                sparkforge::weights::resolve_checkpoint_dir(&d2).and_then(|cd| sparkforge::weights::read_config(&cd)).map(|_| ())
            } else {
                Checkpoint::open(&d2).map(|_| ())
            };
            let _ = tx.send(r.err().map(|e| format!("{e:#}")));
        });
        let err = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap_or_else(|_| panic!("{name}: must not block")).expect("must fail");
        assert!(err.contains("not a regular file"), "{name}: {err}");
    }
}

#[cfg(unix)]
#[test]
fn a_dangling_index_symlink_is_an_error_not_a_fallback() {
    let d = single("dangling-index", bf16(tensors(false)));
    std::os::unix::fs::symlink(d.join("missing-index.json"), d.join("model.safetensors.index.json")).unwrap();
    let err = format!("{:#}", Checkpoint::open(&d).err().expect("must fail"));
    assert!(err.contains("model.safetensors.index.json"), "{err}");
}

#[test]
fn an_untied_model_without_lm_head_is_refused() {
    let ts: Vec<_> = bf16(tensors(false)).into_iter().filter(|t| t.0 != "lm_head.weight").collect();
    let d = single("untied-no-head", ts);
    let err = bind(&Checkpoint::open(&d).unwrap(), &spec(false)).err().expect("must fail").to_string();
    assert!(err.contains("lm_head.weight"), "{err}");
    // With tying on, the same checkpoint binds lm_head to the embedding.
    assert!(bind(&Checkpoint::open(&d).unwrap(), &spec(true)).is_ok());
}

// ---- SHA-765: codex-review cycle 2 follow-ups ----

#[cfg(unix)]
#[test]
fn a_shard_link_through_a_switchable_alias_is_refused() {
    // In generation A, the shard links through `current`, an alias a publisher can switch to B
    // between the config read and the shard open. The link is refused, not followed.
    let g = std::fs::canonicalize(dir("alias-link")).unwrap();
    let data = g.join("data-a");
    std::fs::create_dir(&data).unwrap();
    write_shard(&data.join("weights.safetensors"), &bf16(tensors(false)), 0);
    std::os::unix::fs::symlink("data-a", g.join("current")).unwrap();
    std::os::unix::fs::symlink("current/weights.safetensors", g.join("model.safetensors")).unwrap();
    let err = format!("{:#}", Checkpoint::open(&g).err().expect("an alias link must fail"));
    assert!(err.contains("passes through another symlink"), "{err}");
    // A link straight to the file inside the checkpoint directory still opens.
    std::fs::remove_file(g.join("model.safetensors")).unwrap();
    std::os::unix::fs::symlink("data-a/weights.safetensors", g.join("model.safetensors")).unwrap();
    bind(&Checkpoint::open(&g).unwrap(), &spec(false)).unwrap();
}

#[cfg(unix)]
#[test]
fn the_user_lock_opens_read_write() {
    // An exclusive lock over an NFS home needs a writable descriptor, so the user lock is opened
    // read-write: a lock file this user cannot write is an error, not a read-only fallback.
    if my_uid() == 0 {
        return; // root writes anything
    }
    use sparkforge::weights::admit;
    let lock = dir("rw-lock").join("sparkforge-gpu-admission.lock");
    std::fs::write(&lock, b"").unwrap();
    chmod(&lock, 0o400);
    let err = format!("{:#}", admit(&user_lock(&lock), 0, 0, || Some(64 << 30), std::time::Duration::from_secs(1)).err().expect("read-only lock must fail"));
    assert!(err.contains("cannot open the admission lock"), "{err}");
}

#[test]
fn a_load_only_admission_cannot_authorize_verification() {
    use sparkforge::weights::{admit_load_with, bytes_to_copy, load_overhead_bytes};
    let d = single("verify-scope", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let b = bind(&ck, &spec(false)).unwrap();
    let lock = dir("verify-scope-lock").join("admission.lock");
    let adm = admit_load_with(&ck, &b, false, &user_lock(&lock), || Some(64 << 30)).unwrap();
    assert!(adm.confirm(bytes_to_copy(&ck, &b), load_overhead_bytes(&b, false), || Some(64 << 30)).is_ok());
    let err = adm.confirm(bytes_to_copy(&ck, &b), load_overhead_bytes(&b, true), || Some(64 << 30)).unwrap_err().to_string();
    assert!(err.contains("admission covers only"), "{err}");
}

// ---- SHA-787: codex-review round-3 follow-ups on #13 ----

#[cfg(unix)]
#[test]
fn a_direct_link_outside_the_checkpoint_is_refused() {
    // The shard links straight to /published/model.safetensors. /published is a real directory, so
    // the no-symlink walk passes, yet a publisher can replace it with generation B between the
    // config read and the shard open. Such a link is refused (absolute or relative).
    use sparkforge::weights::{read_config, resolve_checkpoint_dir};
    let root = std::fs::canonicalize(dir("direct-link")).unwrap();
    let published = root.join("published");
    std::fs::create_dir(&published).unwrap();
    write_shard(&published.join("model.safetensors"), &bf16(tensors(false)), 0);
    for (name, target) in [("abs", published.join("model.safetensors")), ("rel", "../published/model.safetensors".into())] {
        let g = root.join(format!("gen-{name}"));
        std::fs::create_dir(&g).unwrap();
        std::fs::write(g.join("config.json"), "{}").unwrap();
        std::os::unix::fs::symlink(&target, g.join("model.safetensors")).unwrap();
        let opened = resolve_checkpoint_dir(&g).unwrap();
        read_config(&opened).unwrap();
        let err = format!("{:#}", Checkpoint::open_dir(&opened).err().expect("an outside link must fail"));
        assert!(err.contains("outside the checkpoint directory"), "{name}: {err}");
    }
    // Leaving the directory counts even when the link comes back in: the walk passes through
    // the parent, which can be replaced. Staying inside opens.
    let g = root.join("gen-inside");
    std::fs::create_dir_all(g.join("blobs")).unwrap();
    write_shard(&g.join("blobs/w.safetensors"), &bf16(tensors(false)), 0);
    std::os::unix::fs::symlink("../gen-inside/blobs/w.safetensors", g.join("model.safetensors")).unwrap();
    let err = format!("{:#}", Checkpoint::open(&g).err().expect("leaving the directory, even to come back, is outside"));
    assert!(err.contains("outside the checkpoint directory"), "{err}");
    std::fs::remove_file(g.join("model.safetensors")).unwrap();
    std::os::unix::fs::symlink("./blobs/../blobs/w.safetensors", g.join("model.safetensors")).unwrap();
    bind(&Checkpoint::open(&g).unwrap(), &spec(false)).unwrap();
}

#[cfg(unix)]
#[test]
fn only_a_hugging_face_snapshot_may_link_into_its_blobs() {
    let root = std::fs::canonicalize(dir("hf-blobs")).unwrap();
    let blobs = root.join("blobs");
    std::fs::create_dir(&blobs).unwrap();
    write_shard(&blobs.join(hf_blob()), &bf16(tensors(false)), 0);
    write_shard(&blobs.join("latest"), &bf16(tensors(false)), 0);
    // `<repo>/snapshots/<rev>/model.safetensors -> ../../blobs/<sha256>` opens: blobs are
    // content-addressed, so the name cannot come to mean another generation.
    let snap = root.join("snapshots/rev1");
    std::fs::create_dir_all(&snap).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), snap.join("model.safetensors")).unwrap();
    bind(&Checkpoint::open(&snap).unwrap(), &spec(false)).unwrap();
    // A blob name that is not a hash could be rewritten in place: refused.
    let snap2 = root.join("snapshots/rev2");
    std::fs::create_dir_all(&snap2).unwrap();
    std::os::unix::fs::symlink("../../blobs/latest", snap2.join("model.safetensors")).unwrap();
    let err = format!("{:#}", Checkpoint::open(&snap2).err().expect("non-hash blob"));
    assert!(err.contains("outside the checkpoint directory"), "{err}");
    // The same link from a directory that is not a cache snapshot: refused.
    let other = root.join("not-snapshots/rev1");
    std::fs::create_dir_all(&other).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), other.join("model.safetensors")).unwrap();
    let err = format!("{:#}", Checkpoint::open(&other).err().expect("not a snapshot"));
    assert!(err.contains("outside the checkpoint directory"), "{err}");
}

#[cfg(target_os = "linux")]
#[test]
fn a_checkpoint_in_a_search_only_directory_loads() {
    // Mode 0710/0711 directories let group members (or anyone) open a file they know the name of
    // without listing the directory. Opening the checkpoint must need only that search permission.
    if my_uid() == 0 {
        return; // root bypasses directory permissions
    }
    let d = dir("search-only");
    std::fs::create_dir(d.join("sub")).unwrap();
    write_shard(&d.join("sub/model.safetensors"), &bf16(tensors(false)), 0);
    let mut map = serde_json::Map::new();
    for (n, _) in tensors(false) { map.insert(n, "sub/model.safetensors".into()); }
    write_index(&d, map);
    chmod(&d.join("sub"), 0o100);
    chmod(&d, 0o100);
    let r = Checkpoint::open(&d).map(|ck| bind(&ck, &spec(false)).is_ok());
    chmod(&d, 0o700);
    chmod(&d.join("sub"), 0o700);
    assert!(r.as_ref().is_ok_and(|b| *b), "{:#}", r.err().unwrap());
}

// ---- SHA-787: codex-review cycle 1 follow-ups ----

#[cfg(unix)]
#[test]
fn a_replaced_hugging_face_blob_store_cannot_mix_generations() {
    // After the config is read from `<repo>/snapshots/<rev>`, `<repo>/blobs` is replaced by one
    // holding a different, shape-compatible shard under the same hash name. Reads still come from
    // the blob store that existed when the snapshot was opened.
    use sparkforge::weights::{read_config, resolve_checkpoint_dir};
    let root = std::fs::canonicalize(dir("hf-blob-swap")).unwrap();
    let (blobs, snap) = (root.join("blobs"), root.join("snapshots/rev1"));
    std::fs::create_dir(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    write_shard(&blobs.join(hf_blob()), &bf16(tensors(false)), 0);
    std::fs::write(snap.join("config.json"), "{}").unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), snap.join("model.safetensors")).unwrap();
    let opened = resolve_checkpoint_dir(&snap).unwrap();
    read_config(&opened).unwrap();
    std::fs::rename(&blobs, root.join("blobs-a")).unwrap();
    std::fs::create_dir(&blobs).unwrap();
    write_shard(&blobs.join(hf_blob()), &bf16(tensors(false)), 0);
    let mut bytes = std::fs::read(blobs.join(hf_blob())).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 0xff;
    std::fs::write(blobs.join(hf_blob()), bytes).unwrap();
    let ck = Checkpoint::open_dir(&opened).unwrap();
    let t = ck.tensors.values().max_by_key(|t| t.end).unwrap();
    let mut got = vec![0u8; 1];
    ck.read_range(t, t.len() - 1, &mut got).unwrap();
    let want = std::fs::read(root.join("blobs-a").join(hf_blob())).unwrap();
    assert_eq!(got[0], want[want.len() - 1], "the shard comes from the blob store the snapshot was opened with");
}

#[cfg(unix)]
#[test]
fn an_inside_link_is_resolved_without_walking_its_dot_dot() {
    // `sub/../weights.safetensors` is resolved to `weights.safetensors` before any open, so the
    // filesystem never resolves `..` from `sub`, which could have been moved elsewhere by then.
    // Here `sub` doesn't even exist: a walk of the link's text would fail on it.
    let g = dir("dot-dot");
    write_shard(&g.join("weights.safetensors"), &bf16(tensors(false)), 0);
    std::os::unix::fs::symlink("sub/../weights.safetensors", g.join("model.safetensors")).unwrap();
    bind(&Checkpoint::open(&g).unwrap(), &spec(false)).unwrap();
    // Now `sub` leads to `<outside>/inner`, whose parent holds a different shard (no `lm_head`)
    // under the same name, which is where a physical `..` from `sub` would land. It is not read.
    let outside = dir("dot-dot-outside");
    std::fs::create_dir(outside.join("inner")).unwrap();
    write_shard(&outside.join("weights.safetensors"), &bf16(tensors(true)), 0);
    std::os::unix::fs::symlink(outside.join("inner"), g.join("sub")).unwrap();
    let ck = Checkpoint::open(&g).unwrap();
    assert!(ck.tensors.contains_key("lm_head.weight"), "read the checkpoint's own shard");
}

// ---- SHA-788: codex-review follow-ups on #14 ----

#[cfg(unix)]
#[test]
fn an_unusable_blob_store_only_matters_to_checkpoints_that_link_into_it() {
    // <repo>/blobs is a symlink to another volume. A snapshot of plain files still opens; one that
    // links into the blob store fails, naming the store.
    let root = std::fs::canonicalize(dir("hf-blob-link")).unwrap();
    let volume = root.join("volume");
    std::fs::create_dir(&volume).unwrap();
    write_shard(&volume.join(hf_blob()), &bf16(tensors(false)), 0);
    std::os::unix::fs::symlink(&volume, root.join("blobs")).unwrap();
    let plain = root.join("snapshots/plain");
    std::fs::create_dir_all(&plain).unwrap();
    write_shard(&plain.join("model.safetensors"), &bf16(tensors(false)), 0);
    bind(&Checkpoint::open(&plain).unwrap(), &spec(false)).unwrap();
    let linked = root.join("snapshots/linked");
    std::fs::create_dir_all(&linked).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), linked.join("model.safetensors")).unwrap();
    let err = format!("{:#}", Checkpoint::open(&linked).err().expect("blob store is a symlink"));
    assert!(err.contains("links into the blob store") && err.contains("blobs"), "{err}");
}

// ---- SHA-788: codex-review cycle 1 follow-ups ----

// ---- SHA-788: codex-review cycle 2 follow-ups ----

#[cfg(unix)]
#[test]
fn a_blob_whose_bytes_do_not_match_its_name_is_refused() {
    // Whatever rearranged the cache (a blob store swapped between two checks and swapped back, a
    // revision moved to another repository, a corrupt download), the result is a blob name whose
    // bytes are another generation's. Each blob is verified against its name, so it is refused.
    let root = std::fs::canonicalize(dir("hf-mismatch")).unwrap();
    let (blobs, snap) = (root.join("blobs"), root.join("snapshots/rev"));
    std::fs::create_dir(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    write_shard(&blobs.join(hf_blob()), &bf16(tensors(true)), 0); // other bytes under A's name
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), snap.join("model.safetensors")).unwrap();
    let err = format!("{:#}", Checkpoint::open(&snap).err().expect("mismatched blob"));
    assert!(err.contains("hash to") && err.contains("not used"), "{err}");
    // With the right bytes it opens.
    write_shard(&blobs.join(hf_blob()), &bf16(tensors(false)), 0);
    bind(&Checkpoint::open(&snap).unwrap(), &spec(false)).unwrap();
}

#[cfg(unix)]
#[test]
fn a_git_object_blob_is_verified_over_the_bytes_read() {
    // Small files in a Hugging Face cache (config.json) are named by their Git object SHA1.
    use sha1::Digest;
    use sparkforge::weights::{read_config, resolve_checkpoint_dir};
    let root = std::fs::canonicalize(dir("hf-git-blob")).unwrap();
    let (blobs, snap) = (root.join("blobs"), root.join("snapshots/rev"));
    std::fs::create_dir(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    let config = b"{\"gen\": \"A\"}";
    let mut h = sha1::Sha1::new();
    h.update(format!("blob {}\0", config.len()).as_bytes());
    h.update(config);
    let name: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(blobs.join(&name), config).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{name}"), snap.join("config.json")).unwrap();
    assert!(read_config(&resolve_checkpoint_dir(&snap).unwrap()).unwrap().contains("\"A\""));
    std::fs::write(blobs.join(&name), b"{\"gen\": \"B\"}").unwrap();
    let err = format!("{:#}", read_config(&resolve_checkpoint_dir(&snap).unwrap()).unwrap_err());
    assert!(err.contains("hash to"), "{err}");
}


// ---- SHA-789: single-account admission ----

#[cfg(unix)]
#[test]
fn blobs_verify_on_the_calling_thread_when_no_worker_can_be_made() {
    // With no worker threads (as when the OS refuses to create one), every blob is still verified,
    // on the calling thread, and a mismatch is still refused rather than panicking.
    use sparkforge::weights::resolve_checkpoint_dir;
    let root = std::fs::canonicalize(dir("hf-serial")).unwrap();
    let (blobs, snap) = (root.join("blobs"), root.join("snapshots/rev"));
    std::fs::create_dir(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    write_shard(&blobs.join(hf_blob()), &bf16(tensors(false)), 0);
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), snap.join("model.safetensors")).unwrap();
    let ck = Checkpoint::open_dir_with(&resolve_checkpoint_dir(&snap).unwrap(), 0).unwrap();
    bind(&ck, &spec(false)).unwrap();
    write_shard(&blobs.join(hf_blob()), &bf16(tensors(true)), 0);
    let err = format!("{:#}", Checkpoint::open_dir_with(&resolve_checkpoint_dir(&snap).unwrap(), 0).err().expect("mismatch"));
    assert!(err.contains("hash to"), "{err}");
}

// ---- SHA-789: codex-review cycle 1 follow-ups ----

#[cfg(unix)]
#[test]
fn a_holder_that_outlasts_the_wait_never_leads_to_an_admission() {
    // The first attempt finds the lock held. During the pause before the next attempt, the
    // deadline passes and only then is the lock released, so the next attempt would get it.
    // Admission still times out. The pause is the test's own, so the order is exact.
    //
    // The clock starts at the first pause, not before `admit_with` is called. Admission computes
    // its deadline before its first attempt, and the first pause follows that attempt, so waiting
    // past `wait` from the first pause always ends after admission's own deadline, however long
    // the test thread is descheduled before admission starts (codex-review of #16, cycle 3).
    use sparkforge::weights::{admit_with, AdmitHooks};
    use std::time::{Duration, Instant};
    let lock = dir("late-release").join("sparkforge-gpu-admission.lock");
    std::fs::write(&lock, b"").unwrap();
    let holder = std::fs::OpenOptions::new().read(true).write(true).open(&lock).unwrap();
    holder.lock().unwrap();
    let holder = std::sync::Mutex::new(Some(holder));
    let wait = Duration::from_millis(10);
    let first_pause = std::sync::OnceLock::<Instant>::new();
    let pause = |_: Duration| {
        let start = *first_pause.get_or_init(Instant::now);
        while start.elapsed() <= wait {
            std::thread::sleep(Duration::from_millis(1));
        }
        holder.lock().unwrap().take(); // released after the deadline, before the next attempt
    };
    let err = format!("{:#}", admit_with(&lock, 0, 0, || Some(64 << 30), wait, &AdmitHooks { on_wait: &|| {}, sleep: &pause })
        .err().expect("the deadline passed while the lock was held"));
    assert!(err.contains("gave up after"), "{err}");
    assert!(holder.lock().unwrap().is_none(), "the lock was released before the last attempt");
}

#[cfg(unix)]
#[test]
fn a_load_waiting_on_a_replaced_lock_moves_to_the_new_file() {
    // Load A holds the lock. Load B opens it and reports waiting (so B certainly opened the old
    // file). The file is replaced and load C takes the new one. When A finishes, B gets the old
    // file's lock, sees the path now names a new file, and waits on C instead of being admitted
    // alongside it.
    use sparkforge::weights::{admit_with, AdmitHooks};
    use std::time::Duration;
    let lock = dir("relock").join("sparkforge-gpu-admission.lock");
    std::fs::write(&lock, b"").unwrap();
    let open = |p: &std::path::Path| std::fs::OpenOptions::new().read(true).write(true).open(p).unwrap();
    let a = open(&lock);
    a.lock().unwrap();
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let tx = std::sync::Mutex::new(tx);
    let path = lock.clone();
    let b = std::thread::spawn(move || {
        let on_wait = || {
            let _ = tx.lock().unwrap().send(());
        };
        admit_with(&path, 0, 0, || Some(64 << 30), Duration::from_secs(20), &AdmitHooks { on_wait: &on_wait, sleep: &std::thread::sleep }).map(|_held| ())
    });
    rx.recv_timeout(Duration::from_secs(10)).expect("B reports waiting on the old file");
    std::fs::remove_file(&lock).unwrap();
    std::fs::write(&lock, b"").unwrap();
    let c = open(&lock);
    c.lock().unwrap();
    drop(a);
    rx.recv_timeout(Duration::from_secs(10)).expect("B moved to the new file and waits on C");
    assert!(!b.is_finished(), "B must wait for C, which holds the file the path names now");
    drop(c);
    b.join().unwrap().unwrap();
}

// ---- SHA-785: a Hugging Face cache whose blobs link into a shared store ----

#[cfg(unix)]
#[test]
fn a_blob_that_links_into_a_shared_store_is_followed_and_verified() {
    // spark-5's cache: `<repo>/snapshots/<rev>/f -> ../../blobs/<sha256>`, and that blob is itself
    // a link to `hub/blobs/<xx>/<other id>`. The chain is followed because the bytes are checked
    // against the sha256 name; other bytes at the end of the chain are still refused.
    let hub = std::fs::canonicalize(dir("hf-dedup")).unwrap();
    let store = hub.join("blobs/ab");
    std::fs::create_dir_all(&store).unwrap();
    write_shard(&store.join("chunk-id-1"), &bf16(tensors(false)), 0);
    let repo = hub.join("models--org--name");
    let (blobs, snap) = (repo.join("blobs"), repo.join("snapshots/rev"));
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    std::os::unix::fs::symlink("../../blobs/ab/chunk-id-1", blobs.join(hf_blob())).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), snap.join("model.safetensors")).unwrap();
    bind(&Checkpoint::open(&snap).unwrap(), &spec(false)).unwrap();
    // The store's bytes change under the same chain: the name no longer matches, so it is refused.
    write_shard(&store.join("chunk-id-1"), &bf16(tensors(true)), 0);
    let err = format!("{:#}", Checkpoint::open(&snap).err().expect("mismatch"));
    assert!(err.contains("hash to"), "{err}");
}

// ---- SHA-785: codex-review cycle 1 follow-ups ----

#[test]
fn planning_from_an_index_applies_the_loaders_shard_name_rule() {
    use sparkforge::weights::{resolve_checkpoint_dir, tensor_names};
    let d = dir("names-escape");
    for bad in [serde_json::json!("../outside.safetensors"), serde_json::json!("/abs/model.safetensors")] {
        let mut map = serde_json::Map::new();
        map.insert("model.embed_tokens.weight".into(), bad.clone());
        write_index(&d, map);
        let err = format!("{:#}", tensor_names(&resolve_checkpoint_dir(&d).unwrap()).unwrap_err());
        assert!(err.contains("not a plain relative path"), "{bad}: {err}");
    }
    let mut map = serde_json::Map::new();
    map.insert("model.embed_tokens.weight".into(), serde_json::json!(7));
    write_index(&d, map);
    assert!(tensor_names(&resolve_checkpoint_dir(&d).unwrap()).is_err(), "a non-string shard reference");
}

#[test]
fn a_plan_config_must_be_the_checkpoints_own() {
    // `plan <config> --checkpoint <dir>` reads the checkpoint's config.json through the opened
    // directory; a config from another generation is refused instead of being mixed with this
    // checkpoint's modules.
    use sparkforge::weights::{resolve_checkpoint_dir, spec_from_checkpoint};
    let d = single("cfg-match", bf16(tensors(false)));
    let ours = CFG.replace("TIE", "false");
    std::fs::write(d.join("config.json"), &ours).unwrap();
    let s = spec_from_checkpoint(&resolve_checkpoint_dir(&d).unwrap(), None, Some(&ours)).unwrap();
    assert!(s.resolved.is_some() && s.vocab == 64);
    let other = CFG.replace("TIE", "false").replace("\"vocab_size\": 64", "\"vocab_size\": 65");
    let err = format!("{:#}", spec_from_checkpoint(&resolve_checkpoint_dir(&d).unwrap(), None, Some(&other)).unwrap_err());
    assert!(err.contains("differs from"), "{err}");
}

#[test]
fn a_checkpoint_spec_comes_from_the_checkpoints_own_config() {
    // codex-review fp:4abc58b2da3e: the spec is parsed inside the helper from the config.json read
    // through the opened directory, so a library caller can't resolve another checkpoint's spec
    // against these tensor names.
    use sparkforge::weights::{resolve_checkpoint_dir, spec_from_checkpoint};
    let d = single("cfg-own", bf16(tensors(true)));
    std::fs::write(d.join("config.json"), CFG.replace("TIE", "true")).unwrap();
    let s = spec_from_checkpoint(&resolve_checkpoint_dir(&d).unwrap(), None, None).unwrap();
    assert!(s.tie_embeddings && s.resolved.is_some(), "the checkpoint's own config ties embeddings");
}

#[cfg(unix)]
#[test]
fn a_blob_link_out_of_the_shared_store_is_refused_before_opening() {
    // Only `../../blobs/<xx>/<name>` is followed, inside the cache's shared store. A blob linking
    // anywhere else (here an absolute path to a valid shard) is refused without opening it.
    let hub = std::fs::canonicalize(dir("hf-escape")).unwrap();
    let outside = std::fs::canonicalize(dir("hf-escape-outside")).unwrap();
    write_shard(&outside.join("private.safetensors"), &bf16(tensors(false)), 0);
    let repo = hub.join("models--org--name");
    let (blobs, snap) = (repo.join("blobs"), repo.join("snapshots/rev"));
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::create_dir_all(hub.join("blobs")).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{}", hf_blob()), snap.join("model.safetensors")).unwrap();
    for target in [outside.join("private.safetensors").display().to_string(), "../../blobs/ab/../../x".into(), "../../other/ab/x".into(), "../../blobs/abc/x".into()] {
        let _ = std::fs::remove_file(blobs.join(hf_blob()));
        std::os::unix::fs::symlink(&target, blobs.join(hf_blob())).unwrap();
        let err = format!("{:#}", Checkpoint::open(&snap).err().expect("escaping chain"));
        assert!(err.contains("only into the cache's shared store"), "{target}: {err}");
    }
}

// ---- SHA-767: hybrid checkpoints are checked tensor by tensor ----


fn tiny_spec() -> ModelSpec {
    ModelSpec::from_json(&common::tiny_config().to_string(), None).unwrap()
}

fn tiny_tensors() -> Vec<common::Tensor> {
    common::hybrid_tensors(&tiny_spec(), "layers.1.ple", common::TINY_TABLE_ROWS)
}

/// Opens a hybrid checkpoint the way `sparkforge weights` does and checks every tensor.
fn check_dir(d: &std::path::Path) -> anyhow::Result<sparkforge::weights::HybridCheck> {
    use sparkforge::weights::{check_hybrid, resolve_checkpoint_dir, spec_from_checkpoint};
    let dir = resolve_checkpoint_dir(d)?;
    let spec = spec_from_checkpoint(&dir, None, None)?;
    let ck = Checkpoint::open_dir(&dir)?;
    check_hybrid(&ck, &spec)
}

fn check_tensors(name: &str, ts: &[common::Tensor]) -> String {
    format!("{:#}", check_dir(&write_hybrid(name, &common::tiny_config(), ts, &common::TINY_VOCAB_SIZES)).unwrap_err())
}

#[test]
fn a_hybrid_checkpoint_passes_when_every_tensor_matches_the_plan() {
    let ts = tiny_tensors();
    let c = check_dir(&write_hybrid("hy-ok", &common::tiny_config(), &ts, &common::TINY_VOCAB_SIZES)).unwrap();
    // Every tensor but the two vision-tower ones is checked; the PLE shards hold 60 rows / 4.
    assert_eq!((c.tensors, c.skipped, c.table_rows), (ts.len() - 2, 2, 15));
    let group = |g: &str| c.groups.iter().find(|(k, _)| *k == g).map(|(_, b)| *b).unwrap();
    // 4 layers x 4 experts x 3 projections of 32 x 64 NVFP4: values, E4M3 scales, two F32 scalars.
    assert_eq!(group("routed_experts"), 4 * 4 * 3 * (32 * 32 + 32 * 4 + 8));
    // 4 shards of 15 rows x 16 FP4 values (8 bytes) + 1 scale, and the F32 global scale.
    assert_eq!(group("ple_table"), 4 * 15 * (8 + 1) + 4);
    assert_eq!(c.bytes, ts.iter().filter(|t| !t.0.starts_with("model.visual.")).map(|t| {
        t.2.iter().product::<usize>() as u64 * match t.1 { "I64" => 8, "F32" => 4, "BF16" => 2, _ => 1 }
    }).sum::<u64>());
}

#[test]
fn a_hybrid_tensor_with_the_wrong_dtype_or_shape_is_named() {
    let mut ts = tiny_tensors();
    let scale = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale";
    ts.iter_mut().find(|t| t.0 == scale).unwrap().1 = "F8_E4M3";
    let expert = "model.language_model.layers.2.mlp.experts.3.down_proj.weight";
    ts.iter_mut().find(|t| t.0 == expert).unwrap().2 = vec![64, 32];
    let err = check_tensors("hy-dtype", &ts);
    assert!(err.contains(&format!("{scale}: expected U8 [96, 2], found F8_E4M3 [96, 2]")), "{err}");
    assert!(err.contains(&format!("{expert}: expected U8 [64, 16], found U8 [64, 32]")), "{err}");
    assert!(err.contains("2 problem(s)"), "{err}");
}

#[test]
fn a_missing_scale_and_an_unexpected_text_tensor_fail_while_vision_is_skipped() {
    // Module-level resolution passes (the expert's module exists); the tensor check names the
    // missing activation scale and a text tensor the plan has no use for.
    let mut ts = tiny_tensors();
    let gone = "model.language_model.layers.1.mlp.experts.0.gate_proj.input_scale";
    ts.retain(|t| t.0 != gone);
    ts.push(("model.language_model.layers.0.linear_attn.extra.weight".into(), "BF16", vec![4]));
    ts.push(("model.visual.merger.norm.weight".into(), "BF16", vec![4]));
    let err = check_tensors("hy-missing", &ts);
    assert!(err.contains("missing layers.1.mlp.experts.0.gate_proj.input_scale (expected F32 [])"), "{err}");
    assert!(err.contains("unexpected model.language_model.layers.0.linear_attn.extra.weight"), "{err}");
    assert!(!err.contains("model.visual"), "{err}");
}

#[test]
fn two_spellings_of_one_hybrid_tensor_are_refused() {
    let mut ts = tiny_tensors();
    ts.push(("model.layers.0.linear_attn.A_log".into(), "BF16", vec![2]));
    let err = check_tensors("hy-alias", &ts);
    assert!(err.contains("model.language_model.layers.0.linear_attn.A_log") && err.contains("model.layers.0.linear_attn.A_log")
        && err.contains("two names"), "{err}");
}

#[test]
fn a_format_map_that_disagrees_with_the_tensors_is_refused() {
    // The map says GDN in_proj_z is NVFP4, but the checkpoint holds MXFP8 tensors for it: the
    // tensor set of the priced format is what must exist.
    let mut cfg = common::tiny_config();
    cfg["quantization_config"]["quantized_layers"]["model.language_model.layers.0.linear_attn.in_proj_z"] =
        serde_json::json!({"quant_algo": "NVFP4", "group_size": 16});
    let d = write_hybrid("hy-map", &cfg, &tiny_tensors(), &common::TINY_VOCAB_SIZES);
    let err = format!("{:#}", check_dir(&d).unwrap_err());
    let z = "layers.0.linear_attn.in_proj_z";
    assert!(err.contains(&format!("model.language_model.{z}.weight: expected U8 [32, 32], found F8_E4M3 [32, 64]")), "{err}");
    assert!(err.contains(&format!("model.language_model.{z}.weight_scale: expected F8_E4M3 [32, 4], found U8 [32, 2]")), "{err}");
    assert!(err.contains(&format!("missing {z}.weight_scale_2")) && err.contains(&format!("missing {z}.input_scale")), "{err}");
}

#[test]
fn ple_table_rows_come_from_the_per_head_sizes() {
    // Sizes summing to 64 make shards of 16 rows, so the 15-row shards no longer match.
    let d = write_hybrid("hy-rows", &common::tiny_config(), &tiny_tensors(), &[11, 13, 17, 23]);
    let err = format!("{:#}", check_dir(&d).unwrap_err());
    assert!(err.contains("shard_0.weight: expected U8 [16, 8], found U8 [15, 8]"), "{err}");
    // A head below ngram_vocab_size_base (10) is refused.
    let d = write_hybrid("hy-small", &common::tiny_config(), &tiny_tensors(), &[9, 13, 17, 21]);
    let err = format!("{:#}", check_dir(&d).unwrap_err());
    assert!(err.contains("PLE head 0 has 9 rows, below ngram_vocab_size_base 10"), "{err}");
    // Sizes whose padded total doesn't split into 4 shards: 61 rows pad to 64 (multiple of 4),
    // which does split, while a multiple of 6 doesn't.
    let mut cfg = common::tiny_config();
    cfg["text_config"]["make_ngram_vocab_size_divisible_by"] = serde_json::json!(6);
    let d = write_hybrid("hy-split", &cfg, &tiny_tensors(), &[11, 13, 17, 20]);
    let err = format!("{:#}", check_dir(&d).unwrap_err());
    assert!(err.contains("66 rows (padded to a multiple of 6), which do not split evenly into 4 shards"), "{err}");
}

#[test]
fn a_hybrid_checkpoint_is_refused_by_the_dense_binding() {
    let d = write_hybrid("hy-bind", &common::tiny_config(), &tiny_tensors(), &common::TINY_VOCAB_SIZES);
    let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
    let spec = sparkforge::weights::spec_from_checkpoint(&dir, None, None).unwrap();
    let err = format!("{:#}", bind(&Checkpoint::open_dir(&dir).unwrap(), &spec).unwrap_err());
    assert!(err.contains("load into GPU memory with `sparkforge load-weights`") && err.contains("not through the dense binding"), "{err}");
}

// ---- SHA-767: codex-review cycle 1 follow-ups on #18 ----

#[test]
fn planning_needs_no_shard_count_but_checking_the_tensors_does() {
    // fp:3600da7c4e15: `split_ngram_parts` is optional for planning. `plan --checkpoint` resolves
    // the same checkpoint without it, while `weights` refuses to check the table's shards.
    let mut cfg = common::tiny_config();
    cfg["text_config"].as_object_mut().unwrap().remove("split_ngram_parts");
    let d = write_hybrid("hy-noparts", &cfg, &tiny_tensors(), &common::TINY_VOCAB_SIZES);
    let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
    let given = std::fs::read_to_string(d.join("config.json")).unwrap();
    let spec = sparkforge::weights::spec_from_checkpoint(&dir, None, Some(&given)).unwrap();
    assert!(spec.hybrid.as_ref().unwrap().ple.split_parts.is_none() && spec.resolved.is_some());
    let err = format!("{:#}", check_dir(&d).unwrap_err());
    assert!(err.contains("split_ngram_parts"), "{err}");
}

#[test]
fn the_checked_table_rows_price_the_resident_budget() {
    // fp:06e6fe0d9f78: two checkpoints of one config whose per-head sizes make 15- and 16-row
    // shards. Priced at the checked rows, the second costs 4 shards x 1 row x 9 bytes more.
    use sparkforge::{plan, target::GB10};
    let resident = |name: &str, sizes: &[i64], rows: usize| {
        let ts = common::hybrid_tensors(&tiny_spec(), "layers.1.ple", rows);
        let d = write_hybrid(name, &common::tiny_config(), &ts, sizes);
        let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
        let mut spec = sparkforge::weights::spec_from_checkpoint(&dir, None, None).unwrap();
        let c = sparkforge::weights::check_hybrid(&Checkpoint::open_dir(&dir).unwrap(), &spec).unwrap();
        assert_eq!(c.table_rows, rows);
        sparkforge::weights::apply_checked_table(&mut spec, &c);
        assert_eq!(spec.hybrid.as_ref().unwrap().ple.table_rows(), 4 * rows as u64);
        plan::build(&spec, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 }).budget.weight_gib * (1u64 << 30) as f64
    };
    let (a, b) = (resident("hy-rows15", &[11, 13, 17, 19], 15), resident("hy-rows16", &[11, 13, 17, 23], 16));
    assert!((b - a - 36.0).abs() < 1e-3, "{}", b - a);
}

#[test]
fn each_mtp_layer_is_checked_and_priced_at_its_own_formats() {
    // fp:de9c37520703: two draft layers (set on the spec; configs are still limited to one), with
    // layer 0's v_proj MXFP8 and layer 1's BF16. Each layer's tensors follow its own format, and
    // the budget prices layer 0's projection alone at MXFP8.
    use sparkforge::{config::WeightFormat, plan, target::GB10};
    let mut spec = tiny_spec();
    spec.hybrid.as_mut().unwrap().mtp_layers = 2;
    // Layer 1's experts get their own entry: each layer's formats are looked up at its own index.
    spec.formats.modules.insert("mtp.layers.1.mlp.experts".into(), WeightFormat::Nvfp4WeightOnly);
    let bf16 = spec.clone();
    spec.formats.modules.insert("mtp.layers.0.self_attn.v_proj".into(), WeightFormat::Mxfp8);
    let mut ts = common::hybrid_tensors(&spec, "layers.1.ple", common::TINY_TABLE_ROWS);
    let v = "mtp.layers.0.self_attn.v_proj";
    ts.retain(|t| t.0 != format!("{v}.weight"));
    ts.push((format!("{v}.weight"), "F8_E4M3", vec![32, 64]));
    ts.push((format!("{v}.weight_scale"), "U8", vec![32, 2]));
    let d = write_hybrid("hy-mtp2", &common::tiny_config(), &ts, &common::TINY_VOCAB_SIZES);
    let ck = Checkpoint::open(&d).unwrap();
    let mut resolved = spec.clone();
    resolved.resolve_modules(ck.tensors.keys().map(String::as_str)).unwrap();
    sparkforge::weights::check_hybrid(&ck, &resolved).unwrap();
    let gib = |s: &ModelSpec| plan::build(s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 }).budget.weight_gib * (1u64 << 30) as f64;
    assert!((gib(&bf16) - gib(&resolved) - 32.0 * 64.0 * (2.0 - 1.03125)).abs() < 1e-3, "{}", gib(&bf16) - gib(&resolved));
    assert_eq!(gib(&spec), gib(&resolved), "config-only and checkpoint pricing agree");
}

#[test]
fn a_tied_nvfp4_embedding_carries_the_head_gemvs_activation_scale() {
    // fp:7027c462f116: tied, the embedding matrix is also the head GEMV's operand, so NVFP4 needs
    // its `input_scale`; weight-only NVFP4 (W4A16) doesn't.
    for (algo, scale) in [("NVFP4", true), ("W4A16_NVFP4", false)] {
        let mut cfg = common::tiny_config();
        cfg["tie_word_embeddings"] = serde_json::json!(true);
        cfg["text_config"]["tie_word_embeddings"] = serde_json::json!(true);
        cfg["quantization_config"]["quantized_layers"]["model.language_model.embed_tokens"] = serde_json::json!({"quant_algo": algo, "group_size": 16});
        let spec = ModelSpec::from_json(&cfg.to_string(), None).unwrap();
        let e = "model.language_model.embed_tokens";
        let mut ts = common::hybrid_tensors(&spec, "layers.1.ple", common::TINY_TABLE_ROWS);
        ts.retain(|t| t.0 != format!("{e}.weight"));
        ts.extend([(format!("{e}.weight"), "U8", vec![128, 32]), (format!("{e}.weight_scale"), "F8_E4M3", vec![128, 4]), (format!("{e}.weight_scale_2"), "F32", vec![])]);
        let without = check_dir(&write_hybrid(&format!("hy-tied-{scale}"), &cfg, &ts, &common::TINY_VOCAB_SIZES));
        ts.push((format!("{e}.input_scale"), "F32", vec![]));
        let with = check_dir(&write_hybrid(&format!("hy-tied-scale-{scale}"), &cfg, &ts, &common::TINY_VOCAB_SIZES));
        if scale {
            assert!(with.is_ok(), "{algo}: {:#}", with.unwrap_err());
            let err = format!("{:#}", without.unwrap_err());
            assert!(err.contains("missing embed_tokens.input_scale"), "{algo}: {err}");
        } else {
            assert!(without.is_ok(), "{algo}: {:#}", without.unwrap_err());
            assert!(format!("{:#}", with.unwrap_err()).contains("unexpected model.language_model.embed_tokens.input_scale"), "{algo}");
        }
    }
}

#[test]
fn ple_row_counts_that_overflow_when_padded_are_an_error() {
    // fp:c89c702ebdf6: sizes whose total fits in 64 bits but whose padding to a multiple of 4
    // doesn't.
    let d = write_hybrid("hy-overflow", &common::tiny_config(), &tiny_tensors(), &[11, 13, i64::MAX - 24, i64::MAX]);
    let err = format!("{:#}", check_dir(&d).unwrap_err());
    assert!(err.contains("overflow a 64-bit row count"), "{err}");
}

// ---- SHA-767: codex-review cycle 2 follow-ups on #18 ----

fn sparkforge_cli(args: &[&std::ffi::OsStr]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_sparkforge")).args(args).output().unwrap()
}

#[test]
fn a_w4a16_planning_override_works_with_a_checkpoint_while_weights_checks_the_stored_formats() {
    // fp:e5adf673bf61: resolving names for planning lists modules only, so a W4A16 override (not a
    // format the loader reads) still plans against the checkpoint. `weights` checks the formats
    // the checkpoint declares, and a spec carrying the override fails the tensor check.
    let d = write_hybrid("hy-w4a16", &common::tiny_config(), &tiny_tensors(), &common::TINY_VOCAB_SIZES);
    let cfg = d.join("config.json");
    let out = sparkforge_cli(&["plan".as_ref(), cfg.as_os_str(), "--checkpoint".as_ref(), d.as_os_str(), "--format".as_ref(), "w4a16".as_ref()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("W4A16"), "{}", String::from_utf8_lossy(&out.stdout));
    let out = sparkforge_cli(&["weights".as_ref(), d.as_os_str()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("tensors match the plan"));
    let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
    let spec = sparkforge::weights::spec_from_checkpoint(&dir, Some(sparkforge::config::WeightFormat::W4A16 { group: 128 }), None).unwrap();
    let err = format!("{:#}", sparkforge::weights::check_hybrid(&Checkpoint::open_dir(&dir).unwrap(), &spec).unwrap_err());
    assert!(err.contains("W4A16") && err.contains("only BF16, MXFP8 and NVFP4"), "{err}");
}

#[test]
fn without_a_shard_count_the_table_must_still_hold_numbered_shards() {
    // fp:6b3518dc167a: with `split_ngram_parts` omitted, planning accepts the complete checkpoint
    // but refuses one whose table has no shards, or a gap in their numbering.
    let mut cfg = common::tiny_config();
    cfg["text_config"].as_object_mut().unwrap().remove("split_ngram_parts");
    let plan = |name: &str, ts: &[common::Tensor]| {
        let d = write_hybrid(name, &cfg, ts, &common::TINY_VOCAB_SIZES);
        sparkforge::weights::spec_from_checkpoint(&sparkforge::weights::resolve_checkpoint_dir(&d).unwrap(), None, None).map(|_| ())
    };
    plan("hy-shards-ok", &tiny_tensors()).unwrap();
    let none: Vec<_> = tiny_tensors().into_iter().filter(|t| !t.0.contains(".shard_")).collect();
    let err = format!("{:#}", plan("hy-shards-none", &none).unwrap_err());
    assert!(err.contains("must hold shards") && err.contains("none"), "{err}");
    let gap: Vec<_> = tiny_tensors().into_iter().filter(|t| !t.0.contains(".shard_1.")).collect();
    let err = format!("{:#}", plan("hy-shards-gap", &gap).unwrap_err());
    assert!(err.contains("must hold shards") && err.contains("ngram_embedding.shard_3"), "{err}");
}

#[test]
fn the_shared_mtp_input_projections_are_resident_once() {
    // fp:d53ae8818cd5: `mtp.fc_embedding` and `mtp.fc_hidden` (and the final mixer) exist once,
    // however many draft layers there are: a second layer adds only its own layer, so
    // 2 x (one layer) - (two layers) is exactly those shared bytes.
    use sparkforge::{plan, target::GB10};
    let resident = |layers: u32| {
        let mut s = tiny_spec();
        s.hybrid.as_mut().unwrap().mtp_layers = layers;
        // Both draft layers' experts in the same format, so the layers differ in nothing else.
        s.formats.modules.insert("mtp.layers.1.mlp.experts".into(), sparkforge::config::WeightFormat::Nvfp4WeightOnly);
        plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 }).budget.weight_gib * (1u64 << 30) as f64
    };
    let (w0, w1, w2) = (resident(0), resident(1), resident(2));
    // fc: 2 x 64 x 64 BF16; final mixer: down and up (8 x 256 each, BF16) plus a BF16 norm of 256.
    // fp:04b498446b41 on #19: the two pre-FC norms (`mtp.pre_fc_norm_embedding`, hidden width 64,
    // and `mtp.pre_fc_norm_hidden`, residual width 256, both BF16) are shared too: 640 bytes.
    let pre_fc_norms = (64.0 + 256.0) * 2.0;
    assert_eq!(pre_fc_norms, 640.0);
    let shared = 2.0 * 64.0 * 64.0 * 2.0 + (2.0 * 8.0 * 256.0) * 2.0 + 256.0 * 2.0 + pre_fc_norms;
    assert!((2.0 * (w1 - w0) - (w2 - w0) - shared).abs() < 1e-3, "{}", 2.0 * (w1 - w0) - (w2 - w0));
}

#[test]
fn an_overflowing_ple_vocab_base_is_a_config_error_not_a_panic() {
    // fp:8269509d4acb: `ngram_vocab_size_base` near i64::MAX times 4 heads used to overflow while
    // building the plan, before the checkpoint was checked. It is refused when the config is parsed.
    let mut cfg = common::tiny_config();
    cfg["text_config"]["ngram_vocab_size_base"] = serde_json::json!(i64::MAX);
    let err = format!("{:#}", ModelSpec::from_json(&cfg.to_string(), None).unwrap_err());
    assert!(err.contains("overflows a 64-bit row count"), "{err}");
    let d = write_hybrid("hy-base", &cfg, &tiny_tensors(), &common::TINY_VOCAB_SIZES);
    let out = sparkforge_cli(&["weights".as_ref(), d.as_os_str()]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success() && stderr.contains("overflows a 64-bit row count") && !stderr.contains("panicked"), "{stderr}");
}

#[test]
fn checked_table_rows_do_not_survive_resolving_another_checkpoint() {
    // fp:1b31cfcd2a33: a spec checked against the 15-row checkpoint and then resolved onto the
    // 16-row one prices the table from the config again, until the second checkpoint is checked.
    use sparkforge::{plan, target::GB10};
    let open = |name: &str, sizes: &[i64], rows: usize| {
        let ts = common::hybrid_tensors(&tiny_spec(), "layers.1.ple", rows);
        Checkpoint::open(&write_hybrid(name, &common::tiny_config(), &ts, sizes)).unwrap()
    };
    let (a, b) = (open("hy-reuse15", &[11, 13, 17, 19], 15), open("hy-reuse16", &[11, 13, 17, 23], 16));
    let rows = |s: &ModelSpec| s.hybrid.as_ref().unwrap().ple.checked_rows;
    let mut s = tiny_spec();
    s.resolve_modules(a.tensors.keys().map(String::as_str)).unwrap();
    let c = sparkforge::weights::check_hybrid(&a, &s).unwrap();
    sparkforge::weights::apply_checked_table(&mut s, &c);
    assert_eq!(rows(&s), Some(60));
    s.resolve_modules(b.tensors.keys().map(String::as_str)).unwrap();
    assert_eq!(rows(&s), None);
    let gib = |s: &ModelSpec| plan::build(s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 }).budget.weight_gib;
    let mut fresh = tiny_spec();
    fresh.resolve_modules(b.tensors.keys().map(String::as_str)).unwrap();
    assert_eq!(gib(&s), gib(&fresh));
    let c = sparkforge::weights::check_hybrid(&b, &s).unwrap();
    sparkforge::weights::apply_checked_table(&mut s, &c);
    assert_eq!(rows(&s), Some(64));
}

// ---- SHA-767: codex-review cycle 3 follow-ups on #18 ----

/// The tiny config with its PLE table left at the BF16 default, and matching tensors.
fn bf16_table() -> (serde_json::Value, Vec<common::Tensor>) {
    let mut cfg = common::tiny_config();
    cfg["quantization_config"]["quantized_layers"].as_object_mut().unwrap().remove("model.language_model.layers.1.ple.ple_embedding.ngram_embedding");
    let table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    let mut ts: Vec<common::Tensor> = tiny_tensors().into_iter().filter(|t| !t.0.starts_with(table)).collect();
    for j in 0..4 {
        ts.push((format!("{table}.shard_{j}.weight"), "BF16", vec![common::TINY_TABLE_ROWS, 16]));
    }
    (cfg, ts)
}

#[test]
fn an_nvfp4_planning_override_needs_no_table_scale_from_a_bf16_table() {
    // fp:8bf2edacbd78: a BF16 table has no global scale. Planning it with `--format nvfp4` lists
    // modules only, so nothing format-specific is required; `weights` checks the table as BF16.
    let (cfg, ts) = bf16_table();
    let d = write_hybrid("hy-nvfp4-plan", &cfg, &ts, &common::TINY_VOCAB_SIZES);
    let c = d.join("config.json");
    let out = sparkforge_cli(&["plan".as_ref(), c.as_os_str(), "--checkpoint".as_ref(), d.as_os_str(), "--format".as_ref(), "nvfp4".as_ref()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = sparkforge_cli(&["weights".as_ref(), d.as_os_str()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let c = check_dir(&d).unwrap();
    assert_eq!(c.groups.iter().find(|(g, _)| *g == "ple_table").unwrap().1, 4 * 15 * 16 * 2);
}

#[test]
fn a_sole_model_level_table_beside_the_ple_projections_resolves() {
    // fp:545c13177f21: projections under `model.language_model.ple` and the only n-gram embedding
    // at the bare `model.language_model.ple_embedding` are one PLE and one table.
    let mut cfg = common::tiny_config();
    let q = cfg["quantization_config"]["quantized_layers"].as_object_mut().unwrap();
    q.remove("model.language_model.layers.1.ple.ple_embedding.ngram_embedding");
    q.insert("model.language_model.ple_embedding.ngram_embedding".into(), serde_json::json!({"quant_algo": "NVFP4", "group_size": 16}));
    let ts: Vec<common::Tensor> = common::hybrid_tensors(&tiny_spec(), "ple", common::TINY_TABLE_ROWS).into_iter()
        .map(|(n, d, s)| (n.replace("model.language_model.ple.ple_embedding", "model.language_model.ple_embedding"), d, s)).collect();
    let d = write_hybrid("hy-bare", &cfg, &ts, &common::TINY_VOCAB_SIZES);
    let c = check_dir(&d).unwrap();
    assert_eq!(c.table_rows, 15);
    // The same projections with tables at both the nested and the bare path: two tables.
    let mut both = ts.clone();
    both.push(("model.language_model.ple.ple_embedding.ngram_embedding.shard_0.weight".into(), "U8", vec![15, 8]));
    let err = format!("{:#}", check_dir(&write_hybrid("hy-bare-both", &cfg, &both, &common::TINY_VOCAB_SIZES)).unwrap_err());
    assert!(err.contains("PLE n-gram tables at ple.ple_embedding and ple_embedding"), "{err}");
}

#[test]
fn a_hollow_shard_or_a_weight_at_the_experts_path_is_refused_on_resolve() {
    // fp:a2758f4d8976: with no configured count, a shard holding only an observer has no module
    // at its own path; a weight at `layers.0.mlp.experts` itself is no planned operand.
    let mut cfg = common::tiny_config();
    cfg["text_config"].as_object_mut().unwrap().remove("split_ngram_parts");
    let resolve = |name: &str, ts: &[common::Tensor]| {
        let d = write_hybrid(name, &cfg, ts, &common::TINY_VOCAB_SIZES);
        sparkforge::weights::spec_from_checkpoint(&sparkforge::weights::resolve_checkpoint_dir(&d).unwrap(), None, None).map(|_| ())
    };
    let shard = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1";
    let mut hollow: Vec<_> = tiny_tensors().into_iter().filter(|t| !t.0.starts_with(&format!("{shard}."))).collect();
    hollow.push((format!("{shard}.observer.weight"), "BF16", vec![4]));
    let err = format!("{:#}", resolve("hy-hollow", &hollow).unwrap_err());
    assert!(err.contains("layers.1.ple.ple_embedding.ngram_embedding.shard_1") && err.contains("no tensors of their own"), "{err}");
    let mut root = tiny_tensors();
    root.push(("model.language_model.layers.0.mlp.experts.weight".into(), "BF16", vec![4]));
    let err = format!("{:#}", resolve("hy-experts-root", &root).unwrap_err());
    assert!(err.contains("under layers.0.mlp.experts that the plan doesn't price: layers.0.mlp.experts"), "{err}");
}

#[test]
fn the_mtp_shared_expert_gate_is_priced_at_its_own_format() {
    // fp:1a87af5b409b: an MXFP8 MTP shared-expert gate passes the tensor check, lowers the resident
    // budget by 64 x (2 - 1.03125) = 62 bytes, and its entry is used.
    use sparkforge::{plan, target::GB10};
    let mut cfg = common::tiny_config();
    let gate = "mtp.layers.0.mlp.shared_expert_gate";
    cfg["quantization_config"]["quantized_layers"][gate] = serde_json::json!({"quant_algo": "MXFP8", "group_size": 32});
    let mut ts: Vec<_> = tiny_tensors().into_iter().filter(|t| t.0 != format!("{gate}.weight")).collect();
    ts.push((format!("{gate}.weight"), "F8_E4M3", vec![1, 64]));
    ts.push((format!("{gate}.weight_scale"), "U8", vec![1, 2]));
    let d = write_hybrid("hy-mtp-gate", &cfg, &ts, &common::TINY_VOCAB_SIZES);
    check_dir(&d).unwrap();
    let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
    let opts = plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 };
    let p = plan::build(&sparkforge::weights::spec_from_checkpoint(&dir, None, None).unwrap(), &GB10, &opts);
    let base = plan::build(&tiny_spec(), &GB10, &opts);
    assert!(((base.budget.weight_gib - p.budget.weight_gib) * (1u64 << 30) as f64 - 62.0).abs() < 1e-3);
    assert!(!p.notes.iter().any(|n| n.contains(gate)), "{:#?}", p.notes);
}

#[test]
fn an_overflowing_ple_head_count_is_a_config_error() {
    // fp:a6ec42c0ccb6: (3 - 1) x 2^31 heads overflows 32 bits before any row arithmetic.
    let mut cfg = common::tiny_config();
    cfg["text_config"]["heads_per_ngram"] = serde_json::json!(2147483648u64);
    let err = format!("{:#}", ModelSpec::from_json(&cfg.to_string(), None).unwrap_err());
    assert!(err.contains("overflows the PLE head count"), "{err}");
}

#[test]
fn an_extreme_shard_count_fails_promptly_on_the_live_shards() {
    // fp:d21035eb929c: split_ngram_parts of u32::MAX with 4 live shards fails from the live names,
    // without listing a module per declared shard.
    let mut cfg = common::tiny_config();
    cfg["text_config"]["split_ngram_parts"] = serde_json::json!(u32::MAX);
    let d = write_hybrid("hy-parts", &cfg, &tiny_tensors(), &common::TINY_VOCAB_SIZES);
    let t0 = std::time::Instant::now();
    let err = format!("{:#}", check_dir(&d).unwrap_err());
    assert!(err.contains("split_ngram_parts is 4294967295") && err.contains("lacks 4294967291 of them"), "{err}");
    assert!(t0.elapsed() < std::time::Duration::from_secs(5), "{:?}", t0.elapsed());
}

#[test]
fn a_failed_switch_keeps_the_previous_checkpoints_resolution_and_rows() {
    // fp:a2fd6e1bac29: a spec checked against checkpoint A, then resolved onto a checkpoint B with
    // two PLE tables, fails and keeps A's formats, checked rows and budget.
    use sparkforge::{plan, target::GB10};
    let a = Checkpoint::open(&write_hybrid("hy-switch-a", &common::tiny_config(), &tiny_tensors(), &common::TINY_VOCAB_SIZES)).unwrap();
    let mut s = tiny_spec();
    s.resolve_modules(a.tensors.keys().map(String::as_str)).unwrap();
    let c = sparkforge::weights::check_hybrid(&a, &s).unwrap();
    sparkforge::weights::apply_checked_table(&mut s, &c);
    let gib = |s: &ModelSpec| plan::build(s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 }).budget.weight_gib;
    let before = gib(&s);
    let mut names: Vec<String> = a.tensors.keys().cloned().collect();
    names.push("model.language_model.ple_embedding.ngram_embedding.shard_0.weight".into());
    let err = format!("{:#}", s.resolve_modules(names.iter().map(String::as_str)).unwrap_err());
    assert!(err.contains("PLE n-gram tables"), "{err}");
    assert_eq!(s.hybrid.as_ref().unwrap().ple.checked_rows, Some(60));
    assert!(s.resolved.is_some());
    assert_eq!(gib(&s), before);
    sparkforge::weights::check_hybrid(&a, &s).unwrap();
}

// ---- SHA-791: codex-review follow-ups on #18 ----

#[test]
fn an_mtp_input_projection_is_checked_and_priced_at_its_declared_format() {
    // fp:25aaaee34a54: an MXFP8 entry for `mtp.fc_embedding` beside a BF16 weight with no scale
    // fails the tensor check; the MXFP8 tensors pass and price 64 x 64 x (2 - 1.03125) = 3,968
    // bytes below the BF16 projection.
    use sparkforge::{plan, target::GB10};
    let mut cfg = common::tiny_config();
    let fc = "mtp.fc_embedding";
    cfg["quantization_config"]["quantized_layers"][fc] = serde_json::json!({"quant_algo": "MXFP8", "group_size": 32});
    let err = format!("{:#}", check_dir(&write_hybrid("hy-fc-bf16", &cfg, &tiny_tensors(), &common::TINY_VOCAB_SIZES)).unwrap_err());
    assert!(err.contains(&format!("{fc}.weight: expected F8_E4M3 [64, 64], found BF16 [64, 64]")) && err.contains(&format!("missing {fc}.weight_scale")), "{err}");
    let mut ts: Vec<_> = tiny_tensors().into_iter().filter(|t| t.0 != format!("{fc}.weight")).collect();
    ts.push((format!("{fc}.weight"), "F8_E4M3", vec![64, 64]));
    ts.push((format!("{fc}.weight_scale"), "U8", vec![64, 2]));
    let d = write_hybrid("hy-fc-mxfp8", &cfg, &ts, &common::TINY_VOCAB_SIZES);
    check_dir(&d).unwrap();
    let opts = plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 };
    let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
    let p = plan::build(&sparkforge::weights::spec_from_checkpoint(&dir, None, None).unwrap(), &GB10, &opts);
    let base = plan::build(&tiny_spec(), &GB10, &opts);
    assert!(((base.budget.weight_gib - p.budget.weight_gib) * (1u64 << 30) as f64 - 3968.0).abs() < 1e-3);
    assert!(!p.notes.iter().any(|n| n.contains(fc)), "{:#?}", p.notes);
}

#[test]
fn a_matrix_at_the_table_root_is_refused_on_resolve() {
    // fp:347106d738b2: a `weight` at the table's own path shares the module of the legitimate
    // `weight_scale_2`, so module checks alone accepted it unpriced. Only the scalar may sit there,
    // and a BF16 table with no scalar still plans under an NVFP4 override.
    let mut ts = tiny_tensors();
    let root = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    ts.push((format!("{root}.weight"), "U8", vec![4, 8]));
    let d = write_hybrid("hy-table-root", &common::tiny_config(), &ts, &common::TINY_VOCAB_SIZES);
    let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
    let err = format!("{:#}", sparkforge::weights::spec_from_checkpoint(&dir, None, None).unwrap_err());
    assert!(err.contains("may hold only its shared weight_scale_2") && err.contains(&format!("{root}.weight")), "{err}");
    let (cfg, ts) = bf16_table();
    let d = write_hybrid("hy-table-root-override", &cfg, &ts, &common::TINY_VOCAB_SIZES);
    let dir = sparkforge::weights::resolve_checkpoint_dir(&d).unwrap();
    sparkforge::weights::spec_from_checkpoint(&dir, Some(sparkforge::config::WeightFormat::Nvfp4), None).unwrap();
}

#[test]
fn an_oversized_expert_count_fails_on_the_live_experts() {
    // fp:d0f47806a572: u32::MAX declared experts against a 4-expert checkpoint fails from the live
    // members, naming the layer and the missing count, without building entries per declared expert.
    let mut s = tiny_spec();
    s.moe.as_mut().unwrap().num_experts = u32::MAX;
    let names: Vec<String> = tiny_tensors().into_iter().map(|t| t.0).collect();
    let t0 = std::time::Instant::now();
    let err = format!("{:#}", s.resolve_modules(names.iter().map(String::as_str)).unwrap_err());
    assert!(err.contains("declares 4294967295 routed experts") && err.contains("holds 4 of them under layers.0.mlp.experts (4294967291 missing)"), "{err}");
    assert!(t0.elapsed() < std::time::Duration::from_secs(5), "{:?}", t0.elapsed());
    assert!(s.resolved.is_none());
}

/// The tiny checkpoint's tensors as `reference::DenseWeights` (fused like the binding: q|k|v, gate|up).
fn dense_from(ck: &Checkpoint) -> sparkforge::reference::DenseWeights {
    use sparkforge::reference::{DenseWeights, LayerWeights};
    let read = |names: &[String]| -> Vec<u16> {
        let mut out = vec![];
        let mut buf = vec![];
        for n in names {
            ck.read_into(&ck.tensors[n], &mut buf).unwrap();
            out.extend(buf.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
        }
        out
    };
    let one = |n: &str| read(&[n.to_string()]);
    let layers = (0..2)
        .map(|l| {
            let p = format!("model.layers.{l}");
            LayerWeights {
                attn_norm: one(&format!("{p}.input_layernorm.weight")),
                qkv: read(&["q_proj", "k_proj", "v_proj"].map(|x| format!("{p}.self_attn.{x}.weight"))),
                o_proj: one(&format!("{p}.self_attn.o_proj.weight")),
                mlp_norm: one(&format!("{p}.post_attention_layernorm.weight")),
                gate_up: read(&["gate_proj", "up_proj"].map(|x| format!("{p}.mlp.{x}.weight"))),
                down: one(&format!("{p}.mlp.down_proj.weight")),
            }
        })
        .collect();
    DenseWeights { embed: one("model.embed_tokens.weight"), layers, final_norm: one("model.norm.weight"), lm_head: one("lm_head.weight") }
}

#[test]
fn the_reference_reads_a_bound_checkpoint_like_the_same_weights_in_memory() {
    use sparkforge::reference::{decode_step, CheckpointWeights, KvCache};
    let d = single("ref-checkpoint", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let m = spec(false);
    let b = bind(&ck, &m).unwrap();
    let cw = CheckpointWeights { ck: &ck, b: &b, m: &m };
    let dw = dense_from(&ck);
    let (mut c1, mut c2) = (KvCache::new(&m, 2, 3), KvCache::new(&m, 2, 3));
    for (pos, row) in [(0usize, 5usize), (1, 9), (2, 63)] {
        let a = decode_step(&m, &cw, &mut c1, row, pos).unwrap();
        let w = decode_step(&m, &dw, &mut c2, row, pos).unwrap();
        assert_eq!(a.after_attn, w.after_attn, "position {pos}");
        assert_eq!(a.after_mlp, w.after_mlp, "position {pos}");
        assert_eq!(a.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), w.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "position {pos}");
    }
    assert_eq!((c1.k, c1.v), (c2.k, c2.v));
}

#[test]
fn the_reference_refuses_a_model_the_checkpoint_was_not_bound_for() {
    use sparkforge::reference::{decode_step, CheckpointWeights, KvCache};
    let d = single("ref-mismatch", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let bound = spec(false);
    let b = bind(&ck, &bound).unwrap();
    // Same hidden size, layers, vocabulary and attention, twice the MLP width.
    let mut wide = spec(false);
    wide.intermediate = 64;
    let cw = CheckpointWeights { ck: &ck, b: &b, m: &bound };
    let mut cache = KvCache::new(&wide, 2, 2);
    let e = decode_step(&wide, &cw, &mut cache, 0, 0).err().expect("a different MLP width is refused").to_string();
    assert!(e.contains("GateUp(0)") && e.contains("64x16") && e.contains("128x16"), "{e}");
    assert!(cache.k.iter().chain(&cache.v).all(|c| c.iter().all(|&x| x == 0)), "nothing was written");
}

#[test]
fn a_checkpoint_changed_after_opening_fails_the_step_without_writing() {
    use sparkforge::reference::{decode_step, CheckpointWeights, KvCache, Rng};
    let d = single("ref-truncated", bf16(tensors(false)));
    let ck = Checkpoint::open(&d).unwrap();
    let m = spec(false);
    let b = bind(&ck, &m).unwrap();
    let cw = CheckpointWeights { ck: &ck, b: &b, m: &m };
    let mut cache = KvCache::new(&m, 2, 2);
    cache.fill_random(&m, 2, &mut Rng(29));
    let before = (cache.k.clone(), cache.v.clone());
    // The checkpoint notices a shard that changed after opening on its first read (the embedding
    // row), before the step writes anything. The restore after a later failure is tested in
    // tests/reference.rs.
    let f = d.join("model.safetensors");
    let len = std::fs::metadata(&f).unwrap().len();
    std::fs::OpenOptions::new().write(true).open(&f).unwrap().set_len(len - 64).unwrap();
    assert!(decode_step(&m, &cw, &mut cache, 3, 1).is_err(), "the truncated read fails");
    assert_eq!((cache.k.clone(), cache.v.clone()), before, "the cache is unchanged");
}

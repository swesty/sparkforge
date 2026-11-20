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

/// Creates `path` as a new directory only this user can enter. Non-recursive `mkdir` fails with
/// `AlreadyExists` on any existing entry, including a directory or symlink another local user
/// planted, so a fixture can never be written through someone else's path.
fn fresh_private_dir(path: &std::path::Path) -> std::io::Result<()> {
    let mut b = std::fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o700);
    b.create(path)
}

/// A fresh private directory per call. The name is unpredictable (time and a counter), and a
/// collision with any existing entry picks a new name instead of reusing it.
fn dir(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    loop {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!(
            "sparkforge-weights-{}-{nanos}-{}-{name}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        match fresh_private_dir(&d) {
            Ok(()) => return d,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("creating {}: {e}", d.display()),
        }
    }
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
    use sparkforge::weights::admit;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    const GIB: u64 = 1 << 30;
    let lock = dir("admission").join("admission.lock");
    // 20 GiB available; each load needs 14 GiB, so either fits alone but not both.
    let avail = Arc::new(AtomicU64::new(20 * GIB));
    let first = admit(&lock, 14 * GIB, || Some(avail.load(Ordering::SeqCst))).expect("first load fits");
    let second = {
        let (lock, avail) = (lock.clone(), avail.clone());
        std::thread::spawn(move || admit(&lock, 14 * GIB, || Some(avail.load(Ordering::SeqCst))).map(|_| ()))
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
        write_shard(&blobs.join("deadbeef"), &bf16(tensors(false)), 0);
        std::os::unix::fs::symlink("../../blobs/deadbeef", snap.join("model.safetensors")).unwrap();
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

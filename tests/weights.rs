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

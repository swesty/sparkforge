//! Hugging Face safetensors checkpoints: open, validate against a plan, and read tensor bytes.
//! Host-only (no CUDA); GPU binding lives in `device_weights` behind `--features cuda`.
//!
//! safetensors layout: an 8-byte little-endian header length N, then N bytes of JSON mapping each
//! tensor name to `{dtype, shape, data_offsets: [begin, end]}` (offsets relative to byte 8 + N),
//! then the tensor data, which the tensors must cover exactly (no overlaps, no holes).
//!
//! Every read is a fallible positioned read. Nothing is memory-mapped: a mapped file that another
//! process truncates turns a read into SIGBUS, while `pread` returns an error we can report.
use crate::config::{ModelSpec, WeightFormat};
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Largest header we accept (the reference safetensors implementation uses the same 100 MB cap).
const MAX_HEADER_BYTES: u64 = 100 << 20;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shard: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte range inside the shard file.
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

pub struct Shard {
    pub path: PathBuf,
    /// File length when opened; every read re-checks it so a rewritten shard is an error.
    pub len: u64,
    file: File,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub shards: Vec<Shard>,
    pub tensors: HashMap<String, TensorInfo>,
}

fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "BF16" | "F16" | "I16" | "U16" => 2,
        "F32" | "I32" | "U32" => 4,
        "F64" | "I64" | "U64" => 8,
        _ => return None,
    })
}

#[derive(serde::Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn pread(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, at)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], at + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
}

impl Checkpoint {
    /// Opens `dir/model.safetensors.index.json` if present, else `dir/model.safetensors`. With an
    /// index, every entry must name the shard whose header actually holds that tensor, and every
    /// tensor in a shard header must be listed in the index.
    pub fn open(dir: &Path) -> Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let index: Option<BTreeMap<String, String>> = if index_path.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            let idx: Index = serde_json::from_str(&std::fs::read_to_string(&index_path)?)
                .with_context(|| format!("parsing {}", index_path.display()))?;
            Some(idx.weight_map)
        } else {
            None
        };
        let shard_files: Vec<String> = match &index {
            Some(m) => {
                let mut f: Vec<String> = m.values().cloned().collect();
                f.sort();
                f.dedup();
                f
            }
            None => vec!["model.safetensors".into()],
        };
        let mut ck = Checkpoint { dir: dir.to_path_buf(), shards: vec![], tensors: HashMap::new() };
        for f in &shard_files {
            let path = dir.join(f);
            ensure!(path.exists(), "shard {} is missing", path.display());
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let len = file.metadata()?.len();
            let shard = ck.shards.len();
            for t in read_header(&path, &file, len, shard)? {
                if let Some(m) = &index {
                    match m.get(&t.name) {
                        Some(declared) if declared == f => {}
                        Some(declared) => bail!("index lists {} in {declared}, but it is in {}", t.name, path.display()),
                        None => bail!("{} holds {}, which the index does not list", path.display(), t.name),
                    }
                }
                if let Some(prev) = ck.tensors.get(&t.name) {
                    bail!("tensor {} appears in both {} and {}", t.name, ck.shards[prev.shard].path.display(), path.display());
                }
                ck.tensors.insert(t.name.clone(), t);
            }
            ck.shards.push(Shard { path, len, file });
        }
        if let Some(m) = &index {
            for (name, declared) in m {
                ensure!(ck.tensors.contains_key(name), "index lists {name} in {declared}, but that shard's header has no such tensor");
            }
        }
        Ok(ck)
    }

    /// Reads `buf.len()` bytes of tensor `t` starting `off` bytes into it. Fails, rather than
    /// faulting, if the range is outside the tensor or the shard changed size since it was opened.
    pub fn read_range(&self, t: &TensorInfo, off: u64, buf: &mut [u8]) -> Result<()> {
        let end = off.checked_add(buf.len() as u64).context("read range overflows")?;
        ensure!(end <= t.len(), "read of {}..{end} is outside tensor {} ({} bytes)", off, t.name, t.len());
        let shard = &self.shards[t.shard];
        let now = shard.file.metadata()?.len();
        ensure!(now == shard.len, "{} changed size since it was opened ({} -> {now} bytes)", shard.path.display(), shard.len);
        pread(&shard.file, buf, t.start + off).with_context(|| format!("reading {} from {}", t.name, shard.path.display()))
    }

    /// Reads a whole tensor into `buf` (resized to fit).
    pub fn read_into(&self, t: &TensorInfo, buf: &mut Vec<u8>) -> Result<()> {
        buf.resize(usize::try_from(t.len())?, 0);
        self.read_range(t, 0, buf)
    }
}

fn read_header(path: &Path, file: &File, file_len: u64, shard: usize) -> Result<Vec<TensorInfo>> {
    ensure!(file_len >= 8, "{}: shorter than a safetensors header", path.display());
    let mut n8 = [0u8; 8];
    pread(file, &mut n8, 0)?;
    let n = u64::from_le_bytes(n8);
    ensure!(n <= MAX_HEADER_BYTES, "{}: header length {n} exceeds the {MAX_HEADER_BYTES}-byte limit", path.display());
    let data_start = 8u64.checked_add(n).context("header length overflows")?;
    ensure!(data_start <= file_len, "{}: header length {n} runs past the end of the file", path.display());
    let mut header = vec![0u8; n as usize];
    pread(file, &mut header, 8)?;
    parse_header(path, &header, data_start, file_len, shard)
}

/// Validates every tensor entry against the data section `[data_start, file_len)` before forming
/// absolute offsets, then requires the non-empty ranges to tile the data section exactly.
fn parse_header(path: &Path, header: &[u8], data_start: u64, file_len: u64, shard: usize) -> Result<Vec<TensorInfo>> {
    let data_len = file_len - data_start;
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(header).with_context(|| format!("{}: parsing header JSON", path.display()))?;
    let mut out = vec![];
    for (name, v) in raw {
        if name == "__metadata__" {
            continue;
        }
        let t: RawTensor = serde_json::from_value(v).with_context(|| format!("{}: tensor {name}", path.display()))?;
        let [b, e] = t.data_offsets;
        ensure!(b <= e && e <= data_len, "{}: tensor {name} byte range {b}..{e} lies outside the file's {data_len}-byte data section", path.display());
        let elem = dtype_size(&t.dtype).with_context(|| format!("{}: tensor {name} has unknown dtype {}", path.display(), t.dtype))?;
        let want = t
            .shape
            .iter()
            .try_fold(elem, |acc, &d| acc.checked_mul(d))
            .with_context(|| format!("{}: tensor {name} shape {:?} overflows", path.display(), t.shape))?;
        ensure!(e - b == want, "{}: tensor {name} has {} bytes but {} {:?} needs {want}", path.display(), e - b, t.dtype, t.shape);
        let shape = t.shape.iter().map(|&d| usize::try_from(d)).collect::<std::result::Result<Vec<_>, _>>()?;
        out.push(TensorInfo { name, shard, dtype: t.dtype, shape, start: data_start + b, end: data_start + e });
    }
    // The tensors must cover the data section exactly: no overlaps (one tensor aliasing another)
    // and no unaccounted bytes. Empty tensors take no space and are skipped.
    let mut ranges: Vec<&TensorInfo> = out.iter().filter(|t| !t.is_empty()).collect();
    ranges.sort_by_key(|t| (t.start, t.end));
    let mut cursor = data_start;
    for t in &ranges {
        ensure!(t.start >= cursor, "{}: tensor {} ({}..{}) overlaps the tensor before it", path.display(), t.name, t.start - data_start, t.end - data_start);
        ensure!(t.start == cursor, "{}: {} unaccounted bytes before tensor {}", path.display(), t.start - cursor, t.name);
        cursor = t.end;
    }
    ensure!(cursor == file_len, "{}: {} unaccounted bytes after the last tensor", path.display(), file_len - cursor);
    Ok(out)
}

/// f64 reference for a bound linear's GEMV, `y = W x`, read from the checkpoint in bounded chunks
/// so peak host memory is about `threads * REF_CHUNK_BYTES`, independent of the tensor size.
pub const REF_CHUNK_BYTES: usize = 4 << 20;
pub const REF_THREADS: usize = 16;

pub fn reference_gemv(ck: &Checkpoint, l: &BoundLinear, x: &[f64]) -> Result<Vec<f64>> {
    ensure!(x.len() == l.k, "input has {} elements, linear needs {}", x.len(), l.k);
    let row_bytes = l.k * 2;
    // (tensor, first row within that tensor, output row) for every part, so threads can split rows.
    let mut parts = vec![];
    let mut out_row = 0;
    for name in &l.parts {
        let t = &ck.tensors[name];
        let rows = usize::try_from(t.len())? / row_bytes;
        parts.push((t, out_row, rows));
        out_row += rows;
    }
    ensure!(out_row == l.n, "{} has {out_row} rows, plan expects {}", l.key(), l.n);
    let mut y = vec![0f64; l.n];
    let per = l.n.div_ceil(REF_THREADS);
    let rows_per_read = (REF_CHUNK_BYTES / row_bytes).max(1);
    std::thread::scope(|sc| -> Result<()> {
        let handles: Vec<_> = y
            .chunks_mut(per)
            .enumerate()
            .map(|(ci, out)| {
                let parts = &parts;
                sc.spawn(move || -> Result<()> {
                    let first = ci * per;
                    let mut buf = vec![0u8; rows_per_read * row_bytes];
                    let mut r = 0;
                    while r < out.len() {
                        let global = first + r;
                        let &(t, base, rows) = parts.iter().find(|(_, b, n)| global >= *b && global < b + n).unwrap();
                        let local = global - base;
                        let take = rows_per_read.min(out.len() - r).min(rows - local);
                        let bytes = &mut buf[..take * row_bytes];
                        ck.read_range(t, (local * row_bytes) as u64, bytes)?;
                        for (j, row) in bytes.chunks_exact(row_bytes).enumerate() {
                            out[r + j] = row
                                .chunks_exact(2)
                                .zip(x)
                                .map(|(c, &xv)| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64 * xv)
                                .sum();
                        }
                        r += take;
                    }
                    Ok(())
                })
            })
            .collect();
        for h in handles {
            h.join().expect("reference thread panicked")?;
        }
        Ok(())
    })?;
    Ok(y)
}

/// One planned linear: the checkpoint tensors that make it, stacked by rows in this order.
#[derive(Debug, Clone)]
pub struct BoundLinear {
    /// Plan linear name (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `lm_head`).
    pub linear: String,
    pub layer: Option<u32>,
    pub parts: Vec<String>,
    pub n: usize,
    pub k: usize,
}

impl BoundLinear {
    pub fn fused(&self) -> bool {
        self.parts.len() > 1
    }
    pub fn key(&self) -> String {
        match self.layer {
            Some(l) => format!("layers.{l}.{}", self.linear),
            None => self.linear.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Binding {
    pub linears: Vec<BoundLinear>,
    /// Non-linear tensors the model needs (norms, embedding, optional q/k norms and biases).
    pub others: Vec<String>,
    /// Tensors in the checkpoint that the plan does not use.
    pub unexpected: Vec<String>,
}

/// Validates the checkpoint against the spec and returns which tensors make each linear.
pub fn bind(ck: &Checkpoint, m: &ModelSpec) -> Result<Binding> {
    if m.moe.is_some() {
        bail!("MoE weight loading is not supported yet ({})", m.arch);
    }
    if m.weights != WeightFormat::Bf16 {
        bail!("weight format {:?} is not supported yet; only BF16 checkpoints load today", m.weights);
    }
    let (h, q, kv, inter) = (m.hidden as usize, m.q_dim() as usize, m.kv_dim() as usize, m.intermediate as usize);
    let mut need: Vec<(String, Vec<usize>)> = vec![];
    let mut linears = vec![];
    let mut others = vec![];
    let mut lin = |linear: &str, layer: Option<u32>, parts: Vec<(String, usize)>, k: usize, need: &mut Vec<(String, Vec<usize>)>| {
        for (name, rows) in &parts {
            need.push((name.clone(), vec![*rows, k]));
        }
        let n = parts.iter().map(|p| p.1).sum();
        linears.push(BoundLinear { linear: linear.into(), layer, parts: parts.into_iter().map(|p| p.0).collect(), n, k });
    };
    for l in 0..m.layers {
        let p = format!("model.layers.{l}");
        let a = format!("{p}.self_attn");
        let f = format!("{p}.mlp");
        lin("qkv_proj", Some(l), vec![(format!("{a}.q_proj.weight"), q), (format!("{a}.k_proj.weight"), kv), (format!("{a}.v_proj.weight"), kv)], h, &mut need);
        lin("o_proj", Some(l), vec![(format!("{a}.o_proj.weight"), h)], q, &mut need);
        lin("gate_up_proj", Some(l), vec![(format!("{f}.gate_proj.weight"), inter), (format!("{f}.up_proj.weight"), inter)], h, &mut need);
        lin("down_proj", Some(l), vec![(format!("{f}.down_proj.weight"), h)], inter, &mut need);
        let mut add = |name: String, shape: Vec<usize>| {
            others.push(name.clone());
            need.push((name, shape));
        };
        add(format!("{p}.input_layernorm.weight"), vec![h]);
        add(format!("{p}.post_attention_layernorm.weight"), vec![h]);
        if m.qk_norm {
            add(format!("{a}.q_norm.weight"), vec![m.head_dim as usize]);
            add(format!("{a}.k_norm.weight"), vec![m.head_dim as usize]);
        }
        if m.qkv_bias {
            add(format!("{a}.q_proj.bias"), vec![q]);
            add(format!("{a}.k_proj.bias"), vec![kv]);
            add(format!("{a}.v_proj.bias"), vec![kv]);
        }
    }
    let embed = "model.embed_tokens.weight".to_string();
    others.push(embed.clone());
    need.push((embed.clone(), vec![m.vocab as usize, h]));
    others.push("model.norm.weight".into());
    need.push(("model.norm.weight".into(), vec![h]));
    let head = if m.tie_embeddings { embed } else { "lm_head.weight".to_string() };
    if !m.tie_embeddings {
        need.push((head.clone(), vec![m.vocab as usize, h]));
    }
    linears.push(BoundLinear { linear: "lm_head".into(), layer: None, parts: vec![head], n: m.vocab as usize, k: h });

    let mut problems = vec![];
    for (name, shape) in &need {
        match ck.tensors.get(name) {
            None => problems.push(format!("missing {name} (expected BF16 {shape:?})")),
            Some(t) if &t.shape != shape => problems.push(format!("{name}: expected shape {shape:?}, found {:?}", t.shape)),
            Some(t) if t.dtype != "BF16" => problems.push(format!("{name}: expected dtype BF16, found {}", t.dtype)),
            Some(_) => {}
        }
    }
    if !problems.is_empty() {
        bail!("checkpoint {} does not match the plan:\n  {}", ck.dir.display(), problems.join("\n  "));
    }
    let used: std::collections::HashSet<&str> = need.iter().map(|(n, _)| n.as_str()).collect();
    let mut unexpected: Vec<String> = ck.tensors.keys().filter(|k| !used.contains(k.as_str())).cloned().collect();
    unexpected.sort();
    Ok(Binding { linears, others, unexpected })
}

/// Bytes `MemAvailable` must keep after a load (design decision 5).
pub const HEADROOM_BYTES: u64 = 4 << 30;

/// Host staging buffer for uploads: tensors are copied to the GPU in pieces of at most this size.
pub const STAGING_BYTES: u64 = 64 << 20;

/// Host memory a load needs beyond the copied weights: the staging buffer plus, when verifying,
/// the reference's per-thread read buffers and its f64 output (one value per output row).
pub fn load_overhead_bytes(b: &Binding, verifying: bool) -> u64 {
    let refs = if verifying {
        let max_n = b.linears.iter().map(|l| l.n).max().unwrap_or(0) as u64;
        (REF_THREADS * REF_CHUNK_BYTES) as u64 + max_n * 8
    } else {
        0
    };
    STAGING_BYTES + refs
}

/// Refuses a load whose copied bytes would leave less than `HEADROOM_BYTES` available.
pub fn check_headroom(copy_bytes: u64, mem_available: u64) -> Result<()> {
    ensure!(
        copy_bytes + HEADROOM_BYTES <= mem_available,
        "refusing to copy {:.2} GiB: MemAvailable is {:.2} GiB and {:.0} GiB must stay free (GB10 pinned/device memory bypasses the OOM killer)",
        copy_bytes as f64 / (1u64 << 30) as f64,
        mem_available as f64 / (1u64 << 30) as f64,
        HEADROOM_BYTES as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

/// `MemAvailable` from /proc/meminfo in bytes (Linux only).
pub fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

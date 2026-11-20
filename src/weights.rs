//! Hugging Face safetensors checkpoints: open, memory-map, and validate against a plan.
//! Host-only (no CUDA); GPU binding lives in `device_weights` behind `--features cuda`.
//!
//! safetensors layout: an 8-byte little-endian header length N, then N bytes of JSON mapping each
//! tensor name to `{dtype, shape, data_offsets: [begin, end]}` (offsets relative to byte 8 + N),
//! then the tensor data.
use crate::config::{ModelSpec, WeightFormat};
use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shard: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte range inside the shard's mapping.
    pub start: usize,
    pub end: usize,
}

pub struct Shard {
    pub path: PathBuf,
    pub map: Mmap,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub shards: Vec<Shard>,
    pub tensors: HashMap<String, TensorInfo>,
}

fn dtype_size(dtype: &str) -> Option<usize> {
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
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl Checkpoint {
    /// Opens `dir/model.safetensors.index.json` if present, else `dir/model.safetensors`.
    pub fn open(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let shard_files: Vec<String> = if index.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            let idx: Index = serde_json::from_str(&std::fs::read_to_string(&index)?)
                .with_context(|| format!("parsing {}", index.display()))?;
            let mut files: Vec<String> = idx.weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            files
        } else {
            vec!["model.safetensors".into()]
        };
        let mut ck = Checkpoint { dir: dir.to_path_buf(), shards: vec![], tensors: HashMap::new() };
        for f in shard_files {
            let path = dir.join(&f);
            ensure!(path.exists(), "shard {} is missing", path.display());
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: read-only mapping of a checkpoint file nobody should be rewriting under us.
            let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
            let shard = ck.shards.len();
            for t in parse_header(&path, &map, shard)? {
                if let Some(prev) = ck.tensors.get(&t.name) {
                    bail!("tensor {} appears in both {} and {}", t.name, ck.shards[prev.shard].path.display(), path.display());
                }
                ck.tensors.insert(t.name.clone(), t);
            }
            ck.shards.push(Shard { path, map });
        }
        Ok(ck)
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.shards[t.shard].map[t.start..t.end]
    }
}

fn parse_header(path: &Path, map: &[u8], shard: usize) -> Result<Vec<TensorInfo>> {
    ensure!(map.len() >= 8, "{}: shorter than a safetensors header", path.display());
    let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
    ensure!(8 + n <= map.len(), "{}: header length {} runs past the end of the file", path.display(), n);
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&map[8..8 + n])
        .with_context(|| format!("{}: parsing header JSON", path.display()))?;
    let data = 8 + n;
    let mut out = vec![];
    for (name, v) in raw {
        if name == "__metadata__" {
            continue;
        }
        let t: RawTensor = serde_json::from_value(v).with_context(|| format!("{}: tensor {name}", path.display()))?;
        let [b, e] = t.data_offsets;
        let (start, end) = (data + b, data + e);
        ensure!(b <= e && end <= map.len(), "{}: tensor {name} byte range {b}..{e} lies outside the file", path.display());
        let elem = dtype_size(&t.dtype).with_context(|| format!("{}: tensor {name} has unknown dtype {}", path.display(), t.dtype))?;
        let want = t.shape.iter().product::<usize>() * elem;
        ensure!(e - b == want, "{}: tensor {name} has {} bytes but {} {:?} needs {want}", path.display(), e - b, t.dtype, t.shape);
        out.push(TensorInfo { name, shard, dtype: t.dtype, shape: t.shape, start, end });
    }
    Ok(out)
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

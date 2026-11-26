//! One dense decode step on the GPU, wired from the generated kernels in the order the CPU
//! reference (`reference::decode_step`) runs: per layer, RMSNorm, the fused qkv GEMV, RoPE with
//! the K/V append, split attention and its combine, the o_proj GEMV and a residual add, then
//! RMSNorm, the fused gate/up GEMV, SiLU-and-multiply, the down GEMV and a residual add; after the
//! last layer, the final RMSNorm and `lm_head`. Plain launches on one stream, no CUDA Graph.
use crate::codegen::{Generated, KernelEntry};
use crate::config::ModelSpec;
use crate::reference::{DenseWeights, KvCache};
use crate::runtime::Engine;
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::Arc;

/// Every kernel a dense decode step launches.
pub const KERNELS: [&str; 11] = [
    "rmsnorm", "rope_kv_append", "attn_decode_split", "attn_decode_combine", "silu_mul", "residual_add",
    "gemv_qkv_proj", "gemv_o_proj", "gemv_gate_up_proj", "gemv_down_proj", "gemv_lm_head",
];

struct Kernel<'a> {
    entry: &'a KernelEntry,
    func: &'a CudaFunction,
}

impl Kernel<'_> {
    fn cfg(&self) -> LaunchConfig {
        let g = self.entry.grid;
        LaunchConfig { grid_dim: (g[0], g[1], g[2]), block_dim: (self.entry.block, 1, 1), shared_mem_bytes: self.entry.dyn_smem }
    }
}

pub struct Kernels<'a>(HashMap<&'static str, Kernel<'a>>);

impl<'a> Kernels<'a> {
    /// Looks up every kernel in `KERNELS`; fails naming the ones that were not generated.
    pub fn new(g: &'a Generated, e: &'a Engine) -> Result<Self> {
        let mut map = HashMap::new();
        let mut missing = vec![];
        for name in KERNELS {
            match (g.kernels.iter().find(|k| k.entry == name && k.status == "generated"), e.funcs.get(name)) {
                (Some(entry), Some(func)) => {
                    map.insert(name, Kernel { entry, func });
                }
                _ => missing.push(name),
            }
        }
        ensure!(missing.is_empty(), "the dense decode step needs kernels that were not generated: {}", missing.join(", "));
        Ok(Self(map))
    }
    fn get(&self, name: &str) -> &Kernel<'a> {
        &self.0[name]
    }
}

pub struct DeviceLayer {
    pub attn_norm: CudaSlice<u16>,
    pub qkv: CudaSlice<u16>,
    pub o_proj: CudaSlice<u16>,
    pub mlp_norm: CudaSlice<u16>,
    pub gate_up: CudaSlice<u16>,
    pub down: CudaSlice<u16>,
    pub k_cache: CudaSlice<u16>,
    pub v_cache: CudaSlice<u16>,
}

/// Weights and K/V caches in GPU memory.
pub struct DeviceModel {
    pub embed: CudaSlice<u16>,
    pub layers: Vec<DeviceLayer>,
    pub final_norm: CudaSlice<u16>,
    pub lm_head: CudaSlice<u16>,
    pub embed_rows: usize,
    pub capacity: usize,
}

impl DeviceModel {
    /// Copies `w` and the current contents of `cache` (so a prefilled prefix carries over).
    pub fn upload(s: &Arc<CudaStream>, m: &ModelSpec, w: &DenseWeights, cache: &KvCache) -> Result<Self> {
        ensure!(cache.k.len() == w.layers.len(), "the cache has {} layers, the weights {}", cache.k.len(), w.layers.len());
        let layers = w
            .layers
            .iter()
            .zip(cache.k.iter().zip(&cache.v))
            .map(|(l, (k, v))| {
                Ok(DeviceLayer {
                    attn_norm: s.memcpy_stod(&l.attn_norm)?,
                    qkv: s.memcpy_stod(&l.qkv)?,
                    o_proj: s.memcpy_stod(&l.o_proj)?,
                    mlp_norm: s.memcpy_stod(&l.mlp_norm)?,
                    gate_up: s.memcpy_stod(&l.gate_up)?,
                    down: s.memcpy_stod(&l.down)?,
                    k_cache: s.memcpy_stod(k)?,
                    v_cache: s.memcpy_stod(v)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed: s.memcpy_stod(&w.embed)?,
            layers,
            final_norm: s.memcpy_stod(&w.final_norm)?,
            lm_head: s.memcpy_stod(&w.lm_head)?,
            embed_rows: w.embed.len() / m.hidden as usize,
            capacity: cache.capacity,
        })
    }
}

/// Activations of one step, reused across steps.
pub struct Scratch {
    x: CudaSlice<u16>,
    xn: CudaSlice<u16>,
    qkv: CudaSlice<u16>,
    att: CudaSlice<u16>,
    o: CudaSlice<u16>,
    gate_up: CudaSlice<u16>,
    act: CudaSlice<u16>,
    down: CudaSlice<u16>,
    logits: CudaSlice<u16>,
    part_ml: CudaSlice<f32>,
    part_acc: CudaSlice<f32>,
}

impl Scratch {
    pub fn new(s: &Arc<CudaStream>, m: &ModelSpec, splits: usize) -> Result<Self> {
        let (h, inter, q, qkv) = (m.hidden as usize, m.intermediate as usize, m.q_dim() as usize, m.qkv_dim() as usize);
        let heads = m.heads as usize;
        Ok(Self {
            x: s.alloc_zeros(h)?,
            xn: s.alloc_zeros(h)?,
            qkv: s.alloc_zeros(qkv)?,
            att: s.alloc_zeros(q)?,
            o: s.alloc_zeros(h)?,
            gate_up: s.alloc_zeros(2 * inter)?,
            act: s.alloc_zeros(inter)?,
            down: s.alloc_zeros(h)?,
            logits: s.alloc_zeros(m.vocab as usize)?,
            part_ml: s.alloc_zeros(heads * splits * 2)?,
            part_acc: s.alloc_zeros(heads * splits * m.head_dim as usize)?,
        })
    }
}

/// The GPU's hidden state after each layer's two residual adds, and its bf16 logits.
pub struct GpuTrace {
    pub after_attn: Vec<Vec<u16>>,
    pub after_mlp: Vec<Vec<u16>>,
    pub logits: Vec<u16>,
}

/// Runs one decode step for embedding row `row` at position `pos`. With `trace`, copies the hidden
/// state back after every residual add (synchronizing the stream each time) and returns it.
#[allow(clippy::too_many_arguments)]
pub fn step(s: &Arc<CudaStream>, k: &Kernels, m: &ModelSpec, dm: &mut DeviceModel, sc: &mut Scratch, row: usize, pos: usize, trace: bool) -> Result<Option<GpuTrace>> {
    let h = m.hidden as usize;
    ensure!(pos < dm.capacity, "position {pos} is past the cache capacity {}", dm.capacity);
    ensure!(row < dm.embed_rows, "embedding row {row} is past the {} uploaded rows", dm.embed_rows);
    let (eps, scale) = (m.rms_eps as f32, 1.0 / (m.head_dim as f32).sqrt());
    let (p32, len32) = (pos as i32, pos as i32 + 1);
    let (norm, rope, split, combine) = (k.get("rmsnorm"), k.get("rope_kv_append"), k.get("attn_decode_split"), k.get("attn_decode_combine"));
    let (silu, add) = (k.get("silu_mul"), k.get("residual_add"));
    let (g_qkv, g_o, g_gu, g_down, g_lm) = (k.get("gemv_qkv_proj"), k.get("gemv_o_proj"), k.get("gemv_gate_up_proj"), k.get("gemv_down_proj"), k.get("gemv_lm_head"));
    // One row per launch: the generated rmsnorm reads row blockIdx.x.
    let one = |kn: &Kernel| LaunchConfig { grid_dim: (1, 1, 1), ..kn.cfg() };
    s.memcpy_dtod(&dm.embed.slice(row * h..(row + 1) * h), &mut sc.x).context("embedding row")?;
    let mut t = trace.then(|| GpuTrace { after_attn: vec![], after_mlp: vec![], logits: vec![] });
    for l in dm.layers.iter_mut() {
        unsafe {
            s.launch_builder(norm.func).arg(&sc.x).arg(&l.attn_norm).arg(&mut sc.xn).arg(&eps).launch(one(norm))?;
            s.launch_builder(g_qkv.func).arg(&l.qkv).arg(&sc.xn).arg(&mut sc.qkv).launch(g_qkv.cfg())?;
            s.launch_builder(rope.func).arg(&mut sc.qkv).arg(&mut l.k_cache).arg(&mut l.v_cache).arg(&p32).launch(rope.cfg())?;
            // q is the first heads * head_dim elements of qkv.
            s.launch_builder(split.func).arg(&sc.qkv).arg(&l.k_cache).arg(&l.v_cache).arg(&mut sc.part_ml).arg(&mut sc.part_acc)
                .arg(&len32).arg(&scale).launch(split.cfg())?;
            s.launch_builder(combine.func).arg(&sc.part_ml).arg(&sc.part_acc).arg(&mut sc.att).launch(combine.cfg())?;
            s.launch_builder(g_o.func).arg(&l.o_proj).arg(&sc.att).arg(&mut sc.o).launch(g_o.cfg())?;
            s.launch_builder(add.func).arg(&mut sc.x).arg(&sc.o).launch(add.cfg())?;
        }
        if let Some(t) = t.as_mut() {
            t.after_attn.push(s.memcpy_dtov(&sc.x)?);
        }
        unsafe {
            s.launch_builder(norm.func).arg(&sc.x).arg(&l.mlp_norm).arg(&mut sc.xn).arg(&eps).launch(one(norm))?;
            s.launch_builder(g_gu.func).arg(&l.gate_up).arg(&sc.xn).arg(&mut sc.gate_up).launch(g_gu.cfg())?;
            s.launch_builder(silu.func).arg(&sc.gate_up).arg(&mut sc.act).launch(silu.cfg())?;
            s.launch_builder(g_down.func).arg(&l.down).arg(&sc.act).arg(&mut sc.down).launch(g_down.cfg())?;
            s.launch_builder(add.func).arg(&mut sc.x).arg(&sc.down).launch(add.cfg())?;
        }
        if let Some(t) = t.as_mut() {
            t.after_mlp.push(s.memcpy_dtov(&sc.x)?);
        }
    }
    unsafe {
        s.launch_builder(norm.func).arg(&sc.x).arg(&dm.final_norm).arg(&mut sc.xn).arg(&eps).launch(one(norm))?;
        s.launch_builder(g_lm.func).arg(&dm.lm_head).arg(&sc.xn).arg(&mut sc.logits).launch(g_lm.cfg())?;
    }
    if let Some(t) = t.as_mut() {
        t.logits = s.memcpy_dtov(&sc.logits)?;
    }
    Ok(t)
}

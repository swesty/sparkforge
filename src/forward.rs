//! Dense decode on the GPU, wired from the generated kernels in the order the CPU reference
//! (`reference::decode_step`) runs. A step reads its token and position from device memory:
//! `embed_gather` looks up the token, then per layer RMSNorm, the fused qkv GEMV, RoPE with the
//! K/V append at the position, split attention and its combine, o_proj and a residual add, RMSNorm,
//! the fused gate/up GEMV, SiLU-and-multiply, down and a residual add; after the last layer, the
//! final RMSNorm, `lm_head`, and `argmax_advance`, which writes the greedy next token, records it
//! and increments the position. Every launch is the same each step, so one captured CUDA Graph
//! replays the whole step with no host work between tokens.
use crate::codegen::{Generated, KernelEntry};
use crate::config::ModelSpec;
use crate::reference::{DenseWeights, KvCache};
use crate::runtime::Engine;
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::Arc;

/// Every kernel a dense decode step launches.
pub const KERNELS: [&str; 13] = [
    "embed_gather", "rmsnorm", "rope_kv_append", "attn_decode_split", "attn_decode_combine", "silu_mul", "residual_add",
    "gemv_qkv_proj", "gemv_o_proj", "gemv_gate_up_proj", "gemv_down_proj", "gemv_lm_head", "argmax_advance",
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

/// Device addresses of one layer's weights, row-major `[n][k]` bf16, fused as the binding packs
/// them (q|k|v, gate|up).
pub struct LayerPtrs {
    pub attn_norm: u64,
    pub qkv: u64,
    pub o_proj: u64,
    pub mlp_norm: u64,
    pub gate_up: u64,
    pub down: u64,
}

/// Device addresses of every weight a step reads; whoever filled it keeps the memory alive.
pub struct WeightPtrs {
    pub embed: u64,
    pub embed_rows: usize,
    pub final_norm: u64,
    pub lm_head: u64,
    pub layers: Vec<LayerPtrs>,
}

fn addr<T>(s: &Arc<CudaStream>, x: &CudaSlice<T>) -> u64 {
    let (p, _sync) = x.device_ptr(s);
    p
}

/// Random weights (`reference::DenseWeights`) copied into GPU memory.
pub struct UploadedWeights {
    _owned: Vec<CudaSlice<u16>>,
    pub ptrs: WeightPtrs,
}

impl UploadedWeights {
    /// Checks every length against `m` before copying anything.
    pub fn new(s: &Arc<CudaStream>, m: &ModelSpec, w: &DenseWeights) -> Result<Self> {
        crate::reference::check_weights(m, w)?;
        let mut owned = vec![];
        let mut put = |v: &[u16]| -> Result<u64> {
            let d = s.memcpy_stod(v)?;
            let p = addr(s, &d);
            owned.push(d);
            Ok(p)
        };
        let mut layers = vec![];
        for l in &w.layers {
            layers.push(LayerPtrs {
                attn_norm: put(&l.attn_norm)?,
                qkv: put(&l.qkv)?,
                o_proj: put(&l.o_proj)?,
                mlp_norm: put(&l.mlp_norm)?,
                gate_up: put(&l.gate_up)?,
                down: put(&l.down)?,
            });
        }
        let ptrs = WeightPtrs {
            embed: put(&w.embed)?,
            embed_rows: w.embed.len() / m.hidden as usize,
            final_norm: put(&w.final_norm)?,
            lm_head: put(&w.lm_head)?,
            layers,
        };
        Ok(Self { _owned: owned, ptrs })
    }
}

/// The GPU's hidden state after each layer's two residual adds, and its bf16 logits.
pub struct GpuTrace {
    pub after_attn: Vec<Vec<u16>>,
    pub after_mlp: Vec<Vec<u16>>,
    pub logits: Vec<u16>,
}

/// One sequence's decode state on the GPU: KV caches, activations, the device-side token and
/// position, and (once captured) the step graph. The graph holds raw addresses into these buffers,
/// so it lives and dies with them; it is declared first so it is destroyed first.
pub struct Runtime<'a> {
    graph: Option<CudaGraph>,
    s: Arc<CudaStream>,
    k: Kernels<'a>,
    m: &'a ModelSpec,
    w: &'a WeightPtrs,
    capacity: usize,
    /// Host copy of the device position: every step is checked against `capacity` before launch.
    pos: usize,
    k_cache: Vec<CudaSlice<u16>>,
    v_cache: Vec<CudaSlice<u16>>,
    d_pos: CudaSlice<i32>,
    d_token: CudaSlice<i32>,
    /// `out[p]` is the greedy token produced by the step at position `p`.
    out: CudaSlice<i32>,
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

/// Bytes a `Runtime` allocates besides the weights: KV caches, activations and token state.
pub fn runtime_bytes(m: &ModelSpec, splits: usize, capacity: usize) -> u64 {
    let (h, inter, q, qkv, kv) = (m.hidden as u64, m.intermediate as u64, m.q_dim() as u64, m.qkv_dim() as u64, m.kv_dim() as u64);
    let kv_bytes = 2 * m.layers as u64 * capacity as u64 * kv * 2;
    let act = (4 * h + qkv + q + 3 * inter + m.vocab as u64) * 2;
    let part = m.heads as u64 * splits as u64 * (2 + m.head_dim as u64) * 4;
    kv_bytes + act + part + (capacity as u64 + 2) * 4
}

impl<'a> Runtime<'a> {
    /// Zeroed KV caches of `capacity` positions for every layer of `w`.
    pub fn new(s: &Arc<CudaStream>, k: Kernels<'a>, m: &'a ModelSpec, w: &'a WeightPtrs, splits: usize, capacity: usize) -> Result<Self> {
        let n = capacity * m.kv_dim() as usize;
        let layers = w.layers.len();
        let k_cache = (0..layers).map(|_| s.alloc_zeros::<u16>(n)).collect::<Result<Vec<_>, _>>()?;
        let v_cache = (0..layers).map(|_| s.alloc_zeros::<u16>(n)).collect::<Result<Vec<_>, _>>()?;
        Self::with_caches(s, k, m, w, splits, capacity, k_cache, v_cache)
    }

    /// KV caches copied from a host cache (for example a random prefix), checked first.
    pub fn from_host(s: &Arc<CudaStream>, k: Kernels<'a>, m: &'a ModelSpec, w: &'a WeightPtrs, splits: usize, cache: &KvCache) -> Result<Self> {
        crate::reference::check_cache(m, w.layers.len(), cache)?;
        let k_cache = cache.k.iter().map(|c| s.memcpy_stod(c)).collect::<Result<Vec<_>, _>>()?;
        let v_cache = cache.v.iter().map(|c| s.memcpy_stod(c)).collect::<Result<Vec<_>, _>>()?;
        Self::with_caches(s, k, m, w, splits, cache.capacity, k_cache, v_cache)
    }

    #[allow(clippy::too_many_arguments)]
    fn with_caches(s: &Arc<CudaStream>, k: Kernels<'a>, m: &'a ModelSpec, w: &'a WeightPtrs, splits: usize, capacity: usize,
        k_cache: Vec<CudaSlice<u16>>, v_cache: Vec<CudaSlice<u16>>) -> Result<Self> {
        let (h, inter, q, qkv) = (m.hidden as usize, m.intermediate as usize, m.q_dim() as usize, m.qkv_dim() as usize);
        let heads = m.heads as usize;
        ensure!(capacity > 0, "a KV cache of zero positions");
        ensure!(!w.layers.is_empty(), "no layers to run");
        Ok(Self {
            graph: None,
            s: s.clone(),
            k,
            m,
            w,
            capacity,
            pos: 0,
            k_cache,
            v_cache,
            d_pos: s.alloc_zeros(1)?,
            d_token: s.alloc_zeros(1)?,
            out: s.alloc_zeros(capacity)?,
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

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// The position the next step decodes at.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Sets the next step's input token and position.
    pub fn set(&mut self, pos: usize, token: usize) -> Result<()> {
        ensure!(pos < self.capacity, "position {pos} is past the cache capacity {}", self.capacity);
        ensure!(token < self.w.embed_rows, "token {token} is past the {} embedding rows", self.w.embed_rows);
        self.s.memcpy_htod(&[pos as i32], &mut self.d_pos)?;
        self.s.memcpy_htod(&[token as i32], &mut self.d_token)?;
        self.pos = pos;
        Ok(())
    }

    /// Captures one step as a CUDA Graph; later `step` calls replay it. The stream must not be the
    /// legacy default stream, which cannot be captured.
    pub fn capture(&mut self) -> Result<()> {
        self.graph = None;
        self.s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).context("begin graph capture")?;
        let r = self.enqueue(None);
        let g = self.s.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        r?;
        self.graph = Some(g.context("end graph capture")?.context("graph capture recorded nothing")?);
        Ok(())
    }

    /// Runs one step (the graph once captured), then the position advances by one.
    pub fn step(&mut self) -> Result<()> {
        ensure!(self.pos < self.capacity, "position {} is past the cache capacity {}", self.pos, self.capacity);
        match &self.graph {
            Some(g) => g.launch()?,
            None => self.enqueue(None)?,
        }
        self.pos += 1;
        Ok(())
    }

    /// Runs one step with plain launches, copying the hidden state back after every residual add
    /// (synchronizing each time).
    pub fn step_traced(&mut self) -> Result<GpuTrace> {
        ensure!(self.pos < self.capacity, "position {} is past the cache capacity {}", self.pos, self.capacity);
        let mut t = GpuTrace { after_attn: vec![], after_mlp: vec![], logits: vec![] };
        self.enqueue(Some(&mut t))?;
        t.logits = self.s.memcpy_dtov(&self.logits)?;
        self.pos += 1;
        Ok(t)
    }

    /// The last step's logits.
    pub fn logits(&self) -> Result<Vec<u16>> {
        Ok(self.s.memcpy_dtov(&self.logits)?)
    }

    /// Greedy tokens produced by the steps at positions `from..to`.
    pub fn tokens(&self, from: usize, to: usize) -> Result<Vec<u32>> {
        ensure!(from <= to && to <= self.capacity, "positions {from}..{to} are outside the cache capacity {}", self.capacity);
        Ok(self.s.memcpy_dtov(&self.out.slice(from..to))?.into_iter().map(|t| t as u32).collect())
    }

    fn enqueue(&mut self, mut trace: Option<&mut GpuTrace>) -> Result<()> {
        let s = &self.s;
        let k = &self.k;
        let w = self.w;
        let (eps, scale) = (self.m.rms_eps as f32, 1.0 / (self.m.head_dim as f32).sqrt());
        let (embed, norm, rope, split, combine) = (k.get("embed_gather"), k.get("rmsnorm"), k.get("rope_kv_append"), k.get("attn_decode_split"), k.get("attn_decode_combine"));
        let (silu, add, argmax) = (k.get("silu_mul"), k.get("residual_add"), k.get("argmax_advance"));
        let (g_qkv, g_o, g_gu, g_down, g_lm) = (k.get("gemv_qkv_proj"), k.get("gemv_o_proj"), k.get("gemv_gate_up_proj"), k.get("gemv_down_proj"), k.get("gemv_lm_head"));
        // One row per launch: the generated rmsnorm reads row blockIdx.x.
        let one = |kn: &Kernel| LaunchConfig { grid_dim: (1, 1, 1), ..kn.cfg() };
        unsafe { s.launch_builder(embed.func).arg(&w.embed).arg(&self.d_token).arg(&mut self.x).launch(embed.cfg())? };
        for (l, lw) in w.layers.iter().enumerate() {
            unsafe {
                s.launch_builder(norm.func).arg(&self.x).arg(&lw.attn_norm).arg(&mut self.xn).arg(&eps).launch(one(norm))?;
                s.launch_builder(g_qkv.func).arg(&lw.qkv).arg(&self.xn).arg(&mut self.qkv).launch(g_qkv.cfg())?;
                s.launch_builder(rope.func).arg(&mut self.qkv).arg(&mut self.k_cache[l]).arg(&mut self.v_cache[l]).arg(&self.d_pos).launch(rope.cfg())?;
                // q is the first heads * head_dim elements of qkv.
                s.launch_builder(split.func).arg(&self.qkv).arg(&self.k_cache[l]).arg(&self.v_cache[l]).arg(&mut self.part_ml).arg(&mut self.part_acc)
                    .arg(&self.d_pos).arg(&scale).launch(split.cfg())?;
                s.launch_builder(combine.func).arg(&self.part_ml).arg(&self.part_acc).arg(&mut self.att).launch(combine.cfg())?;
                s.launch_builder(g_o.func).arg(&lw.o_proj).arg(&self.att).arg(&mut self.o).launch(g_o.cfg())?;
                s.launch_builder(add.func).arg(&mut self.x).arg(&self.o).launch(add.cfg())?;
            }
            if let Some(t) = trace.as_deref_mut() {
                t.after_attn.push(s.memcpy_dtov(&self.x)?);
            }
            unsafe {
                s.launch_builder(norm.func).arg(&self.x).arg(&lw.mlp_norm).arg(&mut self.xn).arg(&eps).launch(one(norm))?;
                s.launch_builder(g_gu.func).arg(&lw.gate_up).arg(&self.xn).arg(&mut self.gate_up).launch(g_gu.cfg())?;
                s.launch_builder(silu.func).arg(&self.gate_up).arg(&mut self.act).launch(silu.cfg())?;
                s.launch_builder(g_down.func).arg(&lw.down).arg(&self.act).arg(&mut self.down).launch(g_down.cfg())?;
                s.launch_builder(add.func).arg(&mut self.x).arg(&self.down).launch(add.cfg())?;
            }
            if let Some(t) = trace.as_deref_mut() {
                t.after_mlp.push(s.memcpy_dtov(&self.x)?);
            }
        }
        unsafe {
            s.launch_builder(norm.func).arg(&self.x).arg(&w.final_norm).arg(&mut self.xn).arg(&eps).launch(one(norm))?;
            s.launch_builder(g_lm.func).arg(&w.lm_head).arg(&self.xn).arg(&mut self.logits).launch(g_lm.cfg())?;
            s.launch_builder(argmax.func).arg(&self.logits).arg(&mut self.d_token).arg(&mut self.out).arg(&mut self.d_pos).launch(argmax.cfg())?;
        }
        Ok(())
    }
}

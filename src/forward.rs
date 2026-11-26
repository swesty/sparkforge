//! Dense decode on the GPU, wired from the generated kernels in the order the CPU reference
//! (`reference::decode_step`) runs. A step reads its token and position from device memory:
//! `embed_gather` looks up the token, then per layer RMSNorm, the fused qkv GEMV, RoPE with the
//! K/V append at the position, split attention and its combine, o_proj and a residual add, RMSNorm,
//! the fused gate/up GEMV, SiLU-and-multiply, down and a residual add; after the last layer, the
//! final RMSNorm, `lm_head`, and `argmax_advance`, which writes the greedy next token, records it
//! and increments the position. Every launch is the same each step, so one captured CUDA Graph
//! replays the whole step with no host work between tokens.
//!
//! `Runtime::prefill` runs a prompt in one pass per chunk with the same kernels launched one row per
//! prompt token (grid.y), the prefill GEMMs in place of the GEMVs and `attn_prefill` in place of the
//! split attention; it fills the KV cache and leaves the token and position where the next step
//! expects them, so the captured graph continues decoding from the prompt.
use crate::codegen::{Generated, KernelEntry};
use crate::plan::{GemmTile, Plan};
use crate::prefill::{gemm_at, Buf};
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

/// Every prefill-only kernel: the prefill GEMMs (with their tiles) and `attn_prefill`.
pub const PREFILL_KERNELS: [&str; 5] = ["gemm_qkv_proj", "gemm_o_proj", "gemm_gate_up_proj", "gemm_down_proj", "attn_prefill"];

/// The prefill GEMMs and attention of a dense plan, each with its plan entry.
pub struct PrefillKernels<'a> {
    gemm: HashMap<&'static str, (&'a GemmTile, Kernel<'a>)>,
    attn: Kernel<'a>,
    block_m: u32,
}

impl<'a> PrefillKernels<'a> {
    /// Looks up every kernel in `PREFILL_KERNELS` with its tile; fails naming the ones that were
    /// not generated.
    pub fn new(p: &'a Plan, g: &'a Generated, e: &'a Engine) -> Result<Self> {
        let found = |name: &str| -> Option<Kernel<'a>> {
            let entry = g.kernels.iter().find(|k| k.entry == name && k.status == "generated")?;
            Some(Kernel { entry, func: e.funcs.get(name)? })
        };
        let mut gemm = HashMap::new();
        let mut missing = vec![];
        for name in &PREFILL_KERNELS[..4] {
            let tile = p.gemm.iter().find(|t| format!("gemm_{}", t.linear) == *name);
            match (tile, found(name)) {
                (Some(t), Some(k)) => { gemm.insert(*name, (t, k)); }
                _ => missing.push(*name),
            }
        }
        let attn = found("attn_prefill");
        let pa = p.prefill_attn.as_ref();
        if attn.is_none() || pa.is_none() { missing.push("attn_prefill"); }
        ensure!(missing.is_empty(), "prefill needs kernels that were not generated: {}", missing.join(", "));
        Ok(Self { gemm, attn: attn.expect("checked"), block_m: pa.expect("checked").block_m })
    }
}

/// Bytes `Runtime::enable_prefill` allocates for chunks of `rows` prompt tokens.
pub fn prefill_bytes(m: &ModelSpec, rows: usize) -> u64 {
    let (h, inter, q, qkv) = (m.hidden as u64, m.intermediate as u64, m.q_dim() as u64, m.qkv_dim() as u64);
    rows as u64 * ((4 * h + qkv + q + 3 * inter) * 2 + 4)
}

/// Activations for a chunk of up to `rows` prompt tokens.
struct PrefillBufs<'a> {
    k: PrefillKernels<'a>,
    rows: usize,
    tokens: CudaSlice<i32>,
    x: CudaSlice<u16>,
    xn: CudaSlice<u16>,
    qkv: CudaSlice<u16>,
    att: CudaSlice<u16>,
    o: CudaSlice<u16>,
    gate_up: CudaSlice<u16>,
    act: CudaSlice<u16>,
    down: CudaSlice<u16>,
}

/// Device addresses of one layer's weights, row-major `[n][k]` bf16, fused as the binding packs
/// them (q|k|v, gate|up).
pub struct LayerPtrs {
    pub(crate) attn_norm: u64,
    pub(crate) qkv: u64,
    pub(crate) o_proj: u64,
    pub(crate) mlp_norm: u64,
    pub(crate) gate_up: u64,
    pub(crate) down: u64,
}

/// Device addresses of every weight a step reads. Only the owners of those allocations build one
/// (`UploadedWeights`, `device_weights::DenseDevice`), and a `Runtime` borrows the owner for its
/// whole life, so the memory can't be freed while a captured graph still holds the addresses.
pub struct WeightPtrs {
    pub(crate) embed: u64,
    pub(crate) embed_rows: usize,
    pub(crate) final_norm: u64,
    pub(crate) lm_head: u64,
    pub(crate) layers: Vec<LayerPtrs>,
}

/// Owns the GPU memory behind a `WeightPtrs`.
pub trait DeviceWeights {
    fn weight_ptrs(&self) -> &WeightPtrs;
}

fn addr<T>(s: &Arc<CudaStream>, x: &CudaSlice<T>) -> u64 {
    let (p, _sync) = x.device_ptr(s);
    p
}

/// Random weights (`reference::DenseWeights`) copied into GPU memory.
pub struct UploadedWeights {
    _owned: Vec<CudaSlice<u16>>,
    ptrs: WeightPtrs,
}

impl DeviceWeights for UploadedWeights {
    fn weight_ptrs(&self) -> &WeightPtrs {
        &self.ptrs
    }
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
    /// The next step's token was set by the host (and checked against the embedding rows), not
    /// written by the previous step's `argmax_advance`.
    host_token: bool,
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
    prefill: Option<PrefillBufs<'a>>,
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
    pub fn new(s: &Arc<CudaStream>, k: Kernels<'a>, m: &'a ModelSpec, w: &'a impl DeviceWeights, splits: usize, capacity: usize) -> Result<Self> {
        let w = w.weight_ptrs();
        let n = capacity * m.kv_dim() as usize;
        let layers = w.layers.len();
        let k_cache = (0..layers).map(|_| s.alloc_zeros::<u16>(n)).collect::<Result<Vec<_>, _>>()?;
        let v_cache = (0..layers).map(|_| s.alloc_zeros::<u16>(n)).collect::<Result<Vec<_>, _>>()?;
        Self::with_caches(s, k, m, w, splits, capacity, k_cache, v_cache)
    }

    /// KV caches copied from a host cache (for example a random prefix), checked first.
    pub fn from_host(s: &Arc<CudaStream>, k: Kernels<'a>, m: &'a ModelSpec, w: &'a impl DeviceWeights, splits: usize, cache: &KvCache) -> Result<Self> {
        let w = w.weight_ptrs();
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
            host_token: false,
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
            prefill: None,
        })
    }

    /// Allocates prefill activations for chunks of up to `rows` prompt tokens (`prefill_bytes`).
    pub fn enable_prefill(&mut self, k: PrefillKernels<'a>, rows: usize) -> Result<()> {
        // grid.y carries the row of the row-wise kernels.
        ensure!((1..=65_535).contains(&rows), "a prefill chunk of {rows} rows (1 to 65535)");
        let m = self.m;
        let (h, inter, q, qkv) = (m.hidden as usize, m.intermediate as usize, m.q_dim() as usize, m.qkv_dim() as usize);
        let s = &self.s;
        self.prefill = Some(PrefillBufs {
            k,
            rows,
            tokens: s.alloc_zeros(rows)?,
            x: s.alloc_zeros(rows * h)?,
            xn: s.alloc_zeros(rows * h)?,
            qkv: s.alloc_zeros(rows * qkv)?,
            att: s.alloc_zeros(rows * q)?,
            o: s.alloc_zeros(rows * h)?,
            gate_up: s.alloc_zeros(rows * 2 * inter)?,
            act: s.alloc_zeros(rows * inter)?,
            down: s.alloc_zeros(rows * h)?,
        });
        Ok(())
    }

    /// Runs `tokens` at positions `start..start + tokens.len()` in chunks of the enabled rows: fills
    /// the KV cache, then computes the last position's logits and greedy token with the decode
    /// kernels, so the next `step` decodes that token at the position after the prompt. Positions
    /// before `start` must already be in the cache.
    pub fn prefill(&mut self, tokens: &[u32], start: usize) -> Result<()> {
        let rows = self.prefill.as_ref().map(|p| p.rows).context("prefill was not enabled")?;
        ensure!(!tokens.is_empty(), "an empty prompt");
        let end = start.checked_add(tokens.len()).context("prompt positions overflow")?;
        ensure!(end <= self.capacity, "positions {start}..{end} are past the cache capacity {}", self.capacity);
        ensure!(end <= i32::MAX as usize, "positions {start}..{end} overflow the kernels' int positions");
        let rows_m = self.w.embed_rows;
        ensure!(tokens.iter().all(|&t| (t as usize) < rows_m), "a prompt token is past the {rows_m} embedding rows");
        for (i, chunk) in tokens.chunks(rows).enumerate() {
            self.prefill_chunk(chunk, start + i * rows)?;
        }
        // The last prompt position's logits and greedy token, exactly as a decode step ends.
        let pb = self.prefill.as_ref().expect("checked");
        let h = self.m.hidden as usize;
        let last = (tokens.len() - 1) % rows;
        let k = &self.k;
        let (norm, g_lm, argmax) = (k.get("rmsnorm"), k.get("gemv_lm_head"), k.get("argmax_advance"));
        let eps = self.m.rms_eps as f32;
        self.s.memcpy_htod(&[(end - 1) as i32], &mut self.d_pos)?;
        let xl = pb.x.slice(last * h..(last + 1) * h);
        unsafe {
            self.s.launch_builder(norm.func).arg(&xl).arg(&self.w.final_norm).arg(&mut self.xn).arg(&eps).launch(LaunchConfig { grid_dim: (1, 1, 1), ..norm.cfg() })?;
            self.s.launch_builder(g_lm.func).arg(&self.w.lm_head).arg(&self.xn).arg(&mut self.logits).launch(g_lm.cfg())?;
            self.s.launch_builder(argmax.func).arg(&self.logits).arg(&mut self.d_token).arg(&mut self.out).arg(&mut self.d_pos).launch(argmax.cfg())?;
        }
        self.pos = end;
        self.host_token = false;
        Ok(())
    }

    fn prefill_chunk(&mut self, tokens: &[u32], start: usize) -> Result<()> {
        let m = self.m;
        let n = tokens.len();
        let (eps, scale_log2) = (m.rms_eps as f32, std::f32::consts::LOG2_E / (m.head_dim as f32).sqrt());
        let s = self.s.clone();
        let k = &self.k;
        let w = self.w;
        let pb = self.prefill.as_mut().expect("checked by prefill");
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        s.memcpy_htod(&ids, &mut pb.tokens.slice_mut(..n))?;
        // RoPE reads the chunk's first position from d_pos and adds the row.
        s.memcpy_htod(&[start as i32], &mut self.d_pos)?;
        let (embed, norm, rope, silu, add) = (k.get("embed_gather"), k.get("rmsnorm"), k.get("rope_kv_append"), k.get("silu_mul"), k.get("residual_add"));
        let rows = |kn: &Kernel| { let c = kn.cfg(); LaunchConfig { grid_dim: (c.grid_dim.0, n as u32, 1), ..c } };
        let norm_cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), ..norm.cfg() };
        let a = &pb.k.attn;
        let attn_cfg = LaunchConfig { grid_dim: (m.heads, n.div_ceil(pb.k.block_m as usize) as u32, 1), ..a.cfg() };
        let (n32, start32) = (n as i32, start as i32);
        let at = |x: &CudaSlice<u16>| Buf { ptr: addr(&s, x), len: x.len() };
        let (bx, bxn, bqkv, batt, bo, bgu, bact, bdown) = (at(&pb.x), at(&pb.xn), at(&pb.qkv), at(&pb.att), at(&pb.o), at(&pb.gate_up), at(&pb.act), at(&pb.down));
        let gemm = |name: &str, a: Buf, wp: u64, c: Buf| -> Result<()> {
            let (t, kn) = &pb.k.gemm[name];
            gemm_at(&s, t, kn.entry, kn.func, a, Buf { ptr: wp, len: (t.n * t.k) as usize }, c, n)
        };
        unsafe { s.launch_builder(embed.func).arg(&w.embed).arg(&pb.tokens).arg(&bx.ptr).launch(rows(embed))? };
        for (l, lw) in w.layers.iter().enumerate() {
            unsafe { s.launch_builder(norm.func).arg(&bx.ptr).arg(&lw.attn_norm).arg(&bxn.ptr).arg(&eps).launch(norm_cfg)? };
            gemm("gemm_qkv_proj", bxn, lw.qkv, bqkv)?;
            unsafe {
                s.launch_builder(rope.func).arg(&bqkv.ptr).arg(&mut self.k_cache[l]).arg(&mut self.v_cache[l]).arg(&self.d_pos).launch(rows(rope))?;
                s.launch_builder(a.func).arg(&bqkv.ptr).arg(&self.k_cache[l]).arg(&self.v_cache[l]).arg(&batt.ptr).arg(&n32).arg(&start32).arg(&scale_log2)
                    .launch(attn_cfg)?;
            }
            gemm("gemm_o_proj", batt, lw.o_proj, bo)?;
            unsafe {
                s.launch_builder(add.func).arg(&bx.ptr).arg(&bo.ptr).launch(rows(add))?;
                s.launch_builder(norm.func).arg(&bx.ptr).arg(&lw.mlp_norm).arg(&bxn.ptr).arg(&eps).launch(norm_cfg)?;
            }
            gemm("gemm_gate_up_proj", bxn, lw.gate_up, bgu)?;
            unsafe { s.launch_builder(silu.func).arg(&bgu.ptr).arg(&bact.ptr).launch(rows(silu))? };
            gemm("gemm_down_proj", bact, lw.down, bdown)?;
            unsafe { s.launch_builder(add.func).arg(&bx.ptr).arg(&bdown.ptr).launch(rows(add))? };
        }
        Ok(())
    }

    /// Layer `layer`'s K and V cache, positions `0..to`.
    pub fn kv(&self, layer: usize, to: usize) -> Result<(Vec<u16>, Vec<u16>)> {
        ensure!(layer < self.k_cache.len() && to <= self.capacity, "layer {layer} positions ..{to} are outside the cache");
        let n = to * self.m.kv_dim() as usize;
        Ok((self.s.memcpy_dtov(&self.k_cache[layer].slice(..n))?, self.s.memcpy_dtov(&self.v_cache[layer].slice(..n))?))
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
        self.host_token = true;
        Ok(())
    }

    /// Zeroes every KV cache slot, so a measurement starts from the same cache whatever ran before.
    pub fn clear_cache(&mut self) -> Result<()> {
        for c in self.k_cache.iter_mut().chain(self.v_cache.iter_mut()) {
            self.s.memset_zeros(c)?;
        }
        Ok(())
    }

    /// Checks a step can run: within the cache, and, when the token comes from the previous step's
    /// `argmax_advance` (any vocabulary index), with an embedding row for every token.
    fn check_step(&self) -> Result<()> {
        ensure!(self.pos < self.capacity, "position {} is past the cache capacity {}", self.pos, self.capacity);
        ensure!(self.host_token || self.w.embed_rows >= self.m.vocab as usize,
            "a step after a step looks up a greedy token from the whole vocabulary, but only {} of {} embedding rows are on the GPU; set the token first",
            self.w.embed_rows, self.m.vocab);
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
        self.check_step()?;
        match &self.graph {
            Some(g) => g.launch()?,
            None => self.enqueue(None)?,
        }
        self.pos += 1;
        self.host_token = false;
        Ok(())
    }

    /// Runs one step with plain launches, copying the hidden state back after every residual add
    /// (synchronizing each time).
    pub fn step_traced(&mut self) -> Result<GpuTrace> {
        self.check_step()?;
        let mut t = GpuTrace { after_attn: vec![], after_mlp: vec![], logits: vec![] };
        self.enqueue(Some(&mut t))?;
        t.logits = self.s.memcpy_dtov(&self.logits)?;
        self.pos += 1;
        self.host_token = false;
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

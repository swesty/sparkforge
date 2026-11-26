//! Scalar CPU reference for one decode step of a dense (Llama-shaped) model, so `verify` can diff
//! the GPU's activations at every layer boundary, not just each kernel in isolation.
//!
//! Rounding follows HF transformers running the model in bf16: every tensor a kernel writes in
//! bf16 is rounded to bf16 here too, and everything inside one op is computed in f64 (f32 where
//! the kernel and HF agree on f32: the RoPE angle and the residual add). Specifically:
//! - RMSNorm rounds `x * rsqrt(mean(x^2) + eps)` to bf16 before multiplying by the weight, as HF's
//!   `LlamaRMSNorm` does (`weight * hidden.to(input_dtype)`), then rounds the product.
//! - Each GEMV, RoPE output, attention output, SiLU-and-multiply and residual sum is rounded to bf16.
//! - Attention keeps the softmax probabilities unrounded, as the SDPA and flash backends do.
//! - The logits are compared unrounded, like every per-kernel reference.
use crate::config::{ModelSpec, WeightFormat};
use anyhow::{ensure, Context, Result};

pub fn f2bf(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
pub fn bf2f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
/// f64 to bf16, round to nearest even, rounded once: going through f32 first can land exactly on
/// a bf16 midpoint and then tie the wrong way, for normal and subnormal results alike.
pub fn f64_to_bf16(x: f64) -> u16 {
    if !x.is_finite() {
        return f2bf(x as f32);
    }
    if x.abs() < f32::MIN_POSITIVE as f64 {
        // bf16 subnormals are multiples of 2^-133 (2^-126 with 7 mantissa bits); 128 of them is the
        // smallest normal, whose bits are 0x0080, so the count is the bit pattern either way.
        let q = x.abs() * 2f64.powi(133);
        let (fl, frac) = (q.floor(), q - q.floor());
        let n = if frac > 0.5 || (frac == 0.5 && fl % 2.0 == 1.0) { fl + 1.0 } else { fl };
        return (n as u16) | if x.is_sign_negative() { 0x8000 } else { 0 };
    }
    // Keep 7 of the 52 mantissa bits, round to nearest even; the result has 8 significant bits,
    // so it converts to f32 exactly (or to infinity past f32::MAX) and its low 16 bits are zero.
    const SHIFT: u32 = 52 - 7;
    let b = x.to_bits();
    let r = b.wrapping_add((1u64 << (SHIFT - 1)) - 1 + ((b >> SHIFT) & 1)) & !((1u64 << SHIFT) - 1);
    f2bf(f64::from_bits(r) as f32)
}
fn round(v: &[f64]) -> Vec<u16> {
    v.iter().map(|&x| f64_to_bf16(x)).collect()
}

/// Seeded xorshift64; `unif` is uniform in [-1, 1).
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn unif(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    pub fn bf16(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| f2bf(self.unif())).collect()
    }
    /// `n` bf16 values uniform in [-a, a).
    pub fn bf16_scaled(&mut self, n: usize, a: f32) -> Vec<u16> {
        (0..n).map(|_| f2bf(self.unif() * a)).collect()
    }
}

/// Fails unless `m` is a dense model the decode step covers: no experts, no hybrid layers, no
/// q/k norm or q/k/v bias, and BF16 weights throughout.
pub fn require_dense(m: &ModelSpec) -> Result<()> {
    ensure!(m.hybrid.is_none(), "{}: hybrid layers are not covered by the dense decode step", m.arch);
    ensure!(m.moe.is_none(), "{}: mixture-of-experts layers are not covered by the dense decode step", m.arch);
    ensure!(!m.qk_norm && !m.qkv_bias, "{}: q/k norm and q/k/v bias are not covered by the dense decode step", m.arch);
    ensure!(matches!(m.weights, WeightFormat::Bf16) && m.formats.is_empty(), "{}: the dense decode step needs BF16 weights", m.arch);
    Ok(())
}

/// Row-major `[n][k]` bf16 matrices, fused like the checkpoint binding: `qkv` is q, k, v rows and
/// `gate_up` is gate rows then up rows.
pub struct LayerWeights {
    pub attn_norm: Vec<u16>,
    pub qkv: Vec<u16>,
    pub o_proj: Vec<u16>,
    pub mlp_norm: Vec<u16>,
    pub gate_up: Vec<u16>,
    pub down: Vec<u16>,
}

pub struct DenseWeights {
    /// `[rows][hidden]`: the embedding rows a run uses (row `i` is the input of a step given `i`).
    pub embed: Vec<u16>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<u16>,
    /// `[vocab][hidden]`.
    pub lm_head: Vec<u16>,
}

impl DenseWeights {
    /// Random weights for the first `layers` layers of `m`. Linear weights are uniform in
    /// ±sqrt(3/K), so a unit-RMS input gives a unit-RMS output; norm weights are 1 ± 0.1; embedding
    /// rows are uniform in [-1, 1).
    pub fn random(m: &ModelSpec, layers: usize, embed_rows: usize, rng: &mut Rng) -> Self {
        let (h, inter) = (m.hidden as usize, m.intermediate as usize);
        let (q, qkv) = (m.q_dim() as usize, m.qkv_dim() as usize);
        let lin = |rng: &mut Rng, n: usize, k: usize| rng.bf16_scaled(n * k, (3.0 / k as f32).sqrt());
        let norm = |rng: &mut Rng, n: usize| (0..n).map(|_| f2bf(1.0 + 0.1 * rng.unif())).collect::<Vec<_>>();
        let embed = rng.bf16(embed_rows * h);
        let layers = (0..layers)
            .map(|_| LayerWeights {
                attn_norm: norm(rng, h),
                qkv: lin(rng, qkv, h),
                o_proj: lin(rng, h, q),
                mlp_norm: norm(rng, h),
                gate_up: lin(rng, 2 * inter, h),
                down: lin(rng, h, inter),
            })
            .collect();
        let final_norm = norm(rng, h);
        let lm_head = lin(rng, m.vocab as usize, h);
        Self { embed, layers, final_norm, lm_head }
    }
}

/// Per-layer bf16 K and V caches, each `[capacity][kv_heads][head_dim]`, as the GPU lays them out.
pub struct KvCache {
    pub k: Vec<Vec<u16>>,
    pub v: Vec<Vec<u16>>,
    pub capacity: usize,
}

impl KvCache {
    pub fn new(m: &ModelSpec, layers: usize, capacity: usize) -> Self {
        let n = capacity * m.kv_dim() as usize;
        Self { k: vec![vec![0; n]; layers], v: vec![vec![0; n]; layers], capacity }
    }
    /// Fills positions `0..len` of every layer with random bf16 in [-1, 1), standing in for a prompt.
    pub fn fill_random(&mut self, m: &ModelSpec, len: usize, rng: &mut Rng) {
        let n = len * m.kv_dim() as usize;
        for c in self.k.iter_mut().chain(self.v.iter_mut()) {
            c[..n].copy_from_slice(&rng.bf16(n));
        }
    }
}

fn len(what: String, got: usize, want: usize) -> Result<()> {
    ensure!(got == want, "{what} has {got} elements, the model needs {want}");
    Ok(())
}

/// Fails, naming the buffer, unless `cache` has a K and a V buffer for each of `layers` layers,
/// each `capacity` positions long.
pub fn check_cache(m: &ModelSpec, layers: usize, cache: &KvCache) -> Result<()> {
    ensure!(layers > 0, "no layers to run");
    ensure!(cache.k.len() == layers && cache.v.len() == layers, "the cache has {} K and {} V layers, the weights {layers}", cache.k.len(), cache.v.len());
    let kv = m.kv_dim() as usize;
    for i in 0..layers {
        len(format!("layer {i} K cache"), cache.k[i].len(), cache.capacity * kv)?;
        len(format!("layer {i} V cache"), cache.v[i].len(), cache.capacity * kv)?;
    }
    Ok(())
}

/// Fails, naming the tensor, unless every weight length matches `m`.
pub fn check_weights(m: &ModelSpec, w: &DenseWeights) -> Result<()> {
    let (h, inter, q, qkv) = (m.hidden as usize, m.intermediate as usize, m.q_dim() as usize, m.qkv_dim() as usize);
    ensure!(!w.layers.is_empty(), "no layers to run");
    ensure!(!w.embed.is_empty() && w.embed.len().is_multiple_of(h), "the embedding has {} elements, not a whole number of {h}-wide rows", w.embed.len());
    len("final_norm".into(), w.final_norm.len(), h)?;
    len("lm_head".into(), w.lm_head.len(), m.vocab as usize * h)?;
    for (i, lw) in w.layers.iter().enumerate() {
        len(format!("layer {i} attn_norm"), lw.attn_norm.len(), h)?;
        len(format!("layer {i} qkv"), lw.qkv.len(), qkv * h)?;
        len(format!("layer {i} o_proj"), lw.o_proj.len(), h * q)?;
        len(format!("layer {i} mlp_norm"), lw.mlp_norm.len(), h)?;
        len(format!("layer {i} gate_up"), lw.gate_up.len(), 2 * inter * h)?;
        len(format!("layer {i} down"), lw.down.len(), h * inter)?;
    }
    Ok(())
}

/// Both checks: what the GPU step indexes with, so `forward` checks before copying anything.
pub fn check_shapes(m: &ModelSpec, w: &DenseWeights, cache: &KvCache) -> Result<()> {
    check_weights(m, w)?;
    check_cache(m, w.layers.len(), cache)
}

/// A dense model's linears, by layer.
#[derive(Clone, Copy, Debug)]
pub enum Linear {
    Qkv(usize),
    O(usize),
    GateUp(usize),
    Down(usize),
    LmHead,
}

/// A dense model's RMSNorm weights, by layer.
#[derive(Clone, Copy, Debug)]
pub enum Norm {
    Attn(usize),
    Mlp(usize),
    Final,
}

/// Where the reference reads weights: random `DenseWeights`, or a checkpoint streamed from disk
/// (`CheckpointWeights`), since GB10's unified memory can't hold a second host copy of a model
/// already resident on the GPU.
pub trait Weights {
    fn layers(&self) -> usize;
    fn embed_rows(&self) -> usize;
    /// Fails unless every tensor has the shape `m` needs.
    fn check(&self, m: &ModelSpec) -> Result<()>;
    fn embed_row(&self, row: usize) -> Result<Vec<u16>>;
    fn norm(&self, n: Norm) -> Result<Vec<u16>>;
    /// `W x` in f64, unrounded.
    fn gemv(&self, l: Linear, x: &[u16]) -> Result<Vec<f64>>;
}

impl Weights for DenseWeights {
    fn layers(&self) -> usize {
        self.layers.len()
    }
    fn embed_rows(&self) -> usize {
        self.embed.len() / self.final_norm.len().max(1)
    }
    fn check(&self, m: &ModelSpec) -> Result<()> {
        check_weights(m, self)
    }
    fn embed_row(&self, row: usize) -> Result<Vec<u16>> {
        let h = self.final_norm.len();
        Ok(self.embed[row * h..(row + 1) * h].to_vec())
    }
    fn norm(&self, n: Norm) -> Result<Vec<u16>> {
        Ok(match n {
            Norm::Attn(l) => self.layers[l].attn_norm.clone(),
            Norm::Mlp(l) => self.layers[l].mlp_norm.clone(),
            Norm::Final => self.final_norm.clone(),
        })
    }
    fn gemv(&self, l: Linear, x: &[u16]) -> Result<Vec<f64>> {
        let w = match l {
            Linear::Qkv(i) => &self.layers[i].qkv,
            Linear::O(i) => &self.layers[i].o_proj,
            Linear::GateUp(i) => &self.layers[i].gate_up,
            Linear::Down(i) => &self.layers[i].down,
            Linear::LmHead => &self.lm_head,
        };
        Ok(gemv(w, w.len() / x.len(), x))
    }
}

/// A BF16 checkpoint read through its binding: linears with `weights::reference_gemv` (bounded
/// chunks, fused parts in kernel row order), norms and embedding rows read on demand.
pub struct CheckpointWeights<'a> {
    pub ck: &'a crate::weights::Checkpoint,
    pub b: &'a crate::weights::Binding,
    pub m: &'a ModelSpec,
}

impl CheckpointWeights<'_> {
    fn bf16(&self, name: &str, off: u64, n: usize) -> Result<Vec<u16>> {
        let t = self.ck.tensors.get(name).with_context(|| format!("{name} is not in the checkpoint"))?;
        ensure!(t.dtype == "BF16", "{name} is {}, the reference reads BF16", t.dtype);
        let mut buf = vec![0u8; n * 2];
        self.ck.read_range(t, off, &mut buf)?;
        Ok(buf.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect())
    }
    fn linear(&self, l: Linear) -> Result<&crate::weights::BoundLinear> {
        let (name, layer) = match l {
            Linear::Qkv(i) => ("qkv_proj", Some(i as u32)),
            Linear::O(i) => ("o_proj", Some(i as u32)),
            Linear::GateUp(i) => ("gate_up_proj", Some(i as u32)),
            Linear::Down(i) => ("down_proj", Some(i as u32)),
            Linear::LmHead => ("lm_head", None),
        };
        self.b.linears.iter().find(|bl| bl.linear == name && bl.layer == layer).with_context(|| format!("{l:?} is not bound"))
    }
}

impl Weights for CheckpointWeights<'_> {
    fn layers(&self) -> usize {
        self.m.layers as usize
    }
    fn embed_rows(&self) -> usize {
        self.m.vocab as usize
    }
    fn check(&self, m: &ModelSpec) -> Result<()> {
        // `weights::bind` checked every tensor's shape and dtype against this model.
        ensure!(std::ptr::eq(m, self.m) || (m.hidden, m.layers, m.vocab) == (self.m.hidden, self.m.layers, self.m.vocab), "the checkpoint was bound for another model");
        Ok(())
    }
    fn embed_row(&self, row: usize) -> Result<Vec<u16>> {
        let h = self.m.hidden as usize;
        self.bf16("model.embed_tokens.weight", (row * h * 2) as u64, h)
    }
    fn norm(&self, n: Norm) -> Result<Vec<u16>> {
        let name = match n {
            Norm::Attn(l) => format!("model.layers.{l}.input_layernorm.weight"),
            Norm::Mlp(l) => format!("model.layers.{l}.post_attention_layernorm.weight"),
            Norm::Final => "model.norm.weight".into(),
        };
        self.bf16(&name, 0, self.m.hidden as usize)
    }
    fn gemv(&self, l: Linear, x: &[u16]) -> Result<Vec<f64>> {
        let xf: Vec<f64> = x.iter().map(|&v| bf2f(v) as f64).collect();
        crate::weights::reference_gemv(self.ck, self.linear(l)?, &xf)
    }
}

/// The hidden state after each layer's attention and MLP residual adds, and the final logits.
pub struct Trace {
    pub after_attn: Vec<Vec<u16>>,
    pub after_mlp: Vec<Vec<u16>>,
    pub logits: Vec<f64>,
}

pub fn rmsnorm(x: &[u16], w: &[u16], eps: f64) -> Vec<u16> {
    let ss: f64 = x.iter().map(|&v| (bf2f(v) as f64).powi(2)).sum();
    let inv = 1.0 / (ss / x.len() as f64 + eps).sqrt();
    // The weight times a bf16 value is exact in f32, so the last rounding is f32 to bf16.
    x.iter().zip(w).map(|(&v, &wv)| f2bf(bf2f(wv) * bf2f(f64_to_bf16(bf2f(v) as f64 * inv)))).collect()
}

/// `W x` for row-major `W` of `n` rows, in f64, unrounded.
pub fn gemv(w: &[u16], n: usize, x: &[u16]) -> Vec<f64> {
    let k = x.len();
    assert_eq!(w.len(), n * k);
    let xf: Vec<f64> = x.iter().map(|&v| bf2f(v) as f64).collect();
    let mut r = vec![0f64; n];
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, out) in r.chunks_mut(chunk).enumerate() {
            let xf = &xf;
            sc.spawn(move || {
                for (j, o) in out.iter_mut().enumerate() {
                    let row = &w[(ci * chunk + j) * k..][..k];
                    *o = row.iter().zip(xf).map(|(&a, &b)| bf2f(a) as f64 * b).sum();
                }
            });
        }
    });
    r
}

/// Rotates the q and k heads of a fused qkv vector in place (NeoX half-split), with the angle
/// formed in f32 as HF does and an exact sin/cos.
pub fn rope(m: &ModelSpec, qkv: &mut [u16], pos: usize) {
    let hd = m.head_dim as usize;
    let inv_freq = crate::codegen::rope_inv_freq(m);
    for h in 0..(m.heads + m.kv_heads) as usize {
        for i in 0..hd / 2 {
            let ang = (pos as f32 * inv_freq[i]) as f64;
            let (x1, x2) = (bf2f(qkv[h * hd + i]) as f64, bf2f(qkv[h * hd + i + hd / 2]) as f64);
            qkv[h * hd + i] = f64_to_bf16(x1 * ang.cos() - x2 * ang.sin());
            qkv[h * hd + i + hd / 2] = f64_to_bf16(x2 * ang.cos() + x1 * ang.sin());
        }
    }
}

/// Decode attention of every query head over cache positions `0..len`; query head `h` reads kv
/// head `h / (heads / kv_heads)`.
pub fn attention(m: &ModelSpec, q: &[u16], k_cache: &[u16], v_cache: &[u16], len: usize) -> Vec<u16> {
    let (hd, kvh, group) = (m.head_dim as usize, m.kv_heads as usize, m.gqa_group() as usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0u16; m.q_dim() as usize];
    for h in 0..m.heads as usize {
        let kv = h / group;
        let qs: Vec<f64> = (0..hd).map(|d| (bf2f(q[h * hd + d]) * scale) as f64).collect();
        let sc: Vec<f64> = (0..len).map(|t| (0..hd).map(|d| qs[d] * bf2f(k_cache[(t * kvh + kv) * hd + d]) as f64).sum()).collect();
        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
        let z: f64 = p.iter().sum();
        for d in 0..hd {
            let o: f64 = (0..len).map(|t| p[t] * bf2f(v_cache[(t * kvh + kv) * hd + d]) as f64).sum();
            out[h * hd + d] = f64_to_bf16(o / z);
        }
    }
    out
}

/// `[gate | up]` -> `silu(gate) * up`.
pub fn silu_mul(gate_up: &[u16]) -> Vec<u16> {
    let inter = gate_up.len() / 2;
    (0..inter)
        .map(|i| {
            let (g, u) = (bf2f(gate_up[i]) as f64, bf2f(gate_up[inter + i]) as f64);
            f64_to_bf16(g / (1.0 + (-g).exp()) * u)
        })
        .collect()
}

/// Residual add, `h + d` summed in f32 and rounded to bf16.
pub fn add(h: &[u16], d: &[u16]) -> Vec<u16> {
    h.iter().zip(d).map(|(&a, &b)| f2bf(bf2f(a) + bf2f(b))).collect()
}

/// One decode step for the token whose embedding is row `row`, at position `pos`: appends this
/// step's K and V to `cache` and returns the activations at each layer boundary. Every shape, the
/// row and the position are checked before anything is written, so a refused step leaves the
/// cache as it was.
pub fn decode_step(m: &ModelSpec, w: &impl Weights, cache: &mut KvCache, row: usize, pos: usize) -> Result<Trace> {
    let (q, kv) = (m.q_dim() as usize, m.kv_dim() as usize);
    let eps = m.rms_eps;
    w.check(m)?;
    check_cache(m, w.layers(), cache)?;
    ensure!(pos < cache.capacity, "position {pos} is past the cache capacity {}", cache.capacity);
    ensure!(row < w.embed_rows(), "embedding row {row} is past the {} rows", w.embed_rows());
    let mut x = w.embed_row(row)?;
    let mut t = Trace { after_attn: vec![], after_mlp: vec![], logits: vec![] };
    for l in 0..w.layers() {
        let xn = rmsnorm(&x, &w.norm(Norm::Attn(l))?, eps);
        let mut qkv = round(&w.gemv(Linear::Qkv(l), &xn)?);
        ensure!(qkv.len() == q + 2 * kv, "layer {l} qkv gives {} outputs, the model needs {}", qkv.len(), q + 2 * kv);
        rope(m, &mut qkv, pos);
        cache.k[l][pos * kv..(pos + 1) * kv].copy_from_slice(&qkv[q..q + kv]);
        cache.v[l][pos * kv..(pos + 1) * kv].copy_from_slice(&qkv[q + kv..]);
        let att = attention(m, &qkv[..q], &cache.k[l], &cache.v[l], pos + 1);
        x = add(&x, &round(&w.gemv(Linear::O(l), &att)?));
        t.after_attn.push(x.clone());
        let xn = rmsnorm(&x, &w.norm(Norm::Mlp(l))?, eps);
        let act = silu_mul(&round(&w.gemv(Linear::GateUp(l), &xn)?));
        x = add(&x, &round(&w.gemv(Linear::Down(l), &act)?));
        t.after_mlp.push(x.clone());
    }
    t.logits = w.gemv(Linear::LmHead, &rmsnorm(&x, &w.norm(Norm::Final)?, eps))?;
    Ok(t)
}

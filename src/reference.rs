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
use anyhow::{ensure, Result};

pub fn f2bf(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
pub fn bf2f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
/// f64 to bf16, round to nearest even, rounded once: going through f32 first can land exactly on
/// a bf16 midpoint and then tie the wrong way. Values below the f32 normal range (irrelevant to
/// these activations) go through f32.
pub fn f64_to_bf16(x: f64) -> u16 {
    if !x.is_finite() || x.abs() < f32::MIN_POSITIVE as f64 {
        return f2bf(x as f32);
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

/// Fails, naming the tensor, unless every weight and cache length matches `m` and the cache's
/// declared capacity, and the cache has a K and a V buffer for every weight layer. The GPU step
/// indexes with these shapes, so `forward::DeviceModel::upload` checks before copying anything.
pub fn check_shapes(m: &ModelSpec, w: &DenseWeights, cache: &KvCache) -> Result<()> {
    let (h, inter, q, qkv, kv) = (m.hidden as usize, m.intermediate as usize, m.q_dim() as usize, m.qkv_dim() as usize, m.kv_dim() as usize);
    let len = |what: String, got: usize, want: usize| -> Result<()> {
        ensure!(got == want, "{what} has {got} elements, the model needs {want}");
        Ok(())
    };
    let l = w.layers.len();
    ensure!(l > 0, "no layers to run");
    ensure!(cache.k.len() == l && cache.v.len() == l, "the cache has {} K and {} V layers, the weights {l}", cache.k.len(), cache.v.len());
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
        len(format!("layer {i} K cache"), cache.k[i].len(), cache.capacity * kv)?;
        len(format!("layer {i} V cache"), cache.v[i].len(), cache.capacity * kv)?;
    }
    Ok(())
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

/// One decode step for the token whose embedding is row `row` of `w.embed`, at position `pos`:
/// appends this step's K and V to `cache` and returns the activations at each layer boundary.
pub fn decode_step(m: &ModelSpec, w: &DenseWeights, cache: &mut KvCache, row: usize, pos: usize) -> Trace {
    let (h, inter, q, kv) = (m.hidden as usize, m.intermediate as usize, m.q_dim() as usize, m.kv_dim() as usize);
    let eps = m.rms_eps;
    assert!(pos < cache.capacity, "position {pos} is past the cache capacity {}", cache.capacity);
    let mut x = w.embed[row * h..(row + 1) * h].to_vec();
    let mut t = Trace { after_attn: vec![], after_mlp: vec![], logits: vec![] };
    for (l, lw) in w.layers.iter().enumerate() {
        let xn = rmsnorm(&x, &lw.attn_norm, eps);
        let mut qkv = round(&gemv(&lw.qkv, q + 2 * kv, &xn));
        rope(m, &mut qkv, pos);
        cache.k[l][pos * kv..(pos + 1) * kv].copy_from_slice(&qkv[q..q + kv]);
        cache.v[l][pos * kv..(pos + 1) * kv].copy_from_slice(&qkv[q + kv..]);
        let att = attention(m, &qkv[..q], &cache.k[l], &cache.v[l], pos + 1);
        x = add(&x, &round(&gemv(&lw.o_proj, h, &att)));
        t.after_attn.push(x.clone());
        let xn = rmsnorm(&x, &lw.mlp_norm, eps);
        let act = silu_mul(&round(&gemv(&lw.gate_up, 2 * inter, &xn)));
        x = add(&x, &round(&gemv(&lw.down, h, &act)));
        t.after_mlp.push(x.clone());
    }
    t.logits = gemv(&w.lm_head, m.vocab as usize, &rmsnorm(&x, &w.final_norm, eps));
    t
}

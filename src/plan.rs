//! The planner: turns a ModelSpec into concrete per-shape kernel decisions for GB10.
//! Every number here is derived from the config + target, nothing is hand-tuned per model.
use crate::config::{ModelSpec, WeightFormat};
use crate::target::Target;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Linear {
    pub name: String,
    /// Output features (rows of W).
    pub n: u32,
    /// Input features (cols of W).
    pub k: u32,
    /// How many copies exist per layer (experts) and how many a token touches.
    pub copies: u32,
    pub active: u32,
    pub format: WeightFormat,
}

impl Linear {
    pub fn bytes_one(&self) -> f64 { self.n as f64 * self.k as f64 * self.format.bytes_per_elem() }
}

/// Decode (M = 1..few) GEMV: purely memory-bound on GB10, so we size the grid to saturate DRAM.
#[derive(Debug, Clone, Serialize)]
pub struct GemvPlan {
    pub linear: String,
    pub n: u32,
    pub k: u32,
    pub warps_per_block: u32,
    pub rows_per_warp: u32,
    pub grid: u32,
    pub smem_bytes: u32,
    pub est_us: f64,
}

/// Prefill GEMM tile choice for mma.sync on SM121.
#[derive(Debug, Clone, Serialize)]
pub struct GemmTile {
    pub linear: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub stages: u32,
    pub warps: u32,
    pub smem_bytes: u32,
    pub ctas_per_sm: u32,
    pub wave_efficiency: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AttnPlan {
    pub head_dim: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub group: u32,
    pub warps: u32,
    /// Flash-decoding sequence splits, chosen so kv_heads * splits fills the SMs.
    pub splits: u32,
    pub grid: [u32; 2],
    pub smem_bytes: u32,
    pub ref_ctx: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct NormPlan { pub hidden: u32, pub threads: u32, pub iters: u32 }

#[derive(Debug, Clone, Serialize)]
pub struct Budget {
    pub weight_gib: f64,
    pub kv_gib_at_ctx: f64,
    pub ctx: u32,
    pub fits: bool,
    pub max_ctx_that_fits: u64,
    pub decode_bytes_per_token_gb: f64,
    pub decode_tok_s_ceiling: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub target: Target,
    pub model: ModelSpec,
    pub linears: Vec<Linear>,
    pub gemv: Vec<GemvPlan>,
    pub gemm: Vec<GemmTile>,
    pub attn: AttnPlan,
    pub norm: NormPlan,
    pub budget: Budget,
    pub notes: Vec<String>,
}

pub struct PlanOpts { pub prefill_tokens: u32, pub ctx: u32 }

pub fn linears(m: &ModelSpec) -> Vec<Linear> {
    let f = m.weights;
    let mut v = vec![
        Linear { name: "qkv_proj".into(), n: m.qkv_dim(), k: m.hidden, copies: 1, active: 1, format: f },
        Linear { name: "o_proj".into(), n: m.hidden, k: m.q_dim(), copies: 1, active: 1, format: f },
    ];
    match &m.moe {
        None => {
            v.push(Linear { name: "gate_up_proj".into(), n: 2 * m.intermediate, k: m.hidden, copies: 1, active: 1, format: f });
            v.push(Linear { name: "down_proj".into(), n: m.hidden, k: m.intermediate, copies: 1, active: 1, format: f });
        }
        Some(e) => {
            v.push(Linear { name: "router".into(), n: e.num_experts, k: m.hidden, copies: 1, active: 1, format: WeightFormat::Bf16 });
            v.push(Linear { name: "expert_gate_up".into(), n: 2 * e.expert_intermediate, k: m.hidden, copies: e.num_experts, active: e.experts_per_tok, format: f });
            v.push(Linear { name: "expert_down".into(), n: m.hidden, k: e.expert_intermediate, copies: e.num_experts, active: e.experts_per_tok, format: f });
            if e.shared_experts > 0 {
                let si = e.expert_intermediate * e.shared_experts;
                v.push(Linear { name: "shared_gate_up".into(), n: 2 * si, k: m.hidden, copies: 1, active: 1, format: f });
                v.push(Linear { name: "shared_down".into(), n: m.hidden, k: si, copies: 1, active: 1, format: f });
            }
        }
    }
    v
}

fn ceil_div(a: u32, b: u32) -> u32 { a.div_ceil(b) }

fn plan_gemv(l: &Linear, t: &Target) -> GemvPlan {
    // Aim for >= 4 resident blocks per SM worth of work so DRAM requests stay in flight.
    let warps_per_block = 8;
    let target_blocks = 4 * t.sm_count;
    let mut rows_per_warp = 1;
    for r in [4u32, 2, 1] {
        if l.n % r == 0 && ceil_div(l.n, warps_per_block * r) >= target_blocks { rows_per_warp = r; break; }
    }
    let grid = ceil_div(l.n, warps_per_block * rows_per_warp);
    let bytes = l.bytes_one() * l.active as f64;
    GemvPlan {
        linear: l.name.clone(), n: l.n, k: l.k, warps_per_block, rows_per_warp, grid,
        smem_bytes: l.k * 2, // bf16 activation vector staged once per block
        est_us: bytes / (t.dram_bw_effective_gbs * 1e3),
    }
}

fn plan_gemm(l: &Linear, m: u32, t: &Target) -> GemmTile {
    let mut best: Option<GemmTile> = None;
    for &bm in &[64u32, 128] {
        for &bn in &[64u32, 128, 256] {
            for &bk in &[32u32, 64] {
                for &stages in &[2u32, 3, 4] {
                    // bf16 A and B tiles, multi-buffered (cp.async / TMA pipeline).
                    let smem = stages * (bm + bn) * bk * 2;
                    if smem > t.smem_per_block_optin { continue; }
                    let warps = (bm / 64).max(1) * (bn / 32); // warp tile 64x32
                    let threads = warps * 32;
                    if threads > t.max_threads_per_block { continue; }
                    // accumulators: 64*32 fp32 per warp = 64 regs/thread; leave headroom
                    let ctas_by_smem = t.smem_per_sm / smem.max(1);
                    let ctas_by_thr = t.max_threads_per_sm / threads;
                    let ctas_by_reg = t.regs_per_sm / (threads * 168);
                    let ctas = ctas_by_smem.min(ctas_by_thr).min(ctas_by_reg);
                    if ctas == 0 { continue; }
                    if l.k % bk != 0 { continue; }
                    let tiles = ceil_div(m, bm) * ceil_div(l.n, bn);
                    let slots = t.sm_count * ctas;
                    let waves = ceil_div(tiles, slots);
                    let wave_eff = tiles as f64 / (waves * slots) as f64;
                    let pad_eff = (m as f64 * l.n as f64) / ((ceil_div(m, bm) * bm) as f64 * (ceil_div(l.n, bn) * bn) as f64);
                    // arithmetic intensity of the tile (FLOP per byte loaded from DRAM/L2)
                    let ai = (bm * bn) as f64 / (bm + bn) as f64 / 64.0;
                    let pipe = if stages >= 3 { 1.0 } else { 0.85 };
                    let score = wave_eff * pad_eff * ai.min(1.0) * pipe * (ctas.min(2) as f64 / 2.0).sqrt();
                    let cand = GemmTile {
                        linear: l.name.clone(), m, n: l.n, k: l.k, bm, bn, bk, stages, warps,
                        smem_bytes: smem, ctas_per_sm: ctas, wave_efficiency: wave_eff, score,
                    };
                    if best.as_ref().map_or(true, |b| cand.score > b.score) { best = Some(cand); }
                }
            }
        }
    }
    best.expect("no legal GEMM tile for this shape")
}

fn plan_attn(m: &ModelSpec, t: &Target, ctx: u32) -> AttnPlan {
    let warps = 8;
    let group = m.gqa_group();
    let want_blocks = 2 * t.sm_count;
    // Each split should still own a meaningful chunk of the sequence.
    let max_splits = (ctx / 256).max(1);
    let splits = ceil_div(want_blocks, m.kv_heads).clamp(1, max_splits);
    // smem: per warp, per group-head: running max, sum, and head_dim accumulators (fp32)
    let smem = warps * group * (m.head_dim + 2) * 4;
    AttnPlan {
        head_dim: m.head_dim, heads: m.heads, kv_heads: m.kv_heads, group, warps, splits,
        grid: [m.kv_heads, splits], smem_bytes: smem, ref_ctx: ctx,
    }
}

fn plan_norm(hidden: u32, t: &Target) -> NormPlan {
    let vecs = hidden / 8;
    let mut threads = 32;
    let mut th = t.max_threads_per_block;
    while th >= 32 {
        if vecs % th == 0 { threads = th; break; }
        th -= 32;
    }
    NormPlan { hidden, threads, iters: vecs / threads }
}

pub fn build(m: &ModelSpec, t: &Target, o: &PlanOpts) -> Plan {
    let mut notes = vec![];
    let ls = linears(m);
    let per_layer_total: f64 = ls.iter().map(|l| l.bytes_one() * l.copies as f64).sum();
    let per_layer_active: f64 = ls.iter().map(|l| l.bytes_one() * l.active as f64).sum();
    let emb = m.vocab as f64 * m.hidden as f64 * 2.0;
    let head = if m.tie_embeddings { 0.0 } else { emb };
    let weight_bytes = per_layer_total * m.layers as f64 + emb + head;
    let kv = m.kv_bytes_per_token() as f64 * o.ctx as f64;
    let usable = t.usable_mem_gib * (1u64 << 30) as f64;
    let max_ctx = ((usable - weight_bytes).max(0.0) / m.kv_bytes_per_token() as f64) as u64;
    // decode: read active weights + lm_head + whole KV cache once per token
    let decode_bytes = per_layer_active * m.layers as f64 + emb + kv;
    let tok_s = t.dram_bw_effective_gbs * 1e9 / decode_bytes;

    if m.head_dim % 32 != 0 { notes.push(format!("head_dim {} not a multiple of 32; attention kernel needs a padded path", m.head_dim)); }
    if m.qk_norm { notes.push("q/k per-head RMSNorm (q_norm/k_norm) is not generated yet: decode output is wrong for this model until it is".into()); }
    if m.qkv_bias { notes.push("q/k/v projection bias is not generated yet: decode output is wrong for this model until it is".into()); }
    if m.hidden % 8 != 0 { notes.push("hidden not a multiple of 8; vectorized 16B loads disabled".into()); }
    if matches!(m.weights, WeightFormat::Nvfp4) {
        notes.push("NVFP4 weights: decode GEMV dequantizes in registers (W4A16-style); prefill should use mma.sync kind::mxf4nvf4 with on-the-fly activation quantization to actually use FP4 tensor cores".into());
    }
    if weight_bytes + kv > usable { notes.push("model + KV at requested ctx exceeds one Spark; plan tensor/pipeline parallel over ConnectX-7".into()); }

    let mut gemv: Vec<GemvPlan> = ls.iter().map(|l| plan_gemv(l, t)).collect();
    let lm = Linear { name: "lm_head".into(), n: m.vocab, k: m.hidden, copies: 1, active: 1, format: WeightFormat::Bf16 };
    gemv.push(plan_gemv(&lm, t));
    let mut gemm: Vec<GemmTile> = ls.iter()
        .filter(|l| l.copies == 1)
        .map(|l| plan_gemm(l, o.prefill_tokens, t)).collect();
    gemm.push(plan_gemm(&lm, o.prefill_tokens, t));
    if let Some(e) = &m.moe {
        // Average tokens per expert during prefill drives the grouped-GEMM tile choice.
        let per_expert = (o.prefill_tokens as u64 * e.experts_per_tok as u64 / e.num_experts as u64).max(1) as u32;
        for l in ls.iter().filter(|l| l.copies > 1) {
            gemm.push(plan_gemm(l, per_expert, t));
        }
        notes.push(format!("MoE prefill: ~{} tokens/expert at {} prompt tokens -> grouped GEMM tiles chosen for that M", per_expert, o.prefill_tokens));
    }

    Plan {
        target: t.clone(), model: m.clone(), linears: ls, gemv, gemm,
        attn: plan_attn(m, t, o.ctx), norm: plan_norm(m.hidden, t),
        budget: Budget {
            weight_gib: weight_bytes / (1u64 << 30) as f64,
            kv_gib_at_ctx: kv / (1u64 << 30) as f64,
            ctx: o.ctx, fits: weight_bytes + kv <= usable, max_ctx_that_fits: max_ctx,
            decode_bytes_per_token_gb: decode_bytes / 1e9, decode_tok_s_ceiling: tok_s,
        },
        notes,
    }
}

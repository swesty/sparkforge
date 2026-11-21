//! The planner: turns a ModelSpec into concrete per-shape kernel decisions for GB10.
//! Every number here is derived from the config + target, nothing is hand-tuned per model.
use crate::config::{Hybrid, LayerKind, ModelSpec, WeightFormat, TEXT_PREFIXES};
use crate::target::Target;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeSet;

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
    /// Instances read per decode token: the layers that contain this linear (or 1 for once-per-token
    /// linears such as `lm_head`). Internal; the hybrid `ops` list carries it in the plan JSON.
    #[serde(skip)]
    pub per_token: u32,
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

/// One distinct kernel's worth of work in a decode step: identical shape and format in every
/// instance, so one generated kernel serves all of them.
#[derive(Debug, Clone, Serialize)]
pub struct OpClass {
    pub name: String,
    /// `gemv`, `elementwise`, `recurrence`, `attention`, `gather` or `select`.
    pub kind: &'static str,
    /// Budget group the bytes count toward (for example `gdn_projections`, `hyperconnection`).
    pub group: &'static str,
    pub shape: String,
    pub format: Option<WeightFormat>,
    pub per_token: u32,
    /// Bytes read (and, for recurrent state, written) per decode token by all instances, excluding
    /// context-dependent reads.
    pub bytes_per_token: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Budget {
    pub weight_gib: f64,
    pub kv_gib_at_ctx: f64,
    pub ctx: u32,
    pub fits: bool,
    pub max_ctx_that_fits: u64,
    pub decode_bytes_per_token_gb: f64,
    /// Decode bytes per token that do not depend on context length (weights and recurrent state).
    /// Reported for hybrid plans.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_bytes_per_token_ctx_free_gb: Option<f64>,
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
    /// Every distinct decode op of a hybrid plan, with shapes and bytes per token.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ops: Vec<OpClass>,
    pub notes: Vec<String>,
}

pub struct PlanOpts { pub prefill_tokens: u32, pub ctx: u32 }

pub fn linears(m: &ModelSpec) -> Vec<Linear> {
    if let Some(h) = &m.hybrid {
        return hybrid_linears(m, h, &Formats::new(m));
    }
    let f = m.weights;
    let mut v = vec![
        Linear { name: "qkv_proj".into(), n: m.qkv_dim(), k: m.hidden, copies: 1, active: 1, format: f, per_token: m.layers },
        Linear { name: "o_proj".into(), n: m.hidden, k: m.q_dim(), copies: 1, active: 1, format: f, per_token: m.layers },
    ];
    match &m.moe {
        None => {
            v.push(Linear { name: "gate_up_proj".into(), n: 2 * m.intermediate, k: m.hidden, copies: 1, active: 1, format: f, per_token: m.layers });
            v.push(Linear { name: "down_proj".into(), n: m.hidden, k: m.intermediate, copies: 1, active: 1, format: f, per_token: m.layers });
        }
        Some(e) => {
            v.push(Linear { name: "router".into(), n: e.num_experts, k: m.hidden, copies: 1, active: 1, format: WeightFormat::Bf16, per_token: m.layers });
            v.push(Linear { name: "expert_gate_up".into(), n: 2 * e.expert_intermediate, k: m.hidden, copies: e.num_experts, active: e.experts_per_tok, format: f, per_token: m.layers });
            v.push(Linear { name: "expert_down".into(), n: m.hidden, k: e.expert_intermediate, copies: e.num_experts, active: e.experts_per_tok, format: f, per_token: m.layers });
            if e.shared_intermediate > 0 {
                let si = e.shared_intermediate;
                v.push(Linear { name: "shared_gate_up".into(), n: 2 * si, k: m.hidden, copies: 1, active: 1, format: f, per_token: m.layers });
                v.push(Linear { name: "shared_down".into(), n: m.hidden, k: si, copies: 1, active: 1, format: f, per_token: m.layers });
            }
        }
    }
    v
}

fn ceil_div(a: u32, b: u32) -> u32 { a.div_ceil(b) }

/// Per-module formats for the planner. With a checkpoint (`ModelSpec::resolved`), every lookup is
/// exact: the format of the modules the checkpoint holds at that path, as the map's entries cover
/// them. Without one, the map is matched by path, and the entries used are remembered so the plan
/// can report entries it never consumed (a path layout it does not understand would otherwise fall
/// back to BF16 silently).
pub struct Formats<'a> {
    m: &'a ModelSpec,
    used: RefCell<BTreeSet<String>>,
}

impl<'a> Formats<'a> {
    pub fn new(m: &'a ModelSpec) -> Self { Formats { m, used: RefCell::new(BTreeSet::new()) } }

    /// Every matching entry counts as used (a map can list one module under two paths). Matches
    /// that disagree resolve to the largest format.
    fn find(&self, pred: impl Fn(&str) -> bool) -> Option<WeightFormat> {
        let mut best: Option<WeightFormat> = None;
        for (k, f) in self.m.formats.modules.iter().filter(|(k, _)| pred(k)) {
            self.used.borrow_mut().insert(k.clone());
            if best.is_none_or(|b| f.bytes_per_elem() > b.bytes_per_elem()) { best = Some(*f); }
        }
        best
    }

    /// Format of `rel` (for example `linear_attn.in_proj_qkv`) in main-model layer `layer`. Only
    /// language-model paths match, so another tower's `layers.N` (vision, MTP) never does.
    pub fn layer(&self, layer: u32, rel: &str) -> WeightFormat {
        let tail = format!("layers.{layer}.{rel}");
        if let Some(r) = &self.m.resolved {
            return r.format_at(&tail, self.m.weights).unwrap_or(self.m.weights);
        }
        self.find(|k| TEXT_PREFIXES.iter().any(|p| k.strip_prefix(p) == Some(tail.as_str()))).unwrap_or(self.m.weights)
    }

    /// The formats of a module class across `layers`, with how many layers use each, most common
    /// first. Each format becomes its own op class, so the budget counts every instance at its own size.
    pub fn class(&self, layers: &[u32], rel: &str) -> Vec<(WeightFormat, u32)> {
        let mut groups: Vec<(WeightFormat, u32)> = vec![];
        for &l in layers {
            let f = self.layer(l, rel);
            match groups.iter_mut().find(|(g, _)| *g == f) {
                Some(g) => g.1 += 1,
                None => groups.push((f, 1)),
            }
        }
        groups.sort_by_key(|g| std::cmp::Reverse(g.1));
        groups
    }

    /// Format of `rel` in the MTP draft layers. With a checkpoint, only the layer indices it
    /// holds; without one, only indices below the configured MTP layer count, so a stale entry such
    /// as `mtp.layers.999.<rel>` never prices the real draft layer.
    pub fn mtp(&self, rel: &str) -> WeightFormat {
        if let Some(r) = &self.m.resolved {
            return r.table.children("mtp.layers").into_iter()
                .filter_map(|n| r.format_at(&format!("mtp.layers.{n}.{rel}"), self.m.weights))
                .reduce(|a, b| if b.bytes_per_elem() > a.bytes_per_elem() { b } else { a })
                .unwrap_or(self.m.weights);
        }
        let layers = self.m.hybrid.as_ref().map_or(0, |h| h.mtp_layers);
        // Exactly `mtp.layers.<n>.<rel>`, so `mtp.layers.0.aux.<rel>` is not the module.
        self.find(|k| {
            k.strip_prefix("mtp.layers.")
                .and_then(|r| r.split_once('.'))
                .is_some_and(|(n, rest)| n.parse::<u32>().is_ok_and(|n| n < layers) && n.bytes().all(|b| b.is_ascii_digit()) && rest == rel)
        }).unwrap_or(self.m.weights)
    }

    /// Format of a PLE module `rel` at its language-model paths, under a text prefix: `ple.<rel>`,
    /// `<rel>` itself when `bare` (the distinctive table name), or `layers.<n>.ple.<rel>` where `n`
    /// is a configured PLE layer `p` or `p - 1`. The checkpoint files the PLE applied before layer 2
    /// under `layers.1.ple`, so both are accepted; any other layer is a stale entry, left unused.
    pub fn ple(&self, rel: &str, bare: bool) -> WeightFormat {
        let want = format!("ple.{rel}");
        let layers: Vec<String> = self.m.hybrid.as_ref().map_or(vec![], |h| {
            h.ple.layer_ids.iter().flat_map(|&p| [Some(p), p.checked_sub(1)]).flatten()
                .map(|n| format!("layers.{n}.{want}")).collect()
        });
        if let Some(r) = &self.m.resolved {
            // Only paths where the checkpoint holds the module; an alias elsewhere is unused.
            let mut paths = layers.clone();
            paths.push(want.clone());
            if bare { paths.push(rel.to_owned()); }
            return paths.iter().filter_map(|p| r.format_at(p, self.m.weights))
                .reduce(|a, b| if b.bytes_per_elem() > a.bytes_per_elem() { b } else { a })
                .unwrap_or(self.m.weights);
        }
        self.find(|k| TEXT_PREFIXES.iter().filter_map(|p| k.strip_prefix(p)).any(|r| {
            r == want || (bare && r == rel) || layers.iter().any(|l| r == l)
        })).unwrap_or(self.m.weights)
    }

    /// Format of the output head, at its model-level paths (`lm_head`, `model.lm_head`, ...),
    /// never by suffix inside a layer. A tied head is the embedding matrix, so it has the
    /// embedding's format (codex-review of swesty/sparkforge#10).
    pub fn head(&self) -> WeightFormat {
        if let Some(r) = &self.m.resolved {
            if let Some(f) = r.format_at("lm_head", self.m.weights) {
                return f;
            }
            return if self.m.tie_embeddings { self.embed() } else { self.m.weights };
        }
        match self.find(|k| TEXT_PREFIXES.iter().any(|p| k.strip_prefix(p) == Some("lm_head"))) {
            Some(f) => f,
            None if self.m.tie_embeddings => self.embed(),
            None => self.m.weights,
        }
    }

    /// Format of the input embedding (`embed_tokens` under a text prefix).
    pub fn embed(&self) -> WeightFormat {
        if let Some(r) = &self.m.resolved {
            return r.format_at("embed_tokens", self.m.weights).unwrap_or(self.m.weights);
        }
        self.find(|k| TEXT_PREFIXES.iter().any(|p| k.strip_prefix(p) == Some("embed_tokens"))).unwrap_or(self.m.weights)
    }

    pub fn unused(&self) -> Vec<String> {
        if let Some(r) = &self.m.resolved {
            return r.unused.clone();
        }
        let used = self.used.borrow();
        self.m.formats.modules.keys().filter(|k| !used.contains(*k)).cloned().collect()
    }
}

fn layers_of(h: &Hybrid, k: LayerKind) -> Vec<u32> {
    h.schedule.iter().enumerate().filter(|(_, &x)| x == k).map(|(i, _)| i as u32).collect()
}

fn lin(name: &str, n: u32, k: u32, format: WeightFormat, per_token: u32) -> Linear {
    Linear { name: name.into(), n, k, copies: 1, active: 1, format, per_token }
}

fn format_slug(f: WeightFormat) -> String {
    match f {
        WeightFormat::Bf16 => "bf16".into(),
        WeightFormat::W4A16 { group } => format!("w4a16g{group}"),
        WeightFormat::Nvfp4 => "nvfp4".into(),
        WeightFormat::Nvfp4WeightOnly => "nvfp4wo".into(),
        WeightFormat::Mxfp8 => "mxfp8".into(),
    }
}

/// One linear per format group of a class: the most common format keeps the plain name, the
/// others get a format suffix (for example `gdn_in_proj_qkv_bf16`).
fn class_linears(name: &str, n: u32, k: u32, copies: u32, active: u32, groups: &[(WeightFormat, u32)], notes: &mut Vec<String>) -> Vec<Linear> {
    if groups.len() > 1 {
        notes.push(format!("{name} uses {} weight formats across layers; each is planned as its own op class", groups.len()));
    }
    groups.iter().enumerate().map(|(i, &(format, count))| Linear {
        name: if i == 0 { name.into() } else { format!("{name}_{}", format_slug(format)) },
        n, k, copies, active, format, per_token: count,
    }).collect()
}

/// Linears of the Qwen3.8 Flash Next hybrid (one entry per distinct shape and format).
pub fn hybrid_linears(m: &ModelSpec, h: &Hybrid, f: &Formats) -> Vec<Linear> {
    hybrid_linears_noted(m, h, f, &mut vec![])
}

fn hybrid_linears_noted(m: &ModelSpec, h: &Hybrid, f: &Formats, notes: &mut Vec<String>) -> Vec<Linear> {
    let gdn = layers_of(h, LayerKind::LinearAttention);
    let qsa = layers_of(h, LayerKind::FullAttention);
    let all: Vec<u32> = (0..m.layers).collect();
    let (g, q) = (&h.gdn, &h.qsa);
    let res = h.hc.streams * m.hidden;
    let mixers = h.hc_mixers();
    let bf = WeightFormat::Bf16;
    // HyperConnection: down projection merged with the 4 injection logits for the next combine
    // (all but the final mixer), and the up projection whose epilogue does the gated mean.
    let mut v = vec![
        lin("hc_down_inject", h.hc.lowrank + h.hc.streams, res, bf, mixers - 1),
        lin("hc_down_final", h.hc.lowrank, res, bf, 1),
        lin("hc_up", res, h.hc.lowrank, bf, mixers),
    ];
    let class = |name: &str, n: u32, k: u32, layers: &[u32], rel: &str, notes: &mut Vec<String>| {
        class_linears(name, n, k, 1, 1, &f.class(layers, rel), notes)
    };
    v.extend(class("gdn_in_proj_qkv", g.qkv_dim(), m.hidden, &gdn, "linear_attn.in_proj_qkv", notes));
    v.extend(class("gdn_in_proj_z", g.value_dim(), m.hidden, &gdn, "linear_attn.in_proj_z", notes));
    v.extend(class("gdn_in_proj_a", g.value_heads, m.hidden, &gdn, "linear_attn.in_proj_a", notes));
    v.extend(class("gdn_in_proj_b", g.value_heads, m.hidden, &gdn, "linear_attn.in_proj_b", notes));
    v.extend(class("gdn_out_proj", m.hidden, g.value_dim(), &gdn, "linear_attn.out_proj", notes));
    v.extend(class("qsa_q_proj", m.q_dim() * (1 + u32::from(q.output_gate)), m.hidden, &qsa, "self_attn.q_proj", notes));
    v.extend(class("qsa_k_proj", m.kv_dim(), m.hidden, &qsa, "self_attn.k_proj", notes));
    v.extend(class("qsa_v_proj", m.kv_dim(), m.hidden, &qsa, "self_attn.v_proj", notes));
    v.extend(class("qsa_index_qk_proj", q.index_qk_dim(), m.hidden, &qsa, "self_attn.indexer.index_qk_proj", notes));
    v.extend(class("qsa_o_proj", m.hidden, m.q_dim(), &qsa, "self_attn.o_proj", notes));
    if let Some(e) = &m.moe {
        v.extend(class("router", e.num_experts, m.hidden, &all, "mlp.gate", notes));
        let ef = f.class(&all, "mlp.experts");
        v.extend(class_linears("expert_gate_up", 2 * e.expert_intermediate, m.hidden, e.num_experts, e.experts_per_tok, &ef, notes));
        v.extend(class_linears("expert_down", m.hidden, e.expert_intermediate, e.num_experts, e.experts_per_tok, &ef, notes));
        if e.shared_intermediate > 0 {
            // gate and up fuse into one GEMV only where both are in the same format; otherwise each
            // is planned (and counted) on its own.
            let same = all.iter().all(|&l| f.layer(l, "mlp.shared_expert.gate_proj") == f.layer(l, "mlp.shared_expert.up_proj"));
            if same {
                v.extend(class("shared_gate_up", 2 * e.shared_intermediate, m.hidden, &all, "mlp.shared_expert.gate_proj", notes));
                f.class(&all, "mlp.shared_expert.up_proj");
            } else {
                notes.push("shared expert gate_proj and up_proj differ in format in some layers; they are planned as separate GEMVs".into());
                v.extend(class("shared_gate", e.shared_intermediate, m.hidden, &all, "mlp.shared_expert.gate_proj", notes));
                v.extend(class("shared_up", e.shared_intermediate, m.hidden, &all, "mlp.shared_expert.up_proj", notes));
            }
            v.extend(class("shared_down", m.hidden, e.shared_intermediate, &all, "mlp.shared_expert.down_proj", notes));
            v.extend(class("shared_expert_gate", 1, m.hidden, &all, "mlp.shared_expert_gate", notes));
        }
    }
    // PLE projections run once per token (at the PLE layer). key_proj widens to the HC residual.
    v.push(lin("ple_key_proj", res, h.ple.embed_dim, f.ple("key_proj", false), 1));
    v.push(lin("ple_value_proj", m.hidden, h.ple.embed_dim, f.ple("value_proj", false), 1));
    v
}

/// Non-GEMV decode ops of the hybrid, with their context-free bytes per token.
fn hybrid_other_ops(m: &ModelSpec, h: &Hybrid, f: &Formats) -> Vec<OpClass> {
    let (g, q) = (&h.gdn, &h.qsa);
    let res = (h.hc.streams * m.hidden) as f64;
    let ngdn = h.count(LayerKind::LinearAttention);
    let nqsa = h.count(LayerKind::FullAttention);
    let table = f.ple("ple_embedding.ngram_embedding", true);
    let op = |name: &str, kind, group, shape: String, format, per_token: u32, bytes_each: f64| OpClass {
        name: name.into(), kind, group, shape, format, per_token, bytes_per_token: bytes_each * per_token as f64,
    };
    vec![
        op("embed_gather", "gather", "other", format!("1 row of {}", m.hidden), Some(f.embed()), 1, m.hidden as f64 * f.embed().bytes_per_elem()),
        op("ple_ngram_gather", "gather", "ple", format!("{} rows of {} ({} heads, orders 2..={})", h.ple.heads(), h.ple.head_dim(), h.ple.heads(), h.ple.ngram_size),
            Some(table), 1, (h.ple.heads() * h.ple.head_dim()) as f64 * table.bytes_per_elem()),
        // conv1d weights (BF16) and three F32 norms of the residual width.
        op("ple_conv_norm_apply", "elementwise", "ple", format!("{} channels, conv kernel {}", res, h.ple.conv_kernel), None, 1,
            res * h.ple.conv_kernel as f64 * 2.0 + 3.0 * res * 4.0),
        // Combine with the injection logits, then a grouped RMSNorm per stream (F32 weights).
        op("hc_combine_norm", "elementwise", "hyperconnection", format!("{} streams x {}", h.hc.streams, m.hidden), None, h.hc_mixers() - 1, res * 4.0),
        op("hc_final_norm", "elementwise", "hyperconnection", format!("{} streams x {}", h.hc.streams, m.hidden), None, 1, res * 4.0),
        op("gdn_conv1d", "elementwise", "gdn_state", format!("{} channels, kernel {}", g.qkv_dim(), g.conv_kernel), None, ngdn,
            2.0 * g.conv_state_bytes() as f64 + g.qkv_dim() as f64 * g.conv_kernel as f64 * 2.0),
        // FP32 state read and written once per token.
        op("gdn_recurrence", "recurrence", "gdn_state", format!("{} heads x {}x{} f32", g.value_heads, g.key_head_dim, g.value_head_dim), None, ngdn,
            2.0 * g.state_bytes() as f64),
        op("gdn_gated_norm", "elementwise", "other", format!("{} heads x {}", g.value_heads, g.value_head_dim), None, ngdn, g.value_head_dim as f64 * 4.0),
        op("qsa_qk_norm_rope_kv_append", "elementwise", "other",
            format!("{} q + {} kv heads x {}, rotate {} dims", m.heads, m.kv_heads, m.head_dim, (m.head_dim as f64 * h.mrope.partial_rotary_factor).round()),
            None, nqsa, 2.0 * m.head_dim as f64 * 4.0),
        op("qsa_indexer_score_topk", "select", "qsa_context", format!("{} heads x {}, top {} groups of {}", q.indexer_heads, q.indexer_head_dim, q.budget / q.compress_ratio, q.compress_ratio),
            None, nqsa, 0.0),
        op("qsa_sparse_attention", "attention", "qsa_context", format!("{} heads / {} kv x {}, <= {} positions", m.heads, m.kv_heads, m.head_dim, q.selected_positions()),
            None, nqsa, 0.0),
        op("moe_route_topk", "select", "other", format!("top {} of {}", m.moe.as_ref().map_or(0, |e| e.experts_per_tok), m.moe.as_ref().map_or(0, |e| e.num_experts)),
            None, m.layers, 0.0),
    ]
}

/// Resident non-GEMV parameters of the hybrid, by component (bytes): norm weights, conv filters
/// and per-head gate parameters. Recurrent state is fixed memory, not a parameter.
pub fn hybrid_param_bytes(m: &ModelSpec, h: &Hybrid) -> Vec<(&'static str, f64)> {
    let (g, q) = (&h.gdn, &h.qsa);
    let res = (h.hc.streams * m.hidden) as f64;
    let ngdn = h.count(LayerKind::LinearAttention) as f64;
    let nqsa = h.count(LayerKind::FullAttention) as f64;
    vec![
        ("hc_norms", h.hc_mixers() as f64 * res * 4.0),
        ("ple_conv_and_norms", res * h.ple.conv_kernel as f64 * 2.0 + 3.0 * res * 4.0),
        ("gdn_conv_filters", ngdn * g.qkv_dim() as f64 * g.conv_kernel as f64 * 2.0),
        ("gdn_norm_and_gates", ngdn * (g.value_head_dim as f64 * 4.0 + 2.0 * g.value_heads as f64 * 4.0)),
        ("qsa_norms", nqsa * (2.0 * m.head_dim as f64 * 4.0 + 2.0 * q.indexer_head_dim as f64 * 4.0)),
    ]
}

/// FP8 KV and indexer-key bytes per context token, across all sparse-attention layers.
fn hybrid_kv_bytes_per_token(m: &ModelSpec, h: &Hybrid) -> f64 {
    let q = &h.qsa;
    let per_layer = 2.0 * m.kv_dim() as f64 + (q.indexer_kv_heads * q.indexer_head_dim) as f64 / q.compress_ratio as f64;
    per_layer * h.count(LayerKind::FullAttention) as f64
}

/// Context-dependent decode reads of the hybrid at `ctx`: gathered KV for the selected positions,
/// and the indexer's scan over every compressed key.
fn hybrid_ctx_bytes(m: &ModelSpec, h: &Hybrid, ctx: u32) -> f64 {
    let q = &h.qsa;
    let gather = ctx.min(q.selected_positions()) as f64 * 2.0 * m.kv_dim() as f64;
    let scan = ctx.div_ceil(q.compress_ratio) as f64 * (q.indexer_kv_heads * q.indexer_head_dim) as f64;
    (gather + scan) * h.count(LayerKind::FullAttention) as f64
}

/// Resident bytes of the MTP draft layers: one sparse-attention layer with its own experts, plus
/// the input projections and three HyperConnection mixers.
fn mtp_weight_bytes(m: &ModelSpec, h: &Hybrid, f: &Formats) -> f64 {
    if h.mtp_layers == 0 { return 0.0; }
    let hd = m.hidden as f64;
    let res = (h.hc.streams * m.hidden) as f64;
    let e = m.moe.as_ref();
    let b = |rel: &str| f.mtp(rel).bytes_per_elem();
    let attn = (m.q_dim() * 2) as f64 * hd * b("self_attn.q_proj") + m.kv_dim() as f64 * hd * (b("self_attn.k_proj") + b("self_attn.v_proj"))
        + h.qsa.index_qk_dim() as f64 * hd * b("self_attn.indexer.index_qk_proj") + hd * m.q_dim() as f64 * b("self_attn.o_proj");
    let experts = e.map_or(0.0, |e| e.num_experts as f64 * 3.0 * e.expert_intermediate as f64 * hd * b("mlp.experts"));
    // Shared-expert formats are looked up only when the shared expert exists, so a stale entry for
    // an absent projection stays reported as unused. The router is counted either way.
    let shared = e.map_or(0.0, |e| {
        let projections = if e.shared_intermediate > 0 {
            e.shared_intermediate as f64 * hd
                * (b("mlp.shared_expert.gate_proj") + b("mlp.shared_expert.up_proj") + b("mlp.shared_expert.down_proj"))
        } else {
            0.0
        };
        projections + e.num_experts as f64 * hd * b("mlp.gate")
    });
    let fc = 2.0 * hd * hd * 2.0;
    let mixer = |inject: bool| ((h.hc.lowrank + if inject { h.hc.streams } else { 0 }) as f64 * res + res * h.hc.lowrank as f64) * 2.0 + res * 4.0;
    h.mtp_layers as f64 * (attn + experts + shared + fc + 2.0 * mixer(true)) + mixer(false)
}

fn linear_op(l: &Linear, group: &'static str) -> OpClass {
    let shape = if l.copies > 1 { format!("N={} K={} ({} of {} copies)", l.n, l.k, l.active, l.copies) } else { format!("N={} K={}", l.n, l.k) };
    OpClass { name: l.name.clone(), kind: "gemv", group, shape, format: Some(l.format), per_token: l.per_token,
        bytes_per_token: l.bytes_one() * l.active as f64 * l.per_token as f64 }
}

fn group_of(name: &str) -> &'static str {
    match name {
        n if n.starts_with("hc_") => "hyperconnection",
        n if n.starts_with("gdn_") => "gdn_projections",
        n if n.starts_with("qsa_") => "qsa_projections",
        n if n.starts_with("expert_") => "routed_experts",
        n if n.starts_with("shared_") => "shared_experts",
        n if n.starts_with("ple_") => "ple",
        n if n.starts_with("router") => "routers",
        "lm_head" => "lm_head",
        _ => "dense",
    }
}

/// Context-free decode bytes per token, summed by budget group, largest first.
pub fn bytes_by_group(p: &Plan) -> Vec<(&'static str, f64)> {
    let mut g: Vec<(&'static str, f64)> = vec![];
    for o in &p.ops {
        match g.iter_mut().find(|(k, _)| *k == o.group) {
            Some(e) => e.1 += o.bytes_per_token,
            None => g.push((o.group, o.bytes_per_token)),
        }
    }
    g.retain(|(_, b)| *b > 0.0);
    g.sort_by(|a, b| b.1.total_cmp(&a.1));
    g
}


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
    let formats = Formats::new(m);
    let ls = match &m.hybrid {
        Some(h) => hybrid_linears_noted(m, h, &formats, &mut notes),
        None => linears(m),
    };
    let lm = Linear { name: "lm_head".into(), n: m.vocab, k: m.hidden, copies: 1, active: 1,
        format: if m.hybrid.is_some() { formats.head() } else { WeightFormat::Bf16 }, per_token: 1 };
    // The embedding at its own format (BF16 unless a mixed-precision map quantizes it).
    let emb = m.vocab as f64 * m.hidden as f64 * if m.hybrid.is_some() { formats.embed().bytes_per_elem() } else { 2.0 };
    let usable = t.usable_mem_gib * (1u64 << 30) as f64;
    let mut ops: Vec<OpClass> = vec![];
    if m.hybrid.is_some() {
        ops = ls.iter().map(|l| linear_op(l, group_of(&l.name))).collect();
        ops.push(linear_op(&lm, "lm_head"));
    }

    let (weight_bytes, fixed_state, kv_per_token, ctx_bytes, decode_ctx_free) = match &m.hybrid {
        None => {
            let total: f64 = ls.iter().map(|l| l.bytes_one() * l.copies as f64 * l.per_token as f64).sum();
            let active: f64 = ls.iter().map(|l| l.bytes_one() * l.active as f64 * l.per_token as f64).sum();
            let head = if m.tie_embeddings { 0.0 } else { emb };
            // decode reads active weights, lm_head (the embedding when tied) and the whole KV cache
            let kvt = m.kv_bytes_per_token() as f64;
            (total + emb + head, 0.0, kvt, kvt * o.ctx as f64, active + emb)
        }
        Some(h) => {
            ops.extend(hybrid_other_ops(m, h, &formats));
            let linear_total: f64 = ls.iter().chain(std::iter::once(&lm)).map(|l| l.bytes_one() * l.copies as f64 * l.per_token as f64).sum();
            let table = formats.ple("ple_embedding.ngram_embedding", true);
            let ple_table = h.ple.table_rows() as f64 * h.ple.head_dim() as f64 * table.bytes_per_elem();
            let small: f64 = hybrid_param_bytes(m, h).iter().map(|(_, b)| b).sum();
            // A tied lm_head is the embedding matrix itself; count it once.
            let lm_resident = if m.tie_embeddings { lm.bytes_one() } else { 0.0 };
            let weights = linear_total - lm_resident + emb + ple_table + mtp_weight_bytes(m, h, &formats) + small;
            let fixed = h.count(LayerKind::LinearAttention) as f64 * (h.gdn.state_bytes() + h.gdn.conv_state_bytes()) as f64;
            let decode: f64 = ops.iter().map(|o| o.bytes_per_token).sum();
            notes.push(format!("{}: every decode op in this plan is planned, not generated; kernels arrive with the milestone M2 tickets, and `emit`/`verify` produce no kernels for this model yet", m.arch));
            notes.push(format!("schedule: {} Gated DeltaNet + {} sparse-attention layers, {} HyperConnection mixers per token, PLE at layer(s) {:?}, {} MTP layer(s)",
                h.count(LayerKind::LinearAttention), h.count(LayerKind::FullAttention), h.hc_mixers(), h.ple.layer_ids, h.mtp_layers));
            notes.push(format!("PLE table {:.1} GB counts as resident; kept on disk the weights are {:.1} GB", ple_table / 1e9, (weights - ple_table) / 1e9));
            notes.push("KV cache and indexer keys are planned as FP8 (the checkpoint sets no kv_cache_quant_algo); the indexer cache dtype is assumed FP8 until checked against the reference".into());
            notes.push(format!("decode budget excludes the MTP draft layers ({:.2} GB resident); speculative steps are planned with the inference-engine change", mtp_weight_bytes(m, h, &formats) / 1e9));
            notes.push("router scoring (softmax or sigmoid) and top-k renormalization are not in the config; read them from the reference before generating the router kernel".into());
            let unused = formats.unused();
            if !unused.is_empty() {
                notes.push(format!("{} quantized_layers entr{} not used by the plan (no module of this model at that path): {}",
                    unused.len(), if unused.len() == 1 { "y is" } else { "ies are" }, unused.join(", ")));
            }
            (weights, fixed, hybrid_kv_bytes_per_token(m, h), hybrid_ctx_bytes(m, h, o.ctx), decode)
        }
    };
    let kv = kv_per_token * o.ctx as f64 + fixed_state;
    let max_ctx = ((usable - weight_bytes - fixed_state).max(0.0) / kv_per_token) as u64;
    let decode_bytes = decode_ctx_free + ctx_bytes;
    let tok_s = t.dram_bw_effective_gbs * 1e9 / decode_bytes;

    if m.hybrid.is_none() {
        if m.head_dim % 32 != 0 { notes.push(format!("head_dim {} not a multiple of 32; attention kernel needs a padded path", m.head_dim)); }
        if m.qk_norm { notes.push("q/k per-head RMSNorm (q_norm/k_norm) is not generated yet: decode output is wrong for this model until it is".into()); }
        if m.qkv_bias { notes.push("q/k/v projection bias is not generated yet: decode output is wrong for this model until it is".into()); }
        if matches!(m.weights, WeightFormat::Nvfp4) {
            notes.push("NVFP4 weights: decode GEMV dequantizes in registers (W4A16-style); prefill should use mma.sync kind::mxf4nvf4 with on-the-fly activation quantization to actually use FP4 tensor cores".into());
        }
    }
    if m.hidden % 8 != 0 { notes.push("hidden not a multiple of 8; vectorized 16B loads disabled".into()); }
    if weight_bytes + kv > usable { notes.push("model + KV at requested ctx exceeds one Spark; plan tensor/pipeline parallel over ConnectX-7".into()); }

    let mut gemv: Vec<GemvPlan> = ls.iter().map(|l| plan_gemv(l, t)).collect();
    gemv.push(plan_gemv(&lm, t));
    for g in gemv.iter().filter(|g| g.smem_bytes > t.smem_per_block_optin) {
        notes.push(format!("gemv {} (K={}) needs {} B of shared memory for its activation vector, above the {} B per-block limit; it is not generated until a non-staged GEMV variant exists", g.linear, g.k, g.smem_bytes, t.smem_per_block_optin));
    }
    let mut gemm = vec![];
    if m.hybrid.is_some() {
        notes.push("prefill GEMM tiles are not planned for this model yet (MXFP8 and NVFP4 prefill arrive with milestone M7)".into());
    } else {
        gemm = ls.iter().filter(|l| l.copies == 1).map(|l| plan_gemm(l, o.prefill_tokens, t)).collect();
        gemm.push(plan_gemm(&lm, o.prefill_tokens, t));
        if let Some(e) = &m.moe {
            // Average tokens per expert during prefill drives the grouped-GEMM tile choice.
            let per_expert = (o.prefill_tokens as u64 * e.experts_per_tok as u64 / e.num_experts as u64).max(1) as u32;
            for l in ls.iter().filter(|l| l.copies > 1) {
                gemm.push(plan_gemm(l, per_expert, t));
            }
            notes.push(format!("MoE prefill: ~{} tokens/expert at {} prompt tokens -> grouped GEMM tiles chosen for that M", per_expert, o.prefill_tokens));
        }
    }
    // Sparse attention reads at most the selected positions, so its splits are sized for those.
    let attn_ctx = m.hybrid.as_ref().map_or(o.ctx, |h| o.ctx.min(h.qsa.selected_positions()));

    Plan {
        target: t.clone(), model: m.clone(), linears: ls, gemv, gemm,
        attn: plan_attn(m, t, attn_ctx), norm: plan_norm(m.hidden, t),
        budget: Budget {
            weight_gib: weight_bytes / (1u64 << 30) as f64,
            kv_gib_at_ctx: kv / (1u64 << 30) as f64,
            ctx: o.ctx, fits: weight_bytes + kv <= usable, max_ctx_that_fits: max_ctx,
            decode_bytes_per_token_gb: decode_bytes / 1e9,
            decode_bytes_per_token_ctx_free_gb: m.hybrid.as_ref().map(|_| decode_ctx_free / 1e9),
            decode_tok_s_ceiling: tok_s,
        },
        ops,
        notes,
    }
}

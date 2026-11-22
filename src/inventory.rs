//! The tensors a hybrid (`qwen4_exp`, Qwen3.8 Flash Next) checkpoint must hold: one list, derived
//! from the spec and from the formats the planner prices, so the plan and the loader can never
//! disagree about which tensors exist or what they contain.
//!
//! Names are canonical (any language-model prefix removed, see `modules::canonical`). The tensor
//! set of a quantized module follows its format: MXFP8 is E4M3 values plus one UE8M0 scale (`U8`)
//! per 32 inputs; NVFP4 is packed E2M1 pairs (`U8`), one E4M3 scale per 16 inputs, an F32 global
//! scale (`weight_scale_2`) and, for activations quantized at run time, an F32 `input_scale`.
//! Dtypes the config doesn't determine (norms, gate parameters) are those of the reference
//! checkpoint, `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision 6909a5bed089.
use crate::config::{Hybrid, LayerKind, ModelSpec, WeightFormat};
use crate::modules::ModuleTable;
use crate::plan::{Formats, PLE_TABLE};
use anyhow::{bail, ensure, Result};

/// One tensor the plan needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Want {
    /// Canonical tensor name (`layers.0.linear_attn.in_proj_qkv.weight`, `mtp.fc_hidden.weight`).
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    /// What the bytes are for, to report a checkpoint's bytes by group.
    pub group: &'static str,
}

/// What the inventory needs from the checkpoint beyond the spec.
pub struct Layout {
    /// The canonical path of the PLE module (`layers.1.ple` in the reference checkpoint).
    pub ple: String,
    /// Rows in each PLE table shard; `None` when only names are checked (shapes then hold 0 rows).
    pub table_rows: Option<usize>,
}

/// The canonical path of the checkpoint's PLE module: `layers.<n>.ple` with `n` the configured PLE
/// layer `p` or `p - 1`, or a model-level `ple`. Exactly one may hold modules, and a model-level
/// table under its bare name (`ple_embedding.ngram_embedding`) counts as a second location: the
/// planner prices one PLE and one table (codex-review of #17).
pub fn ple_path(table: &ModuleTable, h: &Hybrid) -> Result<String> {
    let mut candidates: Vec<String> = h.ple.layer_ids.iter().flat_map(|&p| [Some(p), p.checked_sub(1)]).flatten()
        .map(|n| format!("layers.{n}.ple")).collect();
    candidates.dedup();
    candidates.push("ple".into());
    let mut live: Vec<String> = candidates.iter().filter(|c| table.has(c)).cloned().collect();
    if table.has(PLE_TABLE) {
        live.push(PLE_TABLE.into());
    }
    match live.as_slice() {
        [one] if one != PLE_TABLE => Ok(one.clone()),
        [] | [_] => bail!("the checkpoint holds no PLE module at {}; the plan needs its projections and n-gram table", candidates.join(", ")),
        _ => bail!("the checkpoint holds PLE modules at {}; the planner prices one PLE and one n-gram table, so only one location may hold them", live.join(" and ")),
    }
}

/// Every tensor a hybrid checkpoint must hold for `m`, with its dtype and shape. `f` must be the
/// planner's view of the same spec, so each module's tensor set follows the format it is priced at.
pub fn hybrid(m: &ModelSpec, f: &Formats, layout: &Layout) -> Result<Vec<Want>> {
    let Some(h) = &m.hybrid else { bail!("{} is not a hybrid model", m.arch) };
    let mut inv = Inventory(vec![]);
    let (hd, res) = (m.hidden as usize, (h.hc.streams * m.hidden) as usize);
    let (g, q) = (&h.gdn, &h.qsa);
    inv.gather("embed_tokens", m.vocab, m.hidden, f.embed(), "embed")?;
    if !m.tie_embeddings {
        inv.projection("lm_head", m.vocab, m.hidden, f.head(), "lm_head")?;
    }
    inv.mixer("hyper_connection_mixer", h, res, false, "F32");
    for (i, &kind) in h.schedule.iter().enumerate() {
        let l = i as u32;
        let p = format!("layers.{i}");
        match kind {
            LayerKind::LinearAttention => {
                let a = format!("{p}.linear_attn");
                for (rel, n, k) in [("in_proj_qkv", g.qkv_dim(), m.hidden), ("in_proj_z", g.value_dim(), m.hidden), ("in_proj_a", g.value_heads, m.hidden),
                    ("in_proj_b", g.value_heads, m.hidden), ("out_proj", m.hidden, g.value_dim())] {
                    inv.projection(&format!("{a}.{rel}"), n, k, f.layer(l, &format!("linear_attn.{rel}")), "gdn_projections")?;
                }
                inv.tensor(format!("{a}.A_log"), "BF16", vec![g.value_heads as usize], "params");
                inv.tensor(format!("{a}.dt_bias"), "BF16", vec![g.value_heads as usize], "params");
                inv.tensor(format!("{a}.conv1d.weight"), "BF16", vec![g.qkv_dim() as usize, 1, g.conv_kernel as usize], "params");
                inv.tensor(format!("{a}.norm.weight"), "BF16", vec![g.value_head_dim as usize], "params");
            }
            LayerKind::FullAttention => inv.attention(&format!("{p}.self_attn"), m, q, |rel| f.layer(l, rel), "qsa_projections")?,
        }
        inv.mlp(&p, m, |rel| f.layer(l, rel), f.layer_group(l, "mlp.experts"))?;
        inv.mixer(&format!("{p}.attn_hyper_connection"), h, res, true, "F32");
        inv.mixer(&format!("{p}.mlp_hyper_connection"), h, res, true, "F32");
    }
    // The PLE: projections widening to the residual, its conv and norms, and the n-gram table.
    let ple = &layout.ple;
    let pd = h.ple.embed_dim;
    inv.projection(&format!("{ple}.key_proj"), h.hc.streams * m.hidden, pd, f.ple("key_proj"), "ple")?;
    inv.projection(&format!("{ple}.value_proj"), m.hidden, pd, f.ple("value_proj"), "ple")?;
    inv.tensor(format!("{ple}.conv1d.weight"), "BF16", vec![res, 1, h.ple.conv_kernel as usize], "ple");
    for norm in ["norm_conv", "norm_key", "norm_query"] {
        inv.tensor(format!("{ple}.{norm}.weight"), "F32", vec![res], "ple");
    }
    let emb = format!("{ple}.ple_embedding");
    inv.tensor(format!("{emb}.layer_multipliers"), "I64", vec![h.ple.ngram_size as usize], "ple");
    inv.tensor(format!("{emb}.ngram_heads_offsets"), "I64", vec![h.ple.heads() as usize], "ple");
    inv.tensor(format!("{emb}.ngram_heads_vocab_sizes"), "I64", vec![h.ple.heads() as usize], "ple");
    let Some(parts) = h.ple.split_parts else {
        bail!("config.json has no split_ngram_parts, so the PLE n-gram table's shards can't be checked");
    };
    let table = format!("{ple}.{PLE_TABLE}");
    let rows = layout.table_rows.unwrap_or(0);
    let tf = f.ple_table();
    if matches!(tf, WeightFormat::Nvfp4 | WeightFormat::Nvfp4WeightOnly) {
        inv.tensor(format!("{table}.weight_scale_2"), "F32", vec![1], "ple_table");
    }
    for j in 0..parts {
        inv.weights(&format!("{table}.shard_{j}"), rows, h.ple.head_dim(), tf, Scales::Shared, "ple_table")?;
    }
    // MTP draft layers: one sparse-attention layer with its own experts, the input projections and
    // three mixers. Their norms are BF16 in the reference checkpoint (F32 in the main model).
    if h.mtp_layers > 0 {
        for fc in ["fc_embedding", "fc_hidden"] {
            inv.tensor(format!("mtp.{fc}.weight"), "BF16", vec![hd, hd], "mtp");
        }
        inv.tensor("mtp.pre_fc_norm_embedding.weight".into(), "BF16", vec![hd], "mtp");
        inv.tensor("mtp.pre_fc_norm_hidden.weight".into(), "BF16", vec![res], "mtp");
        inv.mixer("mtp.hyper_connection_mixer", h, res, false, "BF16");
    }
    for i in 0..h.mtp_layers {
        let p = format!("mtp.layers.{i}");
        let start = inv.0.len();
        inv.attention(&format!("{p}.self_attn"), m, q, |rel| f.mtp(rel), "mtp")?;
        inv.mlp(&p, m, |rel| f.mtp(rel), f.mtp_group("mlp.experts"))?;
        inv.mixer(&format!("{p}.attn_hyper_connection"), h, res, true, "BF16");
        inv.mixer(&format!("{p}.mlp_hyper_connection"), h, res, true, "BF16");
        for w in &mut inv.0[start..] {
            w.group = "mtp";
        }
    }
    Ok(inv.0)
}

struct Inventory(Vec<Want>);

/// Where an NVFP4 matrix's scalar scales live.
#[derive(Clone, Copy)]
enum Scales {
    /// Its own global scale, plus an activation scale when the activations are quantized
    /// (`Nvfp4`; `Nvfp4WeightOnly` keeps them BF16) and the matrix is a GEMV operand.
    Own { activation: bool },
    /// A table shard: the table holds one global scale for every shard.
    Shared,
}

impl Inventory {
    fn tensor(&mut self, name: String, dtype: &'static str, shape: Vec<usize>, group: &'static str) {
        self.0.push(Want { name, dtype, shape, group });
    }

    /// A GEMV weight: the tensor set of `f`, NVFP4 with its activation scale.
    fn projection(&mut self, path: &str, n: u32, k: u32, f: WeightFormat, group: &'static str) -> Result<()> {
        self.weights(path, n as usize, k, f, Scales::Own { activation: true }, group)
    }

    /// A gathered table (the embedding): no activation is quantized, so NVFP4 has no `input_scale`.
    fn gather(&mut self, path: &str, n: u32, k: u32, f: WeightFormat, group: &'static str) -> Result<()> {
        self.weights(path, n as usize, k, f, Scales::Own { activation: false }, group)
    }

    /// The tensors of a `rows` x `k` weight matrix stored at `path` in format `f`.
    fn weights(&mut self, path: &str, rows: usize, k: u32, f: WeightFormat, scales: Scales, group: &'static str) -> Result<()> {
        let k = k as usize;
        match f {
            WeightFormat::Bf16 => self.tensor(format!("{path}.weight"), "BF16", vec![rows, k], group),
            WeightFormat::Mxfp8 => {
                ensure!(k % 32 == 0, "{path} is MXFP8 with {k} inputs, which is not a multiple of its 32-element scale blocks");
                self.tensor(format!("{path}.weight"), "F8_E4M3", vec![rows, k], group);
                self.tensor(format!("{path}.weight_scale"), "U8", vec![rows, k / 32], group);
            }
            WeightFormat::Nvfp4 | WeightFormat::Nvfp4WeightOnly => {
                ensure!(k % 16 == 0, "{path} is NVFP4 with {k} inputs, which is not a multiple of its 16-element scale blocks");
                self.tensor(format!("{path}.weight"), "U8", vec![rows, k / 2], group);
                self.tensor(format!("{path}.weight_scale"), "F8_E4M3", vec![rows, k / 16], group);
                if let Scales::Own { activation } = scales {
                    self.tensor(format!("{path}.weight_scale_2"), "F32", vec![], group);
                    if activation && f == WeightFormat::Nvfp4 {
                        self.tensor(format!("{path}.input_scale"), "F32", vec![], group);
                    }
                }
            }
            WeightFormat::W4A16 { group: g } => bail!("{path} is W4A16 (int4, group {g}): only BF16, MXFP8 and NVFP4 tensors load so far"),
        }
        Ok(())
    }

    /// A gated sparse-attention block: the projections (q carries the output gate's rows), the
    /// per-head q/k norms and the indexer.
    fn attention(&mut self, a: &str, m: &ModelSpec, q: &crate::config::Qsa, f: impl Fn(&str) -> WeightFormat, group: &'static str) -> Result<()> {
        for (rel, n, k) in [("q_proj", m.q_dim() * (1 + u32::from(q.output_gate)), m.hidden), ("k_proj", m.kv_dim(), m.hidden), ("v_proj", m.kv_dim(), m.hidden),
            ("o_proj", m.hidden, m.q_dim()), ("indexer.index_qk_proj", q.index_qk_dim(), m.hidden)] {
            self.projection(&format!("{a}.{rel}"), n, k, f(&format!("self_attn.{rel}")), group)?;
        }
        for norm in ["q_norm", "k_norm"] {
            self.tensor(format!("{a}.{norm}.weight"), "BF16", vec![m.head_dim as usize], "params");
        }
        for norm in ["q_layernorm", "k_layernorm"] {
            self.tensor(format!("{a}.indexer.{norm}.weight"), "BF16", vec![q.indexer_head_dim as usize], "params");
        }
        Ok(())
    }

    /// The router, every routed expert (in the experts' one format) and the shared expert.
    fn mlp(&mut self, p: &str, m: &ModelSpec, f: impl Fn(&str) -> WeightFormat, experts: WeightFormat) -> Result<()> {
        let Some(e) = &m.moe else { bail!("{} has no experts, but the hybrid layout is mixture-of-experts", m.arch) };
        self.projection(&format!("{p}.mlp.gate"), e.num_experts, m.hidden, f("mlp.gate"), "routers")?;
        for x in 0..e.num_experts {
            let ex = format!("{p}.mlp.experts.{x}");
            self.projection(&format!("{ex}.gate_proj"), e.expert_intermediate, m.hidden, experts, "routed_experts")?;
            self.projection(&format!("{ex}.up_proj"), e.expert_intermediate, m.hidden, experts, "routed_experts")?;
            self.projection(&format!("{ex}.down_proj"), m.hidden, e.expert_intermediate, experts, "routed_experts")?;
        }
        if e.shared_intermediate > 0 {
            let s = format!("{p}.mlp.shared_expert");
            for (rel, n, k) in [("gate_proj", e.shared_intermediate, m.hidden), ("up_proj", e.shared_intermediate, m.hidden), ("down_proj", m.hidden, e.shared_intermediate)] {
                self.projection(&format!("{s}.{rel}"), n, k, f(&format!("mlp.shared_expert.{rel}")), "shared_experts")?;
            }
            self.projection(&format!("{p}.mlp.shared_expert_gate"), 1, m.hidden, f("mlp.shared_expert_gate"), "shared_experts")?;
        }
        Ok(())
    }

    /// A HyperConnection mixer: low-rank down and up projections over the residual streams, the
    /// grouped norm, and (except for the final mixer) the injection rows for the next combine.
    fn mixer(&mut self, p: &str, h: &Hybrid, res: usize, inject: bool, norm_dtype: &'static str) {
        let lowrank = h.hc.lowrank as usize;
        if inject {
            self.tensor(format!("{p}.block_inject_weight.weight"), "BF16", vec![h.hc.streams as usize, res], "hyperconnection");
        }
        self.tensor(format!("{p}.hc_norm.weight"), norm_dtype, vec![res], "hyperconnection");
        self.tensor(format!("{p}.input_mix_weight_down.weight"), "BF16", vec![lowrank, res], "hyperconnection");
        self.tensor(format!("{p}.input_mix_weight_up.weight"), "BF16", vec![res, lowrank], "hyperconnection");
    }
}

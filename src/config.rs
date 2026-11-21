//! Hugging Face `config.json` -> normalized ModelSpec.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
struct RawLayerQuant {
    quant_algo: String,
    group_size: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawQuant {
    quant_method: Option<String>,
    quant_algo: Option<String>,
    bits: Option<u32>,
    /// GPTQ uses -1 for per-channel scales, so this must be signed.
    group_size: Option<i64>,
    /// ModelOpt `MIXED_PRECISION`: module path -> that module's scheme. Kept as raw JSON and read
    /// only during format detection, so `--format` can override a map this parser would reject.
    quantized_layers: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    architectures: Vec<String>,
    model_type: Option<String>,
    hidden_size: u32,
    intermediate_size: Option<u32>,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: Option<u32>,
    head_dim: Option<u32>,
    vocab_size: u32,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    rope_theta: Option<f64>,
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    tie_word_embeddings: bool,
    // MoE (names differ across families)
    #[serde(alias = "num_local_experts", alias = "n_routed_experts")]
    num_experts: Option<u32>,
    num_experts_per_tok: Option<u32>,
    moe_intermediate_size: Option<u32>,
    n_shared_experts: Option<u32>,
    shared_expert_intermediate_size: Option<u32>,
    /// Raw JSON, decoded only during format detection: with `--format` it is never interpreted,
    /// so malformed quantization metadata cannot block an override.
    quantization_config: Option<serde_json::Value>,
    rope_scaling: Option<serde_json::Value>,
    /// Newer HF configs put `rope_theta`, `rope_type` and partial-rotary fields here.
    rope_parameters: Option<serde_json::Value>,
    partial_rotary_factor: Option<f64>,
    attention_bias: Option<bool>,
    /// Present only on MLA models (DeepSeek V2/V3), which this engine does not plan.
    kv_lora_rank: Option<u32>,
    /// Per-layer attention kind on hybrid models (`linear_attention` / `full_attention`).
    layer_types: Option<Vec<String>>,
}
fn default_eps() -> f64 { 1e-6 }
fn is_zero(x: &u32) -> bool { *x == 0 }

/// Fields only the `qwen4_exp` (Qwen3.8 Flash Next) hybrid carries. All are required for it.
#[derive(Debug, Deserialize)]
struct RawHybrid {
    full_attention_interval: Option<u32>,
    // Gated DeltaNet (linear attention)
    linear_num_key_heads: Option<u32>,
    linear_num_value_heads: Option<u32>,
    linear_key_head_dim: Option<u32>,
    linear_value_head_dim: Option<u32>,
    linear_conv_kernel_dim: Option<u32>,
    // Sparse full attention (QSA) and its indexer
    output_gate_type: Option<String>,
    indexer_n_heads: Option<u32>,
    indexer_head_dim: Option<u32>,
    indexer_kv_heads: Option<u32>,
    indexer_compress_ratio: Option<u32>,
    indexer_budget: Option<u32>,
    // HyperConnection residual
    hc_count: Option<u32>,
    hc_lowrank: Option<u32>,
    // Per-layer n-gram embedding
    ple_layer_ids: Option<Vec<u32>>,
    ple_embed_dim: Option<u32>,
    ple_conv_kernel_size: Option<u32>,
    ngram_size: Option<u32>,
    heads_per_ngram: Option<u32>,
    ngram_vocab_size_base: Option<u64>,
    // Multi-token prediction draft layers (also given as `mtp.num_hidden_layers`)
    mtp_num_hidden_layers: Option<u32>,
    mtp: Option<RawMtp>,
}

#[derive(Debug, Deserialize)]
struct RawMtp {
    num_hidden_layers: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum WeightFormat {
    Bf16,
    /// 4-bit weights, bf16 activations, one bf16 scale per `group` weights.
    W4A16 { group: u32 },
    /// NVFP4: e2m1 weights + e4m3 scale per 16 elements (+ fp32 tensor scale).
    Nvfp4,
    /// NVFP4 storage used weight-only (ModelOpt `W4A16_NVFP4`): same bytes as NVFP4, but the
    /// activations stay BF16 in every phase (no static activation scale to quantize them with).
    Nvfp4WeightOnly,
    /// MXFP8: e4m3 weights + a power-of-two (UE8M0) scale per 32 elements.
    Mxfp8,
}

impl WeightFormat {
    /// Average bytes per weight element including scales.
    pub fn bytes_per_elem(&self) -> f64 {
        match self {
            WeightFormat::Bf16 => 2.0,
            WeightFormat::W4A16 { group } => 0.5 + 2.0 / *group as f64,
            WeightFormat::Nvfp4 | WeightFormat::Nvfp4WeightOnly => 0.5 + 1.0 / 16.0,
            WeightFormat::Mxfp8 => 1.0 + 1.0 / 32.0,
        }
    }
}

/// Per-module weight formats from a ModelOpt `MIXED_PRECISION` checkpoint. Modules absent from
/// the map are BF16 (or F32 for norms and state parameters).
#[derive(Debug, Clone, Default, Serialize)]
#[serde(transparent)]
pub struct FormatMap {
    pub modules: BTreeMap<String, WeightFormat>,
}

/// Prefixes under which a checkpoint may place its language model's modules.
pub const TEXT_PREFIXES: [&str; 5] = ["", "model.", "model.language_model.", "language_model.", "language_model.model."];

/// A module path with any language-model prefix removed (`model.language_model.layers.0.x` ->
/// `layers.0.x`). Paths of other towers keep their tower name (`model.visual.x` -> `visual.x`).
pub fn strip_text_prefix(path: &str) -> &str {
    for p in ["model.language_model.", "language_model.model.", "language_model.", "model."] {
        if let Some(r) = path.strip_prefix(p) {
            return r;
        }
    }
    path
}

impl FormatMap {
    pub fn is_empty(&self) -> bool { self.modules.is_empty() }
    /// Format of the module at exactly `path`, if the map lists it.
    pub fn get(&self, path: &str) -> Option<WeightFormat> { self.modules.get(path).copied() }
}

/// RoPE frequency scaling. Only Llama 3's scheme is implemented; others are rejected at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum RopeScaling {
    Llama3 { factor: f64, low_freq_factor: f64, high_freq_factor: f64, original_max_pos: u32 },
}

#[derive(Debug, Clone, Serialize)]
pub struct Moe {
    pub num_experts: u32,
    pub experts_per_tok: u32,
    pub expert_intermediate: u32,
    pub shared_experts: u32,
    /// Total width of the shared expert(s); 0 without shared experts.
    #[serde(skip_serializing_if = "is_zero")]
    pub shared_intermediate: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LayerKind {
    /// Gated DeltaNet: recurrent linear attention with a fixed-size state.
    LinearAttention,
    /// Sparse full attention (QSA): each query reads indexer-selected positions.
    FullAttention,
}

#[derive(Debug, Clone, Serialize)]
pub struct Gdn {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
}

impl Gdn {
    /// `in_proj_qkv` output: q and k (key heads) plus v (value heads).
    pub fn qkv_dim(&self) -> u32 { 2 * self.key_heads * self.key_head_dim + self.value_heads * self.value_head_dim }
    pub fn value_dim(&self) -> u32 { self.value_heads * self.value_head_dim }
    /// FP32 recurrent state: one key_dim x value_dim matrix per value head.
    pub fn state_bytes(&self) -> u64 { self.value_heads as u64 * self.key_head_dim as u64 * self.value_head_dim as u64 * 4 }
    /// BF16 conv1d state: the previous `conv_kernel - 1` inputs of every `in_proj_qkv` channel.
    pub fn conv_state_bytes(&self) -> u64 { self.qkv_dim() as u64 * (self.conv_kernel as u64 - 1) * 2 }
}

#[derive(Debug, Clone, Serialize)]
pub struct Qsa {
    /// `q_proj` also emits a per-element output gate (sigmoid), doubling its rows.
    pub output_gate: bool,
    pub indexer_heads: u32,
    pub indexer_head_dim: u32,
    pub indexer_kv_heads: u32,
    pub compress_ratio: u32,
    pub budget: u32,
}

impl Qsa {
    /// Positions one query attends to: `budget` selected positions plus the trailing tokens of a
    /// compression group that is not complete yet.
    pub fn selected_positions(&self) -> u32 { self.budget + self.compress_ratio - 1 }
    /// `index_qk_proj` output: indexer queries plus indexer keys.
    pub fn index_qk_dim(&self) -> u32 { (self.indexer_heads + self.indexer_kv_heads) * self.indexer_head_dim }
}

#[derive(Debug, Clone, Serialize)]
pub struct HyperConnection {
    /// Residual streams; the residual state is `streams * hidden` wide.
    pub streams: u32,
    /// Width of each mixer's low-rank projection.
    pub lowrank: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Ple {
    pub layer_ids: Vec<u32>,
    pub embed_dim: u32,
    pub conv_kernel: u32,
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub vocab_base: u64,
}

impl Ple {
    /// One head set per n-gram order 2..=ngram_size.
    pub fn heads(&self) -> u32 { (self.ngram_size - 1) * self.heads_per_ngram }
    pub fn head_dim(&self) -> u32 { self.embed_dim / self.heads() }
    /// Table rows, `heads * vocab_base`. The checkpoint's per-head tables are prime-sized near
    /// `vocab_base`, so the true count differs by a few parts per million.
    pub fn table_rows(&self) -> u64 { self.heads() as u64 * self.vocab_base }
}

#[derive(Debug, Clone, Serialize)]
pub struct Mrope {
    pub partial_rotary_factor: f64,
    pub sections: Vec<u32>,
    pub interleaved: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hybrid {
    pub schedule: Vec<LayerKind>,
    pub gdn: Gdn,
    pub qsa: Qsa,
    pub hc: HyperConnection,
    pub ple: Ple,
    pub mrope: Mrope,
    pub mtp_layers: u32,
}

impl Hybrid {
    pub fn count(&self, k: LayerKind) -> u32 { self.schedule.iter().filter(|&&x| x == k).count() as u32 }
    /// Mixers per token: one before attention and one before the MLP in every layer, plus a final
    /// mixer (without injection) before `lm_head`.
    pub fn hc_mixers(&self) -> u32 { 2 * self.schedule.len() as u32 + 1 }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSpec {
    pub arch: String,
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub max_pos: u32,
    pub tie_embeddings: bool,
    pub moe: Option<Moe>,
    /// Format of every weight not listed in `formats` (all weights when `formats` is empty).
    pub weights: WeightFormat,
    /// Per-module formats of a mixed-precision checkpoint.
    #[serde(skip_serializing_if = "FormatMap::is_empty")]
    pub formats: FormatMap,
    /// Per-head RMSNorm on q and k before RoPE (Qwen3). Not generated yet.
    pub qk_norm: bool,
    /// Bias on the q/k/v projections (Qwen2, some Llama variants). Not generated yet.
    pub qkv_bias: bool,
    /// The checkpoint declares ModelOpt MIXED_PRECISION, whatever format the planner was told to
    /// use. Loading refuses such checkpoints even when `--format` overrode the planning format.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub declared_mixed: bool,
    /// Hybrid linear/sparse-attention layout (Qwen3.8 Flash Next). Planned, not generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid: Option<Hybrid>,
    /// `formats` resolved onto a checkpoint's modules (`resolve_modules`). With it, the planner
    /// prices each module by the entries that cover it in that checkpoint; without it (planning
    /// from a config alone), by path matching.
    #[serde(skip)]
    pub resolved: Option<crate::modules::ResolvedFormats>,
}

fn parse_rope_scaling(v: &Option<serde_json::Value>) -> Result<Option<RopeScaling>> {
    let Some(v) = v.as_ref().filter(|v| !v.is_null()) else { return Ok(None) };
    let kind = v.get("rope_type").or_else(|| v.get("type")).and_then(|t| t.as_str()).unwrap_or("default");
    let num = |k: &str| v.get(k).and_then(|x| x.as_f64()).with_context(|| format!("rope_scaling.{k} missing"));
    match kind {
        "default" => Ok(None),
        "llama3" => Ok(Some(RopeScaling::Llama3 {
            factor: num("factor")?,
            low_freq_factor: num("low_freq_factor")?,
            high_freq_factor: num("high_freq_factor")?,
            original_max_pos: num("original_max_position_embeddings")? as u32,
        })),
        other => bail!("rope_scaling type {other:?} is not supported yet (only llama3)"),
    }
}

/// One `MIXED_PRECISION` entry. Group sizes are fixed by each scheme, so any other size is an error.
fn layer_format(module: &str, q: &RawLayerQuant) -> Result<WeightFormat> {
    let (fmt, group) = match q.quant_algo.as_str() {
        "NVFP4" => (WeightFormat::Nvfp4, 16),
        "W4A16_NVFP4" => (WeightFormat::Nvfp4WeightOnly, 16),
        "MXFP8" => (WeightFormat::Mxfp8, 32),
        other => bail!("module {module}: quant_algo {other:?} is not supported in MIXED_PRECISION (NVFP4, W4A16_NVFP4, MXFP8)"),
    };
    if let Some(g) = q.group_size {
        ensure!(g == group, "module {module}: {} with group_size {g} is not supported (expected {group})", q.quant_algo);
    }
    Ok(fmt)
}

fn detect_format(q: &Option<serde_json::Value>) -> Result<(WeightFormat, FormatMap)> {
    let Some(q) = q.as_ref().filter(|v| !v.is_null()) else { return Ok((WeightFormat::Bf16, FormatMap::default())) };
    let q: RawQuant = serde_json::from_value(q.clone())
        .context("quantization_config is malformed; pass --format to override")?;
    let method = q.quant_method.as_deref().unwrap_or("");
    let uniform = |f| Ok((f, FormatMap::default()));
    match (method, q.quant_algo.as_deref(), q.bits) {
        ("nvfp4", _, _) | ("modelopt", Some("NVFP4"), _) => uniform(WeightFormat::Nvfp4),
        ("modelopt", Some("MIXED_PRECISION"), _) => {
            let raw = q.quantized_layers.as_ref().and_then(|m| m.as_object()).filter(|m| !m.is_empty())
                .context("modelopt MIXED_PRECISION without a quantized_layers map; pass --format to override")?;
            let mut layers: BTreeMap<String, RawLayerQuant> = BTreeMap::new();
            for (module, v) in raw {
                let lq = serde_json::from_value(v.clone())
                    .with_context(|| format!("quantized_layers entry {module}: expected {{\"quant_algo\": ..., \"group_size\": ...}}; pass --format to override"))?;
                layers.insert(module.clone(), lq);
            }
            let layers = &layers;
            let mut map = FormatMap::default();
            // Two paths for one module (with and without a language-model prefix) must agree.
            let mut by_module: BTreeMap<&str, (&str, WeightFormat)> = BTreeMap::new();
            for (module, lq) in layers {
                let f = layer_format(module, lq)?;
                if let Some((other, g)) = by_module.insert(strip_text_prefix(module), (module, f)) {
                    ensure!(g == f, "quantized_layers lists {other} as {g:?} and {module} as {f:?}: conflicting formats for one module");
                }
                map.modules.insert(module.clone(), f);
            }
            Ok((WeightFormat::Bf16, map))
        }
        ("modelopt", algo, _) => bail!("modelopt quant_algo {algo:?} is not supported (NVFP4 or MIXED_PRECISION); pass --format to override"),
        ("gptq" | "awq", _, Some(4)) => match q.group_size.unwrap_or(128) {
            g if g > 0 => uniform(WeightFormat::W4A16 { group: g as u32 }),
            g => bail!("{method} group_size {g} (per-channel scales) is not supported; pass --format to override"),
        },
        _ => bail!("quantization_config {method:?} (bits {:?}) is not supported; pass --format to override", q.bits),
    }
}

/// Multimodal configs nest the language model under `text_config`. Its fields win; the top level
/// supplies what only it carries (the checkpoint's quantization and architecture names).
/// Returns the flattened body, plus the top-level `quantization_config` when it differs from a
/// non-null nested one. That conflict is an error only for format detection, so a format override
/// can still parse the config; the caller decides.
fn text_config(v: serde_json::Value) -> Result<(serde_json::Value, Option<serde_json::Value>)> {
    let serde_json::Value::Object(mut top) = v else { bail!("config.json is not a JSON object") };
    let Some(tc) = top.remove("text_config") else { return Ok((serde_json::Value::Object(top), None)) };
    let serde_json::Value::Object(mut text) = tc else { bail!("text_config is not a JSON object") };
    let mut quant_conflict = None;
    for k in ["quantization_config", "architectures"] {
        let Some(x) = top.remove(k).filter(|x| !x.is_null()) else { continue };
        match text.get(k) {
            None | Some(serde_json::Value::Null) => { text.insert(k.into(), x); }
            Some(nested) if *nested == x => {}
            Some(_) if k == "quantization_config" => quant_conflict = Some(x),
            Some(_) => bail!("config.json declares different {k} values at the top level and in text_config"),
        }
    }
    // The family discriminator: the nested one is more specific (`qwen4_exp_text` under
    // `qwen4_exp`), so the top level only fills a missing or null nested value.
    if let Some(mt) = top.remove("model_type").filter(|x| !x.is_null()) {
        if text.get("model_type").is_none_or(|x| x.is_null()) {
            text.insert("model_type".into(), mt);
        }
    }
    Ok((serde_json::Value::Object(text), quant_conflict))
}

fn need<T>(v: Option<T>, field: &str) -> Result<T> {
    v.with_context(|| format!("qwen4_exp config is missing {field}"))
}

/// A required dimension: present and positive (zero would divide by zero or underflow later).
fn dim(v: Option<u32>, field: &str) -> Result<u32> {
    let x = need(v, field)?;
    ensure!(x > 0, "qwen4_exp config field {field} must be positive, got 0");
    Ok(x)
}

fn parse_hybrid(r: &RawConfig, body: &serde_json::Value, head_dim: u32, partial_rotary: Option<f64>) -> Result<Option<Hybrid>> {
    let model_type = r.model_type.as_deref().unwrap_or("unknown");
    let qwen4 = model_type == "qwen4_exp_text" || model_type == "qwen4_exp";
    // A qwen4_exp model always takes the hybrid path, even with no linear-attention layers: its
    // HyperConnection, PLE and gated sparse attention exist in every variant, and the dense
    // kernels implement none of them.
    let types = match (&r.layer_types, qwen4) {
        (Some(t), _) => t,
        (None, true) => bail!("qwen4_exp config is missing layer_types"),
        (None, false) => return Ok(None),
    };
    let schedule = types.iter().map(|t| match t.as_str() {
        "linear_attention" => Ok(LayerKind::LinearAttention),
        "full_attention" => Ok(LayerKind::FullAttention),
        other => bail!("layer type {other:?} is not supported (linear_attention, full_attention)"),
    }).collect::<Result<Vec<_>>>()?;
    if !qwen4 {
        ensure!(schedule.iter().all(|&k| k == LayerKind::FullAttention),
            "hybrid linear-attention model {model_type:?} is not supported (only qwen4_exp, Qwen3.8 Flash Next)");
        return Ok(None);
    }
    ensure!(schedule.len() as u32 == r.num_hidden_layers,
        "layer_types has {} entries but num_hidden_layers is {}", schedule.len(), r.num_hidden_layers);
    let h: RawHybrid = serde_json::from_value(body.clone()).context("parsing qwen4_exp fields")?;
    if let Some(n) = h.full_attention_interval {
        ensure!(n > 0, "full_attention_interval must be positive, got 0");
        for (i, k) in schedule.iter().enumerate() {
            let want = if (i as u32 + 1) % n == 0 { LayerKind::FullAttention } else { LayerKind::LinearAttention };
            ensure!(*k == want, "layer {i} is {k:?}, but full_attention_interval {n} puts {want:?} there");
        }
    }
    let rp = r.rope_parameters.as_ref();
    let factor = need(partial_rotary, "partial_rotary_factor")?;
    let sections: Vec<u32> = rp.and_then(|p| p.get("mrope_section")).and_then(|s| serde_json::from_value(s.clone()).ok())
        .context("qwen4_exp config is missing rope_parameters.mrope_section")?;
    let rotary = (head_dim as f64 * factor).round() as u32;
    ensure!(2 * sections.iter().sum::<u32>() == rotary,
        "mrope_section {sections:?} covers {} dims, but partial_rotary_factor {factor} rotates {rotary} of {head_dim}", 2 * sections.iter().sum::<u32>());
    let ple = Ple {
        layer_ids: need(h.ple_layer_ids, "ple_layer_ids")?,
        embed_dim: dim(h.ple_embed_dim, "ple_embed_dim")?,
        conv_kernel: dim(h.ple_conv_kernel_size, "ple_conv_kernel_size")?,
        ngram_size: dim(h.ngram_size, "ngram_size")?,
        heads_per_ngram: dim(h.heads_per_ngram, "heads_per_ngram")?,
        vocab_base: need(h.ngram_vocab_size_base, "ngram_vocab_size_base")?,
    };
    ensure!(ple.ngram_size >= 2 && ple.embed_dim % ple.heads() == 0,
        "PLE embed_dim {} is not divisible into {} heads (orders 2..={})", ple.embed_dim, ple.heads(), ple.ngram_size);
    // The planner charges one PLE application and one table per token; other placements are not
    // modelled, so they are rejected rather than mis-budgeted.
    ensure!(ple.layer_ids.len() == 1, "ple_layer_ids {:?}: exactly one PLE layer is supported", ple.layer_ids);
    for &l in &ple.layer_ids {
        ensure!(l < r.num_hidden_layers, "ple_layer_ids names layer {l}, but there are {}", r.num_hidden_layers);
    }
    let nested_mtp = h.mtp.as_ref().and_then(|m| m.num_hidden_layers);
    if let (Some(a), Some(b)) = (h.mtp_num_hidden_layers, nested_mtp) {
        ensure!(a == b, "mtp_num_hidden_layers {a} disagrees with mtp.num_hidden_layers {b}");
    }
    let gate = need(h.output_gate_type, "output_gate_type")?;
    ensure!(gate == "sigmoid", "output_gate_type {gate:?} is not supported (sigmoid)");
    let mtp_layers = h.mtp_num_hidden_layers.or(nested_mtp).unwrap_or(0);
    // The MTP budget resolves one set of formats for the draft layer; more layers could each carry
    // their own, so they are rejected until they are budgeted per layer.
    ensure!(mtp_layers <= 1, "{mtp_layers} MTP layers: at most one MTP layer is planned so far");
    Ok(Some(Hybrid {
        schedule,
        gdn: Gdn {
            key_heads: dim(h.linear_num_key_heads, "linear_num_key_heads")?,
            value_heads: dim(h.linear_num_value_heads, "linear_num_value_heads")?,
            key_head_dim: dim(h.linear_key_head_dim, "linear_key_head_dim")?,
            value_head_dim: dim(h.linear_value_head_dim, "linear_value_head_dim")?,
            conv_kernel: dim(h.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
        },
        qsa: Qsa {
            output_gate: true,
            indexer_heads: dim(h.indexer_n_heads, "indexer_n_heads")?,
            indexer_head_dim: dim(h.indexer_head_dim, "indexer_head_dim")?,
            indexer_kv_heads: dim(h.indexer_kv_heads, "indexer_kv_heads")?,
            compress_ratio: dim(h.indexer_compress_ratio, "indexer_compress_ratio")?,
            budget: dim(h.indexer_budget, "indexer_budget")?,
        },
        hc: HyperConnection { streams: dim(h.hc_count, "hc_count")?, lowrank: dim(h.hc_lowrank, "hc_lowrank")? },
        ple,
        mrope: Mrope {
            partial_rotary_factor: factor,
            sections,
            interleaved: rp.and_then(|p| p.get("mrope_interleaved")).and_then(|b| b.as_bool()).unwrap_or(false),
        },
        mtp_layers,
    }))
}

/// Format-map rules the hybrid planner relies on.
fn check_hybrid_formats(formats: &FormatMap, h: &Hybrid) -> Result<()> {
    for path in formats.modules.keys() {
        // HyperConnection mixers are planned as BF16 GEMVs (down merged with the injection rows).
        ensure!(!path.contains("hyper_connection"),
            "quantized_layers lists {path}: quantized HyperConnection weights are not planned yet (they are BF16 in the planner)");
    }
    // With one MTP layer, every `mtp.layers.<n>.<module>` entry names that layer's module, so
    // duplicate entries (the checkpoint lists the experts under two indices) must agree.
    if h.mtp_layers == 1 {
        let mut seen: BTreeMap<&str, (&str, WeightFormat)> = BTreeMap::new();
        for (path, f) in &formats.modules {
            let Some(rest) = path.strip_prefix("mtp.layers.") else { continue };
            let Some((_, module)) = rest.split_once('.') else { continue };
            if let Some((other, g)) = seen.insert(module, (path, *f)) {
                ensure!(g == *f, "quantized_layers lists {other} as {g:?} and {path} as {f:?}: conflicting formats for the one MTP layer");
            }
        }
    }
    Ok(())
}

impl ModelSpec {
    pub fn from_json(text: &str, override_fmt: Option<WeightFormat>) -> Result<Self> {
        let (body, quant_conflict) = text_config(serde_json::from_str(text).context("parsing config.json")?)?;
        if override_fmt.is_none() {
            ensure!(quant_conflict.is_none(),
                "config.json declares different quantization_config values at the top level and in text_config; pass --format to override");
        }
        let r: RawConfig = serde_json::from_value(body.clone()).context("parsing config.json")?;
        let kv_heads = r.num_key_value_heads.unwrap_or(r.num_attention_heads);
        let head_dim = r.head_dim.unwrap_or(r.hidden_size / r.num_attention_heads);
        if r.kv_lora_rank.is_some() {
            bail!("MLA attention (kv_lora_rank, DeepSeek V2/V3 style) is not supported");
        }
        if r.num_attention_heads % kv_heads != 0 {
            bail!("heads ({}) not divisible by kv_heads ({})", r.num_attention_heads, kv_heads);
        }
        let rp = r.rope_parameters.as_ref();
        let partial_rotary = r.partial_rotary_factor
            .or_else(|| rp.and_then(|p| p.get("partial_rotary_factor")).and_then(|x| x.as_f64()));
        let hybrid = parse_hybrid(&r, &body, head_dim, partial_rotary)?;
        if hybrid.is_none() {
            if let Some(f) = partial_rotary.filter(|&f| f != 1.0) {
                bail!("partial_rotary_factor {f} is not supported: the generated RoPE rotates the whole head");
            }
        }
        let moe = match (r.num_experts, r.num_experts_per_tok) {
            (Some(e), Some(k)) if e > 1 => {
                ensure!((1..=e).contains(&k), "num_experts_per_tok {k} must be between 1 and num_experts ({e})");
                let expert_intermediate = r.moe_intermediate_size.or(r.intermediate_size)
                    .context("MoE model without moe_intermediate_size")?;
                let shared_intermediate = r.shared_expert_intermediate_size
                    .unwrap_or(r.n_shared_experts.unwrap_or(0) * expert_intermediate);
                Some(Moe {
                    num_experts: e,
                    experts_per_tok: k,
                    expert_intermediate,
                    shared_experts: r.n_shared_experts.unwrap_or(u32::from(shared_intermediate > 0)),
                    shared_intermediate,
                })
            }
            _ => None,
        };
        // An explicit --format wins, so an unrecognized quantization_config is not fatal then. It
        // replaces detection entirely, including a mixed-precision map.
        let (weights, formats) = match override_fmt {
            Some(f) => (f, FormatMap::default()),
            None => detect_format(&r.quantization_config)?,
        };
        if let Some(h) = &hybrid {
            check_hybrid_formats(&formats, h)?;
        }
        // Only the hybrid planner reads per-module formats; a dense plan would treat every listed
        // module as the default format, so a dense mixed-precision checkpoint is rejected.
        ensure!(formats.is_empty() || hybrid.is_some(),
            "modelopt MIXED_PRECISION is supported only for qwen4_exp (Qwen3.8 Flash Next) so far; dense planning does not read per-module formats yet. Pass --format to override");
        let arch = r.architectures.first().cloned().unwrap_or_else(|| "unknown".into());
        let rope_theta = r.rope_theta
            .or_else(|| rp.and_then(|p| p.get("rope_theta")).and_then(|x| x.as_f64()))
            .unwrap_or(10_000.0);
        // Read from the raw strings, independently of format detection, so an override never
        // hides that the checkpoint itself is mixed precision.
        // Either level declaring mixed precision is enough: an override never hides it.
        let is_mixed = |q: &serde_json::Value| {
            q.get("quant_method").and_then(|v| v.as_str()) == Some("modelopt")
                && q.get("quant_algo").and_then(|v| v.as_str()) == Some("MIXED_PRECISION")
        };
        let declared_mixed = r.quantization_config.as_ref().is_some_and(is_mixed) || quant_conflict.as_ref().is_some_and(is_mixed);
        Ok(ModelSpec {
            declared_mixed,
            qk_norm: arch.starts_with("Qwen3"),
            qkv_bias: r.attention_bias.unwrap_or(arch.starts_with("Qwen2")),
            rope_scaling: parse_rope_scaling(if r.rope_scaling.is_some() { &r.rope_scaling } else { &r.rope_parameters })?,
            arch,
            hidden: r.hidden_size,
            intermediate: r.intermediate_size.unwrap_or(4 * r.hidden_size),
            layers: r.num_hidden_layers,
            heads: r.num_attention_heads,
            kv_heads,
            head_dim,
            vocab: r.vocab_size,
            rms_eps: r.rms_norm_eps,
            rope_theta,
            max_pos: r.max_position_embeddings.unwrap_or(8192),
            tie_embeddings: r.tie_word_embeddings,
            moe,
            weights,
            formats,
            hybrid,
            resolved: None,
        })
    }

    /// Resolves `formats` onto the modules of a checkpoint, given its tensor names: each entry
    /// applies to the modules at or under its path, entries that cover one module must agree, and
    /// entries that cover none are reported as unused instead of matching by pattern.
    pub fn resolve_modules<'a>(&mut self, names: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let table = crate::modules::ModuleTable::from_tensor_names(names);
        self.resolved = Some(crate::modules::resolve(&self.formats, table)?);
        Ok(())
    }

    pub fn gqa_group(&self) -> u32 { self.heads / self.kv_heads }
    pub fn q_dim(&self) -> u32 { self.heads * self.head_dim }
    pub fn kv_dim(&self) -> u32 { self.kv_heads * self.head_dim }
    /// Fused QKV projection output width.
    pub fn qkv_dim(&self) -> u32 { self.q_dim() + 2 * self.kv_dim() }
    /// KV-cache bytes per token across all layers (bf16 cache).
    pub fn kv_bytes_per_token(&self) -> u64 { 2 * self.layers as u64 * self.kv_dim() as u64 * 2 }
}

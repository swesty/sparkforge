//! Hugging Face `config.json` -> normalized ModelSpec.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct RawQuant {
    quant_method: Option<String>,
    quant_algo: Option<String>,
    bits: Option<u32>,
    /// GPTQ uses -1 for per-channel scales, so this must be signed.
    group_size: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: u32,
    intermediate_size: Option<u32>,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: Option<u32>,
    head_dim: Option<u32>,
    vocab_size: u32,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    rope_theta: f64,
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    tie_word_embeddings: bool,
    // MoE (names differ across families)
    #[serde(alias = "num_local_experts", alias = "n_routed_experts")]
    num_experts: Option<u32>,
    num_experts_per_tok: Option<u32>,
    moe_intermediate_size: Option<u32>,
    n_shared_experts: Option<u32>,
    quantization_config: Option<RawQuant>,
    rope_scaling: Option<serde_json::Value>,
    attention_bias: Option<bool>,
    /// Present only on MLA models (DeepSeek V2/V3), which this engine does not plan.
    kv_lora_rank: Option<u32>,
}
fn default_eps() -> f64 { 1e-6 }
fn default_theta() -> f64 { 10_000.0 }

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum WeightFormat {
    Bf16,
    /// 4-bit weights, bf16 activations, one bf16 scale per `group` weights.
    W4A16 { group: u32 },
    /// NVFP4: e2m1 weights + e4m3 scale per 16 elements (+ fp32 tensor scale).
    Nvfp4,
}

impl WeightFormat {
    /// Average bytes per weight element including scales.
    pub fn bytes_per_elem(&self) -> f64 {
        match self {
            WeightFormat::Bf16 => 2.0,
            WeightFormat::W4A16 { group } => 0.5 + 2.0 / *group as f64,
            WeightFormat::Nvfp4 => 0.5 + 1.0 / 16.0,
        }
    }
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
    pub weights: WeightFormat,
    /// Per-head RMSNorm on q and k before RoPE (Qwen3). Not generated yet.
    pub qk_norm: bool,
    /// Bias on the q/k/v projections (Qwen2, some Llama variants). Not generated yet.
    pub qkv_bias: bool,
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

fn detect_format(q: &Option<RawQuant>) -> Result<WeightFormat> {
    let Some(q) = q else { return Ok(WeightFormat::Bf16) };
    let method = q.quant_method.as_deref().unwrap_or("");
    match (method, q.quant_algo.as_deref(), q.bits) {
        ("nvfp4", _, _) | ("modelopt", Some("NVFP4"), _) => Ok(WeightFormat::Nvfp4),
        ("modelopt", algo, _) => bail!("modelopt quant_algo {algo:?} is not supported (only NVFP4); pass --format to override"),
        ("gptq" | "awq", _, Some(4)) => match q.group_size.unwrap_or(128) {
            g if g > 0 => Ok(WeightFormat::W4A16 { group: g as u32 }),
            g => bail!("{method} group_size {g} (per-channel scales) is not supported; pass --format to override"),
        },
        _ => bail!("quantization_config {method:?} (bits {:?}) is not supported; pass --format to override", q.bits),
    }
}

impl ModelSpec {
    pub fn from_json(text: &str, override_fmt: Option<WeightFormat>) -> Result<Self> {
        let r: RawConfig = serde_json::from_str(text).context("parsing config.json")?;
        let kv_heads = r.num_key_value_heads.unwrap_or(r.num_attention_heads);
        let head_dim = r.head_dim.unwrap_or(r.hidden_size / r.num_attention_heads);
        if r.kv_lora_rank.is_some() {
            bail!("MLA attention (kv_lora_rank, DeepSeek V2/V3 style) is not supported");
        }
        if r.num_attention_heads % kv_heads != 0 {
            bail!("heads ({}) not divisible by kv_heads ({})", r.num_attention_heads, kv_heads);
        }
        let moe = match (r.num_experts, r.num_experts_per_tok) {
            (Some(e), Some(k)) if e > 1 => Some(Moe {
                num_experts: e,
                experts_per_tok: k,
                expert_intermediate: r.moe_intermediate_size.or(r.intermediate_size)
                    .context("MoE model without moe_intermediate_size")?,
                shared_experts: r.n_shared_experts.unwrap_or(0),
            }),
            _ => None,
        };
        // An explicit --format wins, so an unrecognized quantization_config is not fatal then.
        let weights = match override_fmt {
            Some(f) => f,
            None => detect_format(&r.quantization_config)?,
        };
        let arch = r.architectures.first().cloned().unwrap_or_else(|| "unknown".into());
        Ok(ModelSpec {
            qk_norm: arch.starts_with("Qwen3"),
            qkv_bias: r.attention_bias.unwrap_or(arch.starts_with("Qwen2")),
            rope_scaling: parse_rope_scaling(&r.rope_scaling)?,
            arch,
            hidden: r.hidden_size,
            intermediate: r.intermediate_size.unwrap_or(4 * r.hidden_size),
            layers: r.num_hidden_layers,
            heads: r.num_attention_heads,
            kv_heads,
            head_dim,
            vocab: r.vocab_size,
            rms_eps: r.rms_norm_eps,
            rope_theta: r.rope_theta,
            max_pos: r.max_position_embeddings.unwrap_or(8192),
            tie_embeddings: r.tie_word_embeddings,
            moe,
            weights,
        })
    }

    pub fn gqa_group(&self) -> u32 { self.heads / self.kv_heads }
    pub fn q_dim(&self) -> u32 { self.heads * self.head_dim }
    pub fn kv_dim(&self) -> u32 { self.kv_heads * self.head_dim }
    /// Fused QKV projection output width.
    pub fn qkv_dim(&self) -> u32 { self.q_dim() + 2 * self.kv_dim() }
    /// KV-cache bytes per token across all layers (bf16 cache).
    pub fn kv_bytes_per_token(&self) -> u64 { 2 * self.layers as u64 * self.kv_dim() as u64 * 2 }
}

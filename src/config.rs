//! Hugging Face `config.json` -> normalized ModelSpec.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct RawQuant {
    quant_method: Option<String>,
    bits: Option<u32>,
    group_size: Option<u32>,
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
    pub max_pos: u32,
    pub tie_embeddings: bool,
    pub moe: Option<Moe>,
    pub weights: WeightFormat,
}

impl ModelSpec {
    pub fn from_json(text: &str, override_fmt: Option<WeightFormat>) -> Result<Self> {
        let r: RawConfig = serde_json::from_str(text).context("parsing config.json")?;
        let kv_heads = r.num_key_value_heads.unwrap_or(r.num_attention_heads);
        let head_dim = r.head_dim.unwrap_or(r.hidden_size / r.num_attention_heads);
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
        let detected = match &r.quantization_config {
            Some(q) => match (q.quant_method.as_deref(), q.bits) {
                (Some("modelopt"), _) | (Some("nvfp4"), _) => WeightFormat::Nvfp4,
                (_, Some(4)) => WeightFormat::W4A16 { group: q.group_size.unwrap_or(128) },
                _ => WeightFormat::Bf16,
            },
            None => WeightFormat::Bf16,
        };
        Ok(ModelSpec {
            arch: r.architectures.first().cloned().unwrap_or_else(|| "unknown".into()),
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
            weights: override_fmt.unwrap_or(detected),
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

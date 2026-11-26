//! The per-token decode estimate `verify --bench` prints, from isolated kernel timings (host only,
//! so `cargo test` covers the arithmetic without a GPU).
use crate::plan::Plan;
use std::collections::HashMap;

/// One `verify` result row.
pub struct Row {
    pub kernel: String,
    pub case: String,
    pub err: f64,
    pub us: Option<f64>,
    pub gbs: Option<f64>,
}

/// Sum of isolated kernel times for one decode token (no launch gaps, no graph). None if a kernel
/// the token needs was not generated or not timed. An expert GEMV counts once per active expert,
/// except a grouped one (NVFP4, one launch for every active expert), which counts once per layer. A
/// shared expert adds its two GEMVs and one SiLU-and-multiply, and has no estimate when its width
/// differs from the routed one (the only width `silu_mul` is generated at).
pub fn token_estimate_us(p: &Plan, rows: &[Row]) -> Option<f64> {
    let t: HashMap<&str, f64> = rows.iter().filter_map(|r| Some((r.kernel.as_str(), r.us?))).collect();
    let g = |k: &str| t.get(k).copied();
    // Launches per layer of a routed-expert GEMV: one when grouped, else one per active expert.
    let launches = |linear: &str, active: u32| {
        let grouped = p.gemv.iter().any(|g| g.linear == linear && g.fp4.as_ref().is_some_and(|f| f.experts > 0));
        if grouped { 1.0 } else { active as f64 }
    };
    let mut layer = 2.0 * g("rmsnorm")? + g("gemv_qkv_proj")? + g("rope_kv_append")? + g("attn_decode (split+combine)")? + g("gemv_o_proj")?;
    match &p.model.moe {
        None => layer += g("gemv_gate_up_proj")? + g("silu_mul")? + g("gemv_down_proj")?,
        Some(e) => {
            let k = e.experts_per_tok;
            layer += g("gemv_router")? + launches("expert_gate_up", k) * g("gemv_expert_gate_up")? + k as f64 * g("silu_mul")?
                + launches("expert_down", k) * g("gemv_expert_down")?;
            if e.shared_experts > 0 {
                // The shared expert has its own SiLU-and-multiply between its GEMVs. `silu_mul` is
                // generated at the routed width only, so a shared expert of another width has no
                // activation kernel yet, and the token cannot be estimated.
                if e.shared_intermediate != e.expert_intermediate {
                    return None;
                }
                layer += g("gemv_shared_gate_up")? + g("silu_mul")? + g("gemv_shared_down")?;
            }
        }
    }
    Some(p.model.layers as f64 * layer + g("rmsnorm")? + g("gemv_lm_head")?)
}

/// Prefill time for the planned prompt length from isolated GEMM times: per layer the q/k/v,
/// o_proj and MLP GEMMs (for MoE, the router, one expert GEMM pair per expert at the planned tokens
/// per expert, and the shared expert's pair), then `gemv_lm_head` once for the last position.
/// Attention, norms, RoPE and activations are not included. Returns (microseconds, prompt tokens),
/// or the name of the kernel that has no timing.
pub fn prefill_estimate_us(p: &Plan, rows: &[Row]) -> Result<(f64, u32), String> {
    let t: HashMap<&str, f64> = rows.iter().filter_map(|r| Some((r.kernel.as_str(), r.us?))).collect();
    let g = |k: &str| t.get(k).copied().ok_or_else(|| format!("{k} was not generated or not timed"));
    let tokens = p.gemm.iter().find(|t| t.linear == "qkv_proj").ok_or("no qkv_proj GEMM is planned")?.m;
    let mut layer = g("gemm_qkv_proj")? + g("gemm_o_proj")?;
    match &p.model.moe {
        None => layer += g("gemm_gate_up_proj")? + g("gemm_down_proj")?,
        Some(e) => {
            layer += g("gemm_router")? + e.num_experts as f64 * (g("gemm_expert_gate_up")? + g("gemm_expert_down")?);
            if e.shared_experts > 0 {
                layer += g("gemm_shared_gate_up")? + g("gemm_shared_down")?;
            }
        }
    }
    Ok((p.model.layers as f64 * layer + g("gemv_lm_head")?, tokens))
}

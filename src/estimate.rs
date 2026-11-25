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
/// except a grouped one (NVFP4, one launch for every active expert), which counts once per layer.
pub fn token_estimate_us(p: &Plan, rows: &[Row]) -> Option<f64> {
    let t: HashMap<&str, f64> = rows.iter().filter_map(|r| Some((r.kernel.as_str(), r.us?))).collect();
    let g = |k: &str| t.get(k).copied();
    // Launches per layer of a routed-expert GEMV: one when grouped, else one per active expert.
    let launches = |linear: &str, active: u32| {
        let grouped = p.gemv.iter().any(|g| g.linear == linear && g.fp4.as_ref().is_some_and(|f| f.slots > 1));
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
                layer += g("gemv_shared_gate_up")? + g("gemv_shared_down")?;
            }
        }
    }
    Some(p.model.layers as f64 * layer + g("rmsnorm")? + g("gemv_lm_head")?)
}

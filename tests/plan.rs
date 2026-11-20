use sparkforge::{codegen, config::ModelSpec, plan, target::GB10};

fn load(p: &str) -> plan::Plan {
    let s = ModelSpec::from_json(&std::fs::read_to_string(p).unwrap(), None).unwrap();
    plan::build(&s, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 32768 })
}

#[test]
fn llama_shapes_are_derived_from_config() {
    let p = load("configs/llama-3.1-8b.json");
    let n = |name: &str| p.linears.iter().find(|l| l.name == name).unwrap().n;
    assert_eq!(n("qkv_proj"), 32 * 128 + 2 * 8 * 128);
    assert_eq!(n("gate_up_proj"), 2 * 14336);
    assert_eq!(p.attn.group, 4);
    assert!(p.budget.fits);
}

#[test]
fn every_kernel_respects_gb10_limits() {
    for cfg in ["configs/llama-3.1-8b.json", "configs/qwen3-30b-a3b.json"] {
        let p = load(cfg);
        for t in &p.gemm { assert!(t.smem_bytes <= GB10.smem_per_block_optin, "{t:?}"); }
        assert!(p.attn.smem_bytes <= GB10.smem_per_block_optin);
        assert!(p.attn.grid[0] * p.attn.grid[1] >= GB10.sm_count, "attention must fill all 48 SMs");
        for k in codegen::generate(&p).kernels { assert!(k.block <= 1024 && k.block % 32 == 0, "{k:?}"); }
    }
}

#[test]
fn moe_is_detected_and_planned() {
    let p = load("configs/qwen3-30b-a3b.json");
    let e = p.model.moe.as_ref().unwrap();
    assert_eq!((e.num_experts, e.experts_per_tok, e.expert_intermediate), (128, 8, 768));
    assert!(p.gemm.iter().any(|t| t.linear == "expert_gate_up" && t.m == 128));
}

#[test]
fn different_models_get_different_kernels() {
    let a = codegen::generate(&load("configs/llama-3.1-8b.json"));
    let b = codegen::generate(&load("configs/qwen3-30b-a3b.json"));
    assert_ne!(a.cache_key, b.cache_key);
}

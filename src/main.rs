use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use sparkforge::{codegen, config::{ModelSpec, WeightFormat}, plan, target::GB10};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sparkforge", about = "Model-specialized kernels for DGX Spark (GB10/SM121)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Fmt { Auto, Bf16, W4a16, Nvfp4 }

#[derive(Subcommand)]
enum Cmd {
    /// Print the derived shapes, kernel choices, and memory/throughput budget.
    Plan(Common),
    /// Write specialized CUDA source + manifest.json into --out.
    Emit {
        #[command(flatten)]
        c: Common,
        #[arg(long, default_value = "build")]
        out: PathBuf,
    },
    /// Compile with NVRTC for sm_121a and load every generated kernel (needs --features cuda on a Spark).
    #[cfg(feature = "cuda")]
    Load {
        #[command(flatten)]
        c: Common,
        #[arg(long, default_value = "/usr/local/cuda/include")]
        cuda_include: String,
    },
}

#[derive(clap::Args)]
struct Common {
    /// Path to a Hugging Face config.json
    config: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    format: Fmt,
    /// Prompt length used to size prefill GEMM tiles
    #[arg(long, default_value_t = 2048)]
    prefill_tokens: u32,
    /// Context length used for KV budget and attention split choice
    #[arg(long, default_value_t = 32768)]
    ctx: u32,
    #[arg(long)]
    json: bool,
}

fn build(c: &Common) -> Result<plan::Plan> {
    let fmt = match c.format {
        Fmt::Auto => None,
        Fmt::Bf16 => Some(WeightFormat::Bf16),
        Fmt::W4a16 => Some(WeightFormat::W4A16 { group: 128 }),
        Fmt::Nvfp4 => Some(WeightFormat::Nvfp4),
    };
    let spec = ModelSpec::from_json(&std::fs::read_to_string(&c.config)?, fmt)?;
    Ok(plan::build(&spec, &GB10, &plan::PlanOpts { prefill_tokens: c.prefill_tokens, ctx: c.ctx }))
}

fn print_plan(p: &plan::Plan) {
    let m = &p.model;
    println!("== {} on {} ({}) ==", m.arch, p.target.name, p.target.arch);
    println!("hidden {}  layers {}  heads {}/{} kv (group {})  head_dim {}  vocab {}  weights {:?}",
        m.hidden, m.layers, m.heads, m.kv_heads, m.gqa_group(), m.head_dim, m.vocab, m.weights);
    if let Some(e) = &m.moe { println!("MoE: {} experts, top-{}, expert_inter {}", e.num_experts, e.experts_per_tok, e.expert_intermediate); }
    let b = &p.budget;
    println!("\nMemory: weights {:.1} GiB + KV {:.1} GiB @ {} ctx -> fits: {} (max ctx ~{})", b.weight_gib, b.kv_gib_at_ctx, b.ctx, b.fits, b.max_ctx_that_fits);
    println!("Decode roofline: {:.2} GB/token -> ceiling {:.1} tok/s at {:.0} GB/s effective", b.decode_bytes_per_token_gb, b.decode_tok_s_ceiling, p.target.dram_bw_effective_gbs);
    println!("\nDecode GEMV (memory-bound):");
    for g in &p.gemv { println!("  {:<16} N={:<6} K={:<6} warps/blk {} rows/warp {} grid {:<5} ~{:.1} us", g.linear, g.n, g.k, g.warps_per_block, g.rows_per_warp, g.grid, g.est_us); }
    println!("\nPrefill GEMM tiles (mma.sync, smem <= {} B):", p.target.smem_per_block_optin);
    for t in &p.gemm { println!("  {:<16} M={:<5} N={:<6} K={:<6} tile {}x{}x{} stages {} warps {} smem {:>6} ctas/SM {} wave_eff {:.2}", t.linear, t.m, t.n, t.k, t.bm, t.bn, t.bk, t.stages, t.warps, t.smem_bytes, t.ctas_per_sm, t.wave_efficiency); }
    let a = &p.attn;
    println!("\nDecode attention: grid {}x{} (kv_heads x splits), {} warps, smem {} B, GQA group {}", a.grid[0], a.grid[1], a.warps, a.smem_bytes, a.group);
    println!("RMSNorm: {} threads x {} iters of 8 bf16", p.norm.threads, p.norm.iters);
    for n in &p.notes { println!("note: {n}"); }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Plan(c) => {
            let p = build(&c)?;
            if c.json { println!("{}", serde_json::to_string_pretty(&p)?) } else { print_plan(&p) }
        }
        Cmd::Emit { c, out } => {
            let p = build(&c)?;
            let g = codegen::generate(&p);
            let dir = out.join(&g.cache_key);
            std::fs::create_dir_all(&dir)?;
            for (name, body) in &g.files { std::fs::write(dir.join(name), body)?; }
            std::fs::write(dir.join("plan.json"), serde_json::to_string_pretty(&p)?)?;
            std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&g.kernels)?)?;
            println!("wrote {} kernels to {}", g.kernels.len(), dir.display());
        }
        #[cfg(feature = "cuda")]
        Cmd::Load { c, cuda_include } => {
            let g = codegen::generate(&build(&c)?);
            let e = sparkforge::runtime::Engine::load(&g, &cuda_include)?;
            println!("loaded {} kernels for sm_121a", e.funcs.len());
        }
    }
    Ok(())
}

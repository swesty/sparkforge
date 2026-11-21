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

/// Spec, plan and validated binding for a checkpoint directory (config.json beside the weights).
fn open_checkpoint(dir: &std::path::Path) -> Result<(plan::Plan, sparkforge::weights::Checkpoint, sparkforge::weights::Binding)> {
    // One opened directory for the config, the index and every shard.
    let dir = &sparkforge::weights::resolve_checkpoint_dir(dir)?;
    let spec = ModelSpec::from_json(&sparkforge::weights::read_config(dir)?, None)?;
    let p = plan::build(&spec, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 });
    let ck = sparkforge::weights::Checkpoint::open_dir(dir)?;
    let b = sparkforge::weights::bind(&ck, &spec)?;
    Ok((p, ck, b))
}

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
    /// Open a safetensors checkpoint directory (with its config.json) and validate it against the plan.
    Weights {
        /// Checkpoint directory containing config.json and model.safetensors (or an index + shards)
        dir: PathBuf,
    },
    /// GEMVs on real checkpoint weights against an f64 reference (needs --features cuda on a Spark).
    #[cfg(feature = "cuda")]
    VerifyWeights {
        dir: PathBuf,
        #[arg(long, default_value = "/usr/local/cuda/include")]
        cuda_include: String,
        #[arg(long, default_value_t = 0x5eed)]
        seed: u64,
    },
    /// Run every generated kernel on the GPU against a CPU reference (needs --features cuda on a Spark).
    #[cfg(feature = "cuda")]
    Verify {
        #[command(flatten)]
        c: Common,
        #[arg(long, default_value = "/usr/local/cuda/include")]
        cuda_include: String,
        /// Also time each kernel (weights rotated so they cannot sit in L2).
        #[arg(long)]
        bench: bool,
        #[arg(long, default_value_t = 0x5eed)]
        seed: u64,
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
    if let Some(e) = &m.moe { println!("MoE: {} experts, top-{}, expert_inter {}, shared {}", e.num_experts, e.experts_per_tok, e.expert_intermediate, e.shared_intermediate); }
    if !m.formats.is_empty() { println!("mixed precision: {} quantized modules, everything else {:?}", m.formats.modules.len(), m.weights); }
    let b = &p.budget;
    println!("\nMemory: weights {:.1} GiB + KV/state {:.1} GiB @ {} ctx -> fits: {} (max ctx ~{})", b.weight_gib, b.kv_gib_at_ctx, b.ctx, b.fits, b.max_ctx_that_fits);
    println!("Decode roofline: {:.3} GB/token{} -> ceiling {:.1} tok/s at {:.0} GB/s effective",
        b.decode_bytes_per_token_gb, b.decode_bytes_per_token_ctx_free_gb.map_or(String::new(), |c| format!(" ({c:.3} GB context-free)")),
        b.decode_tok_s_ceiling, p.target.dram_bw_effective_gbs);
    if m.hybrid.is_some() {
        println!("\nDecode ops ({} distinct):", p.ops.len());
        for o in &p.ops {
            println!("  {:<28} {:<11} x{:<3} {:<42} {:>9.1} MB/token{}", o.name, o.kind, o.per_token, o.shape, o.bytes_per_token / 1e6,
                o.format.map_or(String::new(), |f| format!("  {f:?}")));
        }
        println!("\nContext-free bytes per token by group:");
        for (g, bytes) in plan::bytes_by_group(p) { println!("  {:<18} {:>9.1} MB", g, bytes / 1e6); }
        for n in &p.notes { println!("note: {n}"); }
        return;
    }
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
        Cmd::Weights { dir } => {
            let t0 = std::time::Instant::now();
            let (p, ck, b) = open_checkpoint(&dir)?;
            let bytes: u64 = ck.shards.iter().map(|s| s.len).sum();
            println!("{}: {} shard(s), {} tensors, {:.2} GiB, opened and validated in {:.2} s",
                p.model.arch, ck.shards.len(), ck.tensors.len(), bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
            println!("bound {} linears ({} fused) and {} other tensors; {} unexpected tensor(s){}",
                b.linears.len(), b.linears.iter().filter(|l| l.fused()).count(), b.others.len(), b.unexpected.len(),
                if b.unexpected.is_empty() { String::new() } else { format!(": {}", b.unexpected.iter().take(5).cloned().collect::<Vec<_>>().join(", ")) });
        }
        #[cfg(feature = "cuda")]
        Cmd::VerifyWeights { dir, cuda_include, seed } => {
            use sparkforge::verify;
            let (p, ck, b) = open_checkpoint(&dir)?;
            let avail0 = sparkforge::weights::mem_available().unwrap_or(0);
            // Admission (lock + memory guard) before any CUDA call; held until verification has
            // freed the weights and its reference scratch.
            let admission = sparkforge::weights::admit_load(&ck, &b, true)?;
            let g = codegen::generate(&p);
            let e = sparkforge::runtime::Engine::load(&g, &cuda_include)?;
            let rep = verify::verify_weights(&p, &g, &e, &ck, &b, seed, &admission)?;
            drop(admission);
            println!("{} ({} layers), MemAvailable at start {:.1} GiB", p.model.arch, p.model.layers, avail0 as f64 / (1u64 << 30) as f64);
            // Sampled inside the upload while the weights were resident, not here after they were freed.
            println!("loaded {:.2} GiB into GPU memory in {:.2} s; MemAvailable with the weights resident {}", rep.bytes_copied as f64 / (1u64 << 30) as f64,
                rep.load_secs, rep.mem_available_after_load.map_or("unknown".into(), |a| format!("{:.1} GiB", a as f64 / (1u64 << 30) as f64)));
            println!("{:<13} {:<16} {:>10} {:>5} {:>10} {:>8}", "linear", "case", "max err", "ok", "us", "GB/s");
            let mut fails = 0;
            for r in &rep.rows {
                let ok = r.err.is_none_or(|e| e < verify::TOL);
                fails += !ok as usize;
                println!("{:<13} {:<16} {:>10} {:>5} {:>10} {:>8}", r.linear, r.case,
                    r.err.map_or("-".into(), |e| format!("{e:.2e}")), if r.err.is_none() { "" } else if ok { "yes" } else { "NO" },
                    r.us.map_or("-".into(), |t| format!("{t:.1}")), r.gbs.map_or("-".into(), |g| format!("{g:.0}")));
            }
            anyhow::ensure!(fails == 0, "{fails} check(s) above tolerance {:.4}", verify::TOL);
        }
        #[cfg(feature = "cuda")]
        Cmd::Verify { c, cuda_include, bench, seed } => {
            use sparkforge::verify;
            let p = build(&c)?;
            let g = codegen::generate(&p);
            // Before any CUDA call, so a planned-only model says so even without a GPU.
            codegen::require_generated(&g, &p.model.arch)?;
            let e = sparkforge::runtime::Engine::load(&g, &cuda_include)?;
            let rows = verify::run(&p, &g, &e, &verify::Opts { bench, seed })?;
            println!("{:<28} {:<26} {:>10} {:>5} {:>10} {:>8}", "kernel", "case", "max err", "ok", "us", "GB/s");
            let mut fails = 0;
            for r in &rows {
                let ok = r.err < verify::TOL;
                fails += !ok as usize;
                let us = r.us.map_or("-".into(), |t| format!("{t:.1}"));
                let gbs = r.gbs.map_or("-".into(), |b| format!("{b:.0}"));
                println!("{:<28} {:<26} {:>10.2e} {:>5} {:>10} {:>8}", r.kernel, r.case, r.err, if ok { "yes" } else { "NO" }, us, gbs);
            }
            let skipped: Vec<_> = g.kernels.iter().filter(|k| k.status != "generated").map(|k| k.entry.as_str()).collect();
            if !skipped.is_empty() { println!("not generated yet (skipped): {}", skipped.join(", ")); }
            if let Some(t) = verify::token_estimate_us(&p, &rows) {
                println!("decode token, sum of isolated kernel times @ ctx {}: {:.2} ms -> {:.1} tok/s (planner ceiling {:.1})",
                    p.budget.ctx, t / 1e3, 1e6 / t, p.budget.decode_tok_s_ceiling);
            }
            anyhow::ensure!(fails == 0, "{fails} check(s) above tolerance {:.4}", verify::TOL);
        }
    }
    Ok(())
}

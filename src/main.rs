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

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, ValueEnum)]
enum PleArg { Disk, Resident }

/// Spec (resolved onto the checkpoint's modules) and opened checkpoint for a checkpoint directory
/// (config.json beside the weights).
fn open_checkpoint(dir: &std::path::Path) -> Result<(ModelSpec, sparkforge::weights::Checkpoint)> {
    // One opened directory for the config, the index and every shard.
    let dir = &sparkforge::weights::resolve_checkpoint_dir(dir)?;
    let mut spec = ModelSpec::from_json(&sparkforge::weights::read_config(dir)?, None)?;
    let ck = sparkforge::weights::Checkpoint::open_dir(dir)?;
    // Formats are priced by the modules this checkpoint actually holds.
    spec.resolve_modules(ck.tensors.keys().map(String::as_str))?;
    Ok((spec, ck))
}

fn checkpoint_plan(spec: &ModelSpec) -> plan::Plan {
    plan::build(spec, &GB10, &plan::PlanOpts { prefill_tokens: 2048, ctx: 4096 })
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
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
    /// Copy a hybrid checkpoint into GPU memory, experts repacked for the decode GEMV, then read
    /// samples back and compare them with the checkpoint (needs --features cuda on a Spark).
    #[cfg(feature = "cuda")]
    LoadWeights {
        dir: PathBuf,
        /// Where the PLE n-gram table lives: read from the checkpoint per token, or copied whole.
        #[arg(long, value_enum, default_value = "disk")]
        ple: PleArg,
        /// Tokens of PLE rows to read in the disk-mode timing (1 to 1048576).
        #[arg(long, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(1..=1 << 20))]
        gather_tokens: u32,
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
    /// Checkpoint directory whose tensor names resolve the per-module formats (otherwise formats
    /// are matched by path from config.json alone)
    #[arg(long)]
    checkpoint: Option<PathBuf>,
}

fn build(c: &Common) -> Result<plan::Plan> {
    let fmt = match c.format {
        Fmt::Auto => None,
        Fmt::Bf16 => Some(WeightFormat::Bf16),
        Fmt::W4a16 => Some(WeightFormat::W4A16 { group: 128 }),
        Fmt::Nvfp4 => Some(WeightFormat::Nvfp4),
    };
    let text = std::fs::read_to_string(&c.config)?;
    let spec = match &c.checkpoint {
        Some(dir) => sparkforge::weights::spec_from_checkpoint(&sparkforge::weights::resolve_checkpoint_dir(dir)?, fmt, Some(&text))?,
        None => ModelSpec::from_json(&text, fmt)?,
    };
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
    for g in &p.gemv {
        match &g.fp4 {
            None => println!("  {:<16} N={:<6} K={:<6} warps/blk {} rows/warp {} grid {:<5} ~{:.1} us", g.linear, g.n, g.k, g.warps_per_block, g.rows_per_warp, g.grid, g.est_us),
            Some(f) => println!("  {:<16} N={:<6} K={:<6} NVFP4 split-K {} x {} K tiles, {} slot(s), grid {}x{}x{}  ~{:.1} us", g.linear, g.n, g.k, f.splits, f.kt_per_split, f.slots, f.grid[0], f.grid[1], f.grid[2], g.est_us),
        }
    }
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
            let (mut spec, ck) = open_checkpoint(&dir)?;
            let bytes: u64 = ck.shards.iter().map(|s| s.len).sum();
            if spec.hybrid.is_some() {
                // The tensors are checked before any plan is built; the plan then prices the PLE
                // table at the rows the check found.
                let c = sparkforge::weights::check_hybrid(&ck, &spec)?;
                sparkforge::weights::apply_checked_table(&mut spec, &c);
                let p = checkpoint_plan(&spec);
                println!("{}: {} shard(s), {} tensors, {:.2} GiB, opened and checked in {:.2} s",
                    p.model.arch, ck.shards.len(), ck.tensors.len(), gib(bytes), t0.elapsed().as_secs_f64());
                println!("{} tensors match the plan in name, dtype and shape ({:.2} GB); {} skipped tower tensor(s) ({:.2} GB); PLE table shards of {} rows",
                    c.tensors, c.bytes as f64 / 1e9, c.skipped, c.skipped_bytes as f64 / 1e9, c.table_rows);
                for (g, b) in &c.groups { println!("  {g:<16} {:>9.2} GB", *b as f64 / 1e9); }
                println!("plan prices {:.2} GB of resident text weights with the PLE table at its checked {} rows",
                    p.budget.weight_gib * (1u64 << 30) as f64 / 1e9, spec.hybrid.as_ref().and_then(|h| h.ple.checked_rows).unwrap_or(0));
                for n in p.notes.iter().filter(|n| n.contains("quantized_layers")) { println!("note: {n}"); }
                return Ok(());
            }
            let p = checkpoint_plan(&spec);
            let b = sparkforge::weights::bind(&ck, &p.model)?;
            println!("{}: {} shard(s), {} tensors, {:.2} GiB, opened and validated in {:.2} s",
                p.model.arch, ck.shards.len(), ck.tensors.len(), gib(bytes), t0.elapsed().as_secs_f64());
            println!("bound {} linears ({} fused) and {} other tensors; {} unexpected tensor(s){}",
                b.linears.len(), b.linears.iter().filter(|l| l.fused()).count(), b.others.len(), b.unexpected.len(),
                if b.unexpected.is_empty() { String::new() } else { format!(": {}", b.unexpected.iter().take(5).cloned().collect::<Vec<_>>().join(", ")) });
        }
        #[cfg(feature = "cuda")]
        Cmd::VerifyWeights { dir, cuda_include, seed } => {
            use sparkforge::verify;
            let (spec, ck) = open_checkpoint(&dir)?;
            let b = sparkforge::weights::bind(&ck, &spec)?;
            let p = checkpoint_plan(&spec);
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
        Cmd::LoadWeights { dir, ple, gather_tokens } => {
            use sparkforge::{device_weights, load_plan, weights};
            let t0 = std::time::Instant::now();
            let (spec, ck) = open_checkpoint(&dir)?;
            anyhow::ensure!(spec.hybrid.is_some(), "load-weights loads hybrid checkpoints; a dense BF16 one loads with verify-weights");
            let mode = match ple { PleArg::Disk => load_plan::PleMode::Disk, PleArg::Resident => load_plan::PleMode::Resident };
            let lp = load_plan::plan(&ck, &spec, mode)?;
            println!("{}: {} tensors checked and planned into {} GPU buffers in {:.2} s", spec.arch, lp.check.tensors, lp.buffers.len(), t0.elapsed().as_secs_f64());
            let gather_tokens = gather_tokens as usize;
            let gather_bytes = if mode == load_plan::PleMode::Disk { lp.ple_table.gather_size(gather_tokens)?.1 } else { 0 };
            let staging = lp.staging_bytes(&ck, weights::STAGING_BYTES);
            let scratch = load_plan::host_scratch_bytes(&lp, &ck, staging, &load_plan::samples(&lp, &ck));
            // Gathered rows are held twice: pinned on the host and in a device buffer.
            let overhead = gather_bytes.checked_mul(2).and_then(|g| g.checked_add(scratch))
                .ok_or_else(|| anyhow::anyhow!("the load's host overhead overflows a byte count"))?;
            println!("admission: {:.2} GiB of weights, {:.1} MiB of host scratch (staging {:.1} MiB), {:.1} MiB of gather buffers",
                gib(lp.bytes), scratch as f64 / (1u64 << 20) as f64, staging as f64 / (1u64 << 20) as f64, 2.0 * gather_bytes as f64 / (1u64 << 20) as f64);
            let avail0 = weights::mem_available();
            // Admission before any CUDA call, held until every allocation of this run is made.
            let admission = weights::admit(&weights::admission_lock()?, lp.bytes, overhead, weights::mem_available, weights::ADMISSION_WAIT)?;
            let ctx = cudarc::driver::CudaContext::new(0).map_err(|e| anyhow::anyhow!("opening GPU 0: {e}"))?;
            let (maj, min) = ctx.compute_capability()?;
            anyhow::ensure!((maj, min) == (12, 1), "expected GB10 (cc 12.1), found {maj}.{min}");
            let stream = ctx.default_stream();
            let w = device_weights::load_hybrid(&stream, &ck, &lp, &admission, overhead, staging)?;
            let gb = lp.bytes as f64 / 1e9;
            println!("PLE table: {}", match mode {
                load_plan::PleMode::Disk => format!("on disk ({:.2} GB in {} shards of {} rows, {} bytes per row)", lp.on_disk_bytes as f64 / 1e9, lp.ple_table.shards.len(), lp.ple_table.rows_per_shard, lp.ple_table.bytes_per_row()),
                load_plan::PleMode::Resident => "resident".into(),
            });
            println!("copied {:.2} GB ({:.2} GiB) into GPU memory in {:.2} s ({:.2} GB/s)", gb, gib(lp.bytes), w.load_secs, gb / w.load_secs);
            println!("MemAvailable: {} before, {} with the weights resident",
                avail0.map_or("unknown".into(), |a| format!("{:.1} GiB", gib(a))), w.mem_available_after.map_or("unknown".into(), |a| format!("{:.1} GiB", gib(a))));
            println!("read back {} pieces of {} buffers ({:.2} MB) in {:.2} s: all match the checkpoint", w.verified.pieces, w.verified.buffers, w.verified.bytes as f64 / 1e6, w.verify_secs);
            if mode == load_plan::PleMode::Disk {
                for (run, seed) in [(1, 0x5eedu64), (2, 0x5eed), (3, 0xfeed)] {
                    let g = device_weights::time_ple_gather(&ctx, &stream, &ck, &lp.ple_table, gather_tokens, seed)?;
                    println!("PLE gather run {run} (seed {seed:#x}): {} tokens x {} rows ({:.2} MB) read into pinned memory in {:.1} ms ({:.1} us/token), copied to the GPU in {:.2} ms",
                        g.tokens, g.rows / g.tokens, g.bytes as f64 / 1e6, g.gather_secs * 1e3, g.gather_secs * 1e6 / g.tokens as f64, g.copy_secs * 1e3);
                }
            }
            drop(w);
            drop(admission);
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

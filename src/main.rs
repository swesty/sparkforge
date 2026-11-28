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
    /// Greedy decode of a dense BF16 checkpoint: the prompt in one prefill pass (prefill GEMMs and
    /// attention, in chunks of the planned prompt length), then one CUDA Graph per step, with
    /// optional checks against the CPU reference and timings (needs --features cuda on a Spark).
    #[cfg(feature = "cuda")]
    Generate {
        /// Checkpoint directory containing config.json and the safetensors
        dir: PathBuf,
        /// Prompt as comma-separated token ids (the prompt is pre-tokenized)
        #[arg(long, value_delimiter = ',', required = true)]
        tokens: Vec<u32>,
        /// Tokens to generate after the prompt
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// Compare the prefill's logits and those of the next N-1 steps with the CPU reference, streamed
        /// from the checkpoint (the reference runs the whole prompt first)
        #[arg(long, default_value_t = 0)]
        check: usize,
        /// tokenizer.json to print the prompt and the output as text
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Also time decode steps at these context lengths
        #[arg(long, value_delimiter = ',')]
        bench_ctx: Vec<usize>,
        /// Also time the prefill of prompts of these lengths (the prompt repeated), from an empty cache
        #[arg(long, value_delimiter = ',')]
        bench_prefill: Vec<usize>,
        #[arg(long, default_value = "/usr/local/cuda/include")]
        cuda_include: String,
    },
}

/// Decode steps timed at each `--bench-ctx` length, after 2 warm-up steps, each from a cleared cache.
#[cfg(feature = "cuda")]
const BENCH_STEPS: usize = 32;

/// Prefills timed at each `--bench-prefill` length, after one warm-up, each from a cleared cache.
#[cfg(feature = "cuda")]
const BENCH_PREFILLS: usize = 3;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn generate(dir: &std::path::Path, prompt: &[u32], steps: usize, check: usize, tokenizer: Option<&std::path::Path>, bench_ctx: &[usize], bench_prefill: &[usize], cuda_include: &str) -> Result<()> {
    use sparkforge::forward::{prefill_bytes, runtime_bytes, Kernels, PrefillKernels, Runtime};
    use sparkforge::reference::{decode_step, CheckpointWeights, KvCache};
    use sparkforge::verify::{bf2f, max_err_scaled, TOL};
    let (spec, ck) = open_checkpoint(dir)?;
    sparkforge::reference::require_dense(&spec)?;
    let p = checkpoint_plan(&spec);
    let m = &p.model;
    let g = codegen::generate(&p);
    let b = sparkforge::weights::bind(&ck, m)?;
    let dec = tokenizer.map(sparkforge::tokenizer::Decoder::from_file).transpose()?;
    anyhow::ensure!(!prompt.is_empty() && steps > 0, "need a prompt and at least one step");
    anyhow::ensure!(prompt.iter().all(|&t| t < m.vocab), "a prompt token is past the vocabulary of {}", m.vocab);
    anyhow::ensure!(bench_prefill.iter().all(|&l| l >= 1), "a prefill length of 0");
    // The prefill predicts the first output token, so `steps` tokens take steps - 1 graph replays,
    // at positions prompt..prompt + steps - 1.
    let replays = prompt.len() + steps - 1;
    let bench_cap = bench_ctx.iter().map(|&c| c + 2 + BENCH_STEPS).chain(bench_prefill.iter().copied()).max().unwrap_or(0);
    let capacity = replays.max(bench_cap);
    anyhow::ensure!(capacity <= m.max_pos as usize, "{capacity} positions exceed max_position_embeddings {}", m.max_pos);
    let splits = p.attn.splits as usize;
    // Prefill chunks of the planned prompt length (the GEMM tiles' M), or less for shorter prompts.
    let planned = p.gemm.iter().find(|t| t.linear == "qkv_proj").map_or(2048, |t| t.m as usize);
    let chunk = planned.min(bench_prefill.iter().copied().chain([prompt.len()]).max().unwrap_or(1));
    // The host reference cache `--check` keeps for the whole prompt shares the same unified memory.
    let check = check.min(steps);
    let ref_bytes = sparkforge::reference::check_cache_bytes(m, prompt.len(), check);
    let rbytes = runtime_bytes(m, splits, capacity) + prefill_bytes(m, chunk) + ref_bytes;
    // Before any CUDA call: every tensor plus the KV caches, activations and reference cache.
    let admission = sparkforge::weights::admit_runtime(&ck, &b, rbytes)?;
    let e = sparkforge::runtime::Engine::load(&g, cuda_include)?;
    unsafe { e.ctx.disable_event_tracking() };
    // The legacy default stream cannot be captured into a graph.
    let s = e.ctx.new_stream()?;
    let dd = sparkforge::device_weights::upload_dense(&s, &ck, &b, m, &admission, rbytes)?;
    let mut rt = Runtime::new(&s, Kernels::new(&p, &g, &e)?, m, &dd, capacity)?;
    rt.enable_prefill(PrefillKernels::new(&p, &g, &e)?, chunk)?;
    // Allocated, and made resident, while admission still holds the lock it was counted under.
    let mut host_cache = KvCache::new(m, m.layers as usize, if check > 0 { prompt.len() + check } else { 1 });
    host_cache.make_resident();
    s.synchronize()?;
    let mem_after = sparkforge::weights::mem_available();
    drop(admission);
    println!("{}: {} layers, loaded {:.2} GiB in {:.1} s; KV cache {} positions ({:.2} GiB); MemAvailable {:.1} GiB before, {} after",
        m.arch, m.layers, gib(dd.bytes_copied), dd.load_secs, capacity, gib(2 * m.layers as u64 * capacity as u64 * m.kv_dim() as u64 * 2),
        gib(dd.mem_available_before), mem_after.map_or("-".into(), |v| format!("{:.1} GiB", gib(v))));
    rt.capture()?;
    if check > 0 {
        println!("reference KV cache for --check: {:.1} MiB on the host, counted in admission", ref_bytes as f64 / (1u64 << 20) as f64);
    }
    if let Some(d) = &dec {
        println!("prompt ({} tokens): {:?}", prompt.len(), d.decode(prompt));
        let unknown = d.unknown(prompt);
        if !unknown.is_empty() {
            println!("warning: the tokenizer does not define prompt ids {unknown:?} (shown as <unknown token ID>); is it this model's?");
        }
    }
    // The CPU reference streams the weights from the checkpoint; it runs every prompt position to
    // build its KV cache, and its logits are compared with the prefill's and then with the next
    // `check - 1` steps'.
    let cw = CheckpointWeights { ck: &ck, b: &b, m };
    let (mut worst, mut agree) = (0f64, 0usize);
    // Compares the GPU's logits and greedy token from the step at `pos` with the reference's
    // logits `r` there.
    let mut compare = |rt: &Runtime, pos: usize, r: &[f64]| -> Result<()> {
        let got: Vec<f32> = rt.logits()?.iter().map(|&v| bf2f(v)).collect();
        let err = max_err_scaled(&got, r);
        let ref_top = r.iter().enumerate().fold((0, f64::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
        let gpu_top = rt.tokens(pos, pos + 1)?[0] as usize;
        agree += (gpu_top == ref_top) as usize;
        worst = worst.max(err);
        println!("check position {pos}: logits error {err:.2e} (tolerance {TOL:.4}), greedy token {gpu_top}, reference argmax {ref_top}");
        Ok(())
    };
    // The prompt in one prefill pass; it ends with the first output token's argmax.
    let t_prompt = std::time::Instant::now();
    rt.prefill(prompt, 0)?;
    s.synchronize()?;
    let ttft = t_prompt.elapsed().as_secs_f64();
    let first = prompt.len();
    if check > 0 {
        let mut last = vec![];
        for (pos, &token) in prompt.iter().enumerate() {
            last = decode_step(m, &cw, &mut host_cache, token as usize, pos)?.logits;
        }
        compare(&rt, first - 1, &last)?;
    }
    // Generation: the graph carries the token and position from step to step on the device. Steps
    // still to be checked run first, one at a time.
    let mut pos = first;
    while pos < replays && pos < first - 1 + check {
        let token = rt.tokens(pos - 1, pos)?[0];
        rt.step()?;
        let r = decode_step(m, &cw, &mut host_cache, token as usize, pos)?;
        compare(&rt, pos, &r.logits)?;
        pos += 1;
    }
    let timed = replays - pos;
    let t0 = std::time::Instant::now();
    for _ in 0..timed {
        rt.step()?;
    }
    s.synchronize()?;
    let gen_secs = t0.elapsed().as_secs_f64();
    // The token predicted at position p is out[p]: the prefill gives the first one.
    let out = rt.tokens(first - 1, replays)?;
    println!("prefill: {first} tokens in chunks of {chunk}, {:.1} ms to the first token ({:.0} tok/s)", ttft * 1e3, first as f64 / ttft);
    if timed > 0 {
        println!("generated {timed} tokens in {:.3} s: {:.2} ms/token, {:.1} tok/s (graph replays at positions {pos}..{replays}; the first output token came from the prefill)",
            gen_secs, gen_secs * 1e3 / timed as f64, timed as f64 / gen_secs);
    }
    println!("output ids: {out:?}");
    if let Some(d) = &dec {
        println!("output: {:?}", d.decode(&out));
        let unknown = d.unknown(&out);
        if !unknown.is_empty() {
            println!("warning: the tokenizer does not define output ids {unknown:?} (shown as <unknown token ID>); is it this model's?");
        }
    }
    if check > 0 {
        println!("checked {check} position(s) against the CPU reference: worst logits error {worst:.2e}, greedy token matches the reference argmax in {agree}/{check}");
    }
    // Prefill timing: the prompt repeated to each length, from a zeroed cache, to the first token.
    for &l in bench_prefill {
        let toks: Vec<u32> = prompt.iter().copied().cycle().take(l).collect();
        let mut run = || -> Result<f64> {
            rt.clear_cache()?;
            s.synchronize()?;
            let t = std::time::Instant::now();
            rt.prefill(&toks, 0)?;
            s.synchronize()?;
            Ok(t.elapsed().as_secs_f64())
        };
        run()?;
        let secs = (0..BENCH_PREFILLS).map(|_| run()).sum::<Result<f64>>()? / BENCH_PREFILLS as f64;
        println!("prefill {l} tokens: {:.1} ms to the first token, {:.0} tok/s (chunks of {chunk}, mean of {BENCH_PREFILLS} after a warm-up)", secs * 1e3, l as f64 / secs);
    }
    // Timing at a context length: the attention span is what matters, not the values in it, so
    // each measurement starts from a zeroed cache (a synthetic prefix) and the same input token,
    // whatever was measured before it.
    for &c in bench_ctx {
        anyhow::ensure!(c >= 1, "a context length of 0");
        rt.clear_cache()?;
        rt.set(c, prompt[prompt.len() - 1] as usize)?;
        rt.step()?;
        rt.step()?;
        s.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..BENCH_STEPS {
            rt.step()?;
        }
        s.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / BENCH_STEPS as f64;
        // The timed steps are at positions c + 2 .. c + 2 + BENCH_STEPS, after the two warm-ups.
        let (from, to) = (c + 2, c + 2 + BENCH_STEPS);
        let ceiling = plan::decode_ceiling_over(&spec, &GB10, 2048, from, to);
        println!("ctx {c}: {ms:.2} ms/token, {:.1} tok/s (planner ceiling {ceiling:.1} over the same positions), {BENCH_STEPS} graph replays at positions {from}..{to} over a zeroed prefix",
            1e3 / ms);
    }
    anyhow::ensure!(worst < TOL, "logits differ from the CPU reference by {worst:.2e}, above tolerance {TOL:.4}");
    Ok(())
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
    for t in &p.gemm { println!("  {:<16} M={:<5} N={:<6} K={:<6} tile {}x{}x{} stages {} warps {} (64x{}) smem {:>6} ctas/SM {} wave_eff {:.2} L2 group {}", t.linear, t.m, t.n, t.k, t.bm, t.bn, t.bk, t.stages, t.warps, t.warp_n, t.smem_bytes, t.ctas_per_sm, t.wave_efficiency, t.group_n); }
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
            // Admission (lock + memory guard) before any CUDA call; held until verification has
            // freed the weights and its reference scratch.
            let admission = sparkforge::weights::admit_load(&ck, &b, true)?;
            let g = codegen::generate(&p);
            let e = sparkforge::runtime::Engine::load(&g, &cuda_include)?;
            let rep = verify::verify_weights(&p, &g, &e, &ck, &b, seed, &admission)?;
            drop(admission);
            // The baseline is the reading admission checked right before allocating, not one taken
            // before waiting for the lock (codex-review fp:69a474a7ea8f on #19).
            println!("{} ({} layers), MemAvailable before the copy {:.1} GiB", p.model.arch, p.model.layers, rep.mem_available_before_load as f64 / (1u64 << 30) as f64);
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
            let staging = lp.staging_bytes(&ck, weights::STAGING_BYTES);
            let scratch = load_plan::host_scratch_bytes(&lp, &ck, staging, &load_plan::samples(&lp, &ck));
            // The load, its read-back and the timed gather run one after another, so admission
            // reserves the largest of them, with the gather's row IDs counted.
            let overhead = load_plan::hybrid_overhead_bytes(&lp, scratch, gather_tokens)?;
            let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
            let gather = if mode == load_plan::PleMode::Disk {
                format!(", the gather peak {:.1} MiB ({gather_tokens} tokens: rows pinned and on the GPU, plus row IDs)", mib(lp.ple_table.gather_peak_bytes(gather_tokens)?))
            } else {
                String::new()
            };
            println!("admission: {:.2} GiB of weights plus {:.1} MiB of host overhead, the largest of: load scratch {:.1} MiB (staging {:.1} MiB){gather}",
                gib(lp.bytes), mib(overhead), mib(scratch), mib(staging));
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
            // "Before" is the reading admission checked right before allocating, not one taken
            // before waiting for the lock (codex-review fp:69a474a7ea8f on #19).
            println!("MemAvailable: {:.1} GiB before the copy, {} with the weights resident",
                gib(w.mem_available_before), w.mem_available_after.map_or("unknown".into(), |a| format!("{:.1} GiB", gib(a))));
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
        Cmd::Generate { dir, tokens, steps, check, tokenizer, bench_ctx, bench_prefill, cuda_include } => {
            generate(&dir, &tokens, steps, check, tokenizer.as_deref(), &bench_ctx, &bench_prefill, &cuda_include)?;
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
            println!("{:<28} {:<30} {:>10} {:>5} {:>10} {:>8}", "kernel", "case", "max err", "ok", "us", "GB/s");
            let mut fails = 0;
            for r in &rows {
                let ok = r.err < verify::TOL;
                fails += !ok as usize;
                let us = r.us.map_or("-".into(), |t| format!("{t:.1}"));
                let gbs = r.gbs.map_or("-".into(), |b| format!("{b:.0}"));
                println!("{:<28} {:<30} {:>10.2e} {:>5} {:>10} {:>8}", r.kernel, r.case, r.err, if ok { "yes" } else { "NO" }, us, gbs);
            }
            let skipped: Vec<_> = g.kernels.iter().filter(|k| k.status != "generated").map(|k| k.entry.as_str()).collect();
            if !skipped.is_empty() { println!("not generated yet (skipped): {}", skipped.join(", ")); }
            if let Err(e) = sparkforge::reference::require_dense(&p.model) { println!("decode step check skipped: {e}"); }
            if let Some(t) = verify::token_estimate_us(&p, &rows) {
                println!("decode token, sum of isolated kernel times @ ctx {}: {:.2} ms -> {:.1} tok/s (planner ceiling {:.1})",
                    p.budget.ctx, t / 1e3, 1e6 / t, p.budget.decode_tok_s_ceiling);
            }
            match sparkforge::estimate::prefill_estimate_us(&p, &rows) {
                Ok((t, tokens)) => println!("prefill, sum of isolated GEMM and attention times for a {tokens}-token prompt: {:.1} ms -> {:.0} tok/s (norms, RoPE and activations not included)",
                    t / 1e3, tokens as f64 * 1e6 / t),
                Err(e) if bench => println!("prefill estimate skipped: {e}"),
                Err(_) => {}
            }
            anyhow::ensure!(fails == 0, "{fails} check(s) above tolerance {:.4}", verify::TOL);
        }
    }
    Ok(())
}

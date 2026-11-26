//! What a hybrid (`qwen4_exp`) checkpoint becomes in GPU memory, and the host side of the copy:
//! the buffers, the bytes of each, how they are produced from the checkpoint, and the check that
//! reads them back. No CUDA here; `device_weights` drives it against the GPU through `Device`,
//! and the tests drive it against host memory.
//!
//! - **Routed experts.** Each layer's experts in NVFP4 become a few buffers holding every expert,
//!   indexed by expert number, in the layout the decode GEMV reads (`repack`):
//!   - `<experts>.gate_up.weight`: per expert, the gate rows then the up rows (one `2I` x `H`
//!     GEMV), tile-packed;
//!   - `<experts>.gate_up.scale`: their per-16 scales as S0E5M3, `(H/16, 2I)` per expert;
//!   - `<experts>.down.weight` and `<experts>.down.scale`: the `H` x `I` down projection the same way;
//!   - `<experts>.weight_scale_2`: the F32 global scales, `[E, 3]` (gate, up, down);
//!   - `<experts>.input_scale`: the F32 activation scales, `[E, 3]`, when activations are quantized.
//!
//!   Repacking moves bytes and changes no count, so each buffer holds exactly the bytes of the
//!   tensors it replaces.
//! - **Everything else** is copied as stored, one buffer per tensor, named by its canonical name.
//! - **The PLE n-gram table** (28.8 GB in the reference) is either resident (its shards copied as
//!   stored) or left on disk, where `PleTable::gather` reads the rows a token needs. The table's
//!   global scale and the small PLE tensors are resident in both modes.
use crate::config::{ModelSpec, WeightFormat};
use crate::repack::{check_fp4_shape, pack_fp4, scales_s0e5m3, unpack_fp4};
use crate::weights::{check_hybrid, text_tensors, Checkpoint, HybridCheck, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

/// Where the PLE n-gram table lives during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PleMode {
    /// Rows are read from the checkpoint as tokens need them.
    Disk,
    /// Every shard is copied into GPU memory.
    Resident,
}

/// One contiguous run of a buffer and how to produce its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    /// A whole tensor as stored (checkpoint name).
    Tensor(String),
    /// NVFP4 weights of one GEMV: the rows of `parts` stacked in order, `n` x `k` in all, packed
    /// with `repack::pack_fp4`.
    PackedFp4 { parts: Vec<String>, n: usize, k: usize },
    /// The per-16 E4M3 scales of the same stacked rows, as `(k/16, n)` S0E5M3 bytes.
    Scales { parts: Vec<String>, n: usize, k: usize },
}

impl Piece {
    fn len(&self, ck: &Checkpoint) -> u64 {
        match self {
            Piece::Tensor(t) => ck.tensors[t].len(),
            Piece::PackedFp4 { n, k, .. } => (n * k / 2) as u64,
            Piece::Scales { n, k, .. } => (n * k / 16) as u64,
        }
    }
}

/// One GPU allocation.
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Canonical tensor name, or `<experts>.gate_up.weight` and the like for a grouped buffer.
    pub name: String,
    /// Inventory group the bytes count toward (`routed_experts`, `ple_table`, ...).
    pub group: &'static str,
    pub bytes: u64,
    pub pieces: Vec<Piece>,
    /// Byte offset of each piece in the buffer.
    pub offsets: Vec<u64>,
}

/// The PLE n-gram table's shards, for reading rows from the checkpoint.
#[derive(Debug, Clone)]
pub struct PleTable {
    pub rows_per_shard: usize,
    /// Checkpoint names of each shard's tensors, in a fixed order (`weight`, then `weight_scale`
    /// when the table is quantized).
    pub shards: Vec<Vec<String>>,
    /// Bytes per row of each of those tensors (80 and 10 in the reference).
    pub row_bytes: Vec<usize>,
    /// Each n-gram head's table as (first global row, rows), in table order (16 in the reference).
    pub heads: Vec<(u64, u64)>,
}

impl PleTable {
    /// Rows across all shards (the padded table).
    pub fn rows(&self) -> u64 {
        (self.rows_per_shard * self.shards.len()) as u64
    }

    /// Bytes `gather` writes per row.
    pub fn bytes_per_row(&self) -> usize {
        self.row_bytes.iter().sum()
    }

    /// Rows and bytes `gather` reads for `tokens` tokens (one row per head each), or an error when
    /// the count is zero or the products overflow.
    pub fn gather_size(&self, tokens: usize) -> Result<(usize, u64)> {
        ensure!(tokens > 0, "a PLE gather needs at least one token");
        let rows = tokens.checked_mul(self.heads.len()).with_context(|| format!("{tokens} tokens x {} PLE heads overflow a row count", self.heads.len()))?;
        let bytes = (rows as u64).checked_mul(self.bytes_per_row() as u64).with_context(|| format!("{rows} PLE rows x {} bytes overflow a byte count", self.bytes_per_row()))?;
        Ok((rows, bytes))
    }

    /// Host bytes a timed gather of `tokens` tokens holds at its peak: the gathered bytes twice
    /// (pinned on the host and in a device buffer, which share GB10's memory) and the row IDs that
    /// `sample_rows` keeps alive alongside them, one `u64` per token and head (codex-review
    /// fp:44b41fc11198 on #19).
    pub fn gather_peak_bytes(&self, tokens: usize) -> Result<u64> {
        let (rows, bytes) = self.gather_size(tokens)?;
        (rows as u64).checked_mul(std::mem::size_of::<u64>() as u64)
            .and_then(|ids| bytes.checked_mul(2)?.checked_add(ids))
            .with_context(|| format!("the peak of a {tokens}-token PLE gather overflows a byte count"))
    }

    /// The rows of `tokens` tokens for timing a gather: per token, one row inside each head's own
    /// range (token-major, heads in order), drawn uniformly by a 64-bit LCG seeded with `seed`, as
    /// distinct n-grams spread over each head's table.
    pub fn sample_rows(&self, tokens: usize, seed: u64) -> Result<Vec<u64>> {
        let (rows, _) = self.gather_size(tokens)?;
        let mut x = seed;
        let mut out = Vec::with_capacity(rows);
        for _ in 0..tokens {
            for &(first, n) in &self.heads {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                out.push(first + (x >> 11) % n);
            }
        }
        Ok(out)
    }

    /// Reads table rows `rows` (global row `r` is row `r % rows_per_shard` of shard
    /// `r / rows_per_shard`) into `out`, one region per shard tensor: `rows.len()` x 80 weight
    /// bytes, then `rows.len()` x 10 scale bytes in the reference.
    pub fn gather(&self, ck: &Checkpoint, rows: &[u64], out: &mut [u8]) -> Result<()> {
        ensure!(out.len() == rows.len() * self.bytes_per_row(), "gathering {} PLE rows needs {} bytes, got {}", rows.len(), rows.len() * self.bytes_per_row(), out.len());
        let mut region = 0;
        for (j, &rb) in self.row_bytes.iter().enumerate() {
            let dst = &mut out[region..region + rows.len() * rb];
            for (i, &r) in rows.iter().enumerate() {
                ensure!(r < self.rows(), "PLE row {r} is outside the table's {} rows", self.rows());
                let (shard, row) = ((r / self.rows_per_shard as u64) as usize, r % self.rows_per_shard as u64);
                let t = &ck.tensors[&self.shards[shard][j]];
                ck.read_range(t, row * rb as u64, &mut dst[i * rb..(i + 1) * rb])?;
            }
            region += rows.len() * rb;
        }
        Ok(())
    }
}

/// Every buffer a hybrid checkpoint becomes on the GPU in one PLE mode.
#[derive(Debug)]
pub struct LoadPlan {
    pub mode: PleMode,
    pub buffers: Vec<Buffer>,
    /// Bytes copied into GPU memory (the sum of `buffers`).
    pub bytes: u64,
    /// Bytes of PLE table shards left on disk (0 when resident).
    pub on_disk_bytes: u64,
    pub ple_table: PleTable,
    /// The tensor check the plan was built on.
    pub check: HybridCheck,
}

impl LoadPlan {
    /// Host staging bytes `load` needs: `base`, or the largest packed or scale piece when that is
    /// larger (those are produced whole; stored tensors go in parts of any staging size).
    pub fn staging_bytes(&self, ck: &Checkpoint, base: u64) -> u64 {
        self.buffers.iter().flat_map(|b| &b.pieces).filter(|p| !matches!(p, Piece::Tensor(_))).map(|p| p.len(ck)).fold(base, u64::max)
    }
}

/// Builds the load plan for a hybrid checkpoint. `m` must be resolved onto `ck`
/// (`ModelSpec::resolve_modules`); every tensor is checked against the plan first
/// (`weights::check_hybrid`), so the buffers cover exactly the checked tensors.
pub fn plan(ck: &Checkpoint, m: &ModelSpec, mode: PleMode) -> Result<LoadPlan> {
    let check = check_hybrid(ck, m)?;
    let (Some(h), Some(r)) = (&m.hybrid, &m.resolved) else { bail!("{} is not a hybrid model resolved onto this checkpoint", m.arch) };
    let names = text_tensors(ck)?.by_name;
    let actual = |canonical: &str| -> String { names[canonical].name.clone() };
    let ple = crate::inventory::ple_paths(&r.table, h)?;
    let f = crate::plan::Formats::new(m);
    let wants = crate::inventory::hybrid(m, &f, &crate::inventory::Layout { ple: ple.clone(), table_rows: Some(check.table_rows) })?;

    // Expert groups packed for the GEMV: every layer's (and MTP layer's) NVFP4 experts.
    let mut groups: Vec<(String, WeightFormat)> = (0..h.schedule.len() as u32).map(|l| (format!("layers.{l}.mlp.experts"), f.layer_group(l, "mlp.experts"))).collect();
    groups.extend((0..h.mtp_layers).map(|l| (format!("mtp.layers.{l}.mlp.experts"), f.mtp_group(l, "mlp.experts"))));
    groups.retain(|(_, fmt)| matches!(fmt, WeightFormat::Nvfp4 | WeightFormat::Nvfp4WeightOnly));
    let packed: HashMap<String, WeightFormat> = groups.into_iter().collect();
    let shard_prefix = format!("{}.shard_", ple.table());

    let mut buffers = vec![];
    let mut on_disk_bytes = 0;
    let mut emitted = std::collections::HashSet::new();
    for w in &wants {
        let group_path = w.name.split_once(".mlp.experts.").map(|(p, _)| format!("{p}.mlp.experts"));
        if let Some((g, fmt)) = group_path.as_ref().and_then(|g| packed.get_key_value(g)) {
            if emitted.insert(g.clone()) {
                buffers.extend(expert_buffers(ck, m, g, *fmt, w.group, &actual)?);
            }
            continue;
        }
        if mode == PleMode::Disk && w.name.starts_with(&shard_prefix) {
            on_disk_bytes += names[w.name.as_str()].len();
            continue;
        }
        buffers.push(buffer(ck, w.name.clone(), w.group, vec![Piece::Tensor(actual(&w.name))]));
    }
    let bytes = buffers.iter().map(|b| b.bytes).sum();
    ensure!(bytes + on_disk_bytes == check.bytes, "the load plan covers {} bytes, but the checked tensors hold {}", bytes + on_disk_bytes, check.bytes);
    let ple_table = ple_table(&names, &ple.table(), h.ple.split_parts.unwrap_or(0) as usize, check.table_rows, &check.table_heads)?;
    Ok(LoadPlan { mode, buffers, bytes, on_disk_bytes, ple_table, check })
}

fn buffer(ck: &Checkpoint, name: String, group: &'static str, pieces: Vec<Piece>) -> Buffer {
    let mut offsets = Vec::with_capacity(pieces.len());
    let mut bytes = 0;
    for p in &pieces {
        offsets.push(bytes);
        bytes += p.len(ck);
    }
    Buffer { name, group, bytes, pieces, offsets }
}

/// The grouped buffers of one layer's NVFP4 experts at `g` (see the module comment).
fn expert_buffers(ck: &Checkpoint, m: &ModelSpec, g: &str, fmt: WeightFormat, group: &'static str, actual: &dyn Fn(&str) -> String) -> Result<Vec<Buffer>> {
    let e = m.moe.as_ref().context("a hybrid model has experts")?;
    let (n_exp, i, h) = (e.num_experts as usize, e.expert_intermediate as usize, m.hidden as usize);
    check_fp4_shape(2 * i, h).with_context(|| format!("{g}: fused gate and up projections"))?;
    check_fp4_shape(h, i).with_context(|| format!("{g}: down projection"))?;
    let t = |x: usize, proj: &str, leaf: &str| actual(&format!("{g}.{x}.{proj}.{leaf}"));
    let per = |f: &dyn Fn(usize) -> Vec<Piece>| (0..n_exp).flat_map(f).collect::<Vec<_>>();
    let mut out = vec![
        buffer(ck, format!("{g}.gate_up.weight"), group, per(&|x| vec![Piece::PackedFp4 { parts: vec![t(x, "gate_proj", "weight"), t(x, "up_proj", "weight")], n: 2 * i, k: h }])),
        buffer(ck, format!("{g}.gate_up.scale"), group, per(&|x| vec![Piece::Scales { parts: vec![t(x, "gate_proj", "weight_scale"), t(x, "up_proj", "weight_scale")], n: 2 * i, k: h }])),
        buffer(ck, format!("{g}.down.weight"), group, per(&|x| vec![Piece::PackedFp4 { parts: vec![t(x, "down_proj", "weight")], n: h, k: i }])),
        buffer(ck, format!("{g}.down.scale"), group, per(&|x| vec![Piece::Scales { parts: vec![t(x, "down_proj", "weight_scale")], n: h, k: i }])),
    ];
    let mut leaves = vec!["weight_scale_2"];
    if fmt == WeightFormat::Nvfp4 {
        leaves.push("input_scale");
    }
    for leaf in leaves {
        out.push(buffer(ck, format!("{g}.{leaf}"), group,
            per(&|x| ["gate_proj", "up_proj", "down_proj"].iter().map(|p| Piece::Tensor(t(x, p, leaf))).collect())));
    }
    Ok(out)
}

fn ple_table(names: &HashMap<&str, &TensorInfo>, table: &str, parts: usize, rows: usize, head_rows: &[u64]) -> Result<PleTable> {
    let mut shards = vec![];
    let mut row_bytes = vec![];
    for j in 0..parts {
        let mut ts = vec![];
        for (k, leaf) in ["weight", "weight_scale"].iter().enumerate() {
            let Some(t) = names.get(format!("{table}.shard_{j}.{leaf}").as_str()) else { continue };
            let rb = usize::try_from(t.len())? / rows.max(1);
            if j == 0 {
                row_bytes.push(rb);
            }
            ensure!(row_bytes.get(k) == Some(&rb), "PLE table shard {j}: {} has {rb} bytes per row, unlike shard 0", t.name);
            ts.push(t.name.clone());
        }
        shards.push(ts);
    }
    let mut heads = Vec::with_capacity(head_rows.len());
    let mut first = 0u64;
    for (i, &n) in head_rows.iter().enumerate() {
        ensure!(n > 0, "PLE head {i} has no rows");
        heads.push((first, n));
        first += n;
    }
    let total = (rows as u64).checked_mul(parts as u64).context("the PLE table's row count overflows")?;
    ensure!(first <= total, "the PLE heads hold {first} rows, more than the table's {total}");
    Ok(PleTable { rows_per_shard: rows, shards, row_bytes, heads })
}

/// Produces the bytes of `piece` into `out` (exactly its length). A `Tensor` piece may be filled in
/// parts: `off` is where `out` starts within it; other pieces are produced whole (`off` 0).
pub fn fill(ck: &Checkpoint, piece: &Piece, off: u64, out: &mut [u8]) -> Result<()> {
    match piece {
        Piece::Tensor(t) => ck.read_range(&ck.tensors[t], off, out),
        Piece::PackedFp4 { parts, n, k } => {
            ensure!(off == 0, "a packed piece is produced whole");
            pack_fp4(&stacked(ck, parts)?, *n, *k, out)
        }
        Piece::Scales { parts, n, k } => {
            ensure!(off == 0, "a scale piece is produced whole");
            scales_s0e5m3(&stacked(ck, parts)?, *n, *k, out)
        }
    }
}

/// The parts' bytes back to back: their rows stacked in order, in one allocation of exactly their
/// length (what `host_scratch_bytes` counts per piece).
fn stacked(ck: &Checkpoint, parts: &[String]) -> Result<Vec<u8>> {
    let total = parts.iter().map(|p| ck.tensors[p].len()).sum::<u64>();
    let mut out = vec![0u8; usize::try_from(total)?];
    let mut at = 0;
    for p in parts {
        let t = &ck.tensors[p];
        let n = usize::try_from(t.len())?;
        ck.read_range(t, 0, &mut out[at..at + n])?;
        at += n;
    }
    Ok(out)
}

/// GPU memory as the loader sees it: allocations it fills and reads back. `device_weights`
/// implements it with CUDA; tests implement it with host vectors.
pub trait Device {
    type Buf;
    fn alloc(&mut self, bytes: usize) -> Result<Self::Buf>;
    /// Copies `src` to `buf` at `off`. `src` may be reused as soon as this returns.
    fn write(&mut self, buf: &mut Self::Buf, off: usize, src: &[u8]) -> Result<()>;
    /// Copies `dst.len()` bytes of `buf` at `off` into `dst`, complete when this returns.
    fn read(&mut self, buf: &Self::Buf, off: usize, dst: &mut [u8]) -> Result<()>;
}

/// Threads producing pieces in parallel (pread and repack); reading and repacking 68 GB of experts
/// on one thread would dominate the load.
pub const FILL_THREADS: usize = 16;

/// Allocates and fills every buffer of `plan`, staging at most `staging` bytes on the host at a time.
/// Fails before allocating anything when `staging` is zero or smaller than a packed or scale piece
/// (see `LoadPlan::staging_bytes`).
pub fn load<D: Device>(dev: &mut D, ck: &Checkpoint, plan: &LoadPlan, staging: usize) -> Result<Vec<D::Buf>> {
    ensure!(staging > 0, "the staging buffer must hold at least one byte");
    let need = plan.staging_bytes(ck, 0);
    ensure!(need <= staging as u64, "a packed or scale piece of {need} bytes does not fit the {staging}-byte staging buffer (pieces of that kind are produced whole)");
    let mut stage = vec![0u8; staging];
    let mut out = Vec::with_capacity(plan.buffers.len());
    for b in &plan.buffers {
        let mut dst = dev.alloc(usize::try_from(b.bytes)?).with_context(|| format!("allocating {} ({} bytes)", b.name, b.bytes))?;
        fill_buffer(dev, ck, b, &mut dst, &mut stage).with_context(|| format!("copying {}", b.name))?;
        out.push(dst);
    }
    Ok(out)
}

fn fill_buffer<D: Device>(dev: &mut D, ck: &Checkpoint, b: &Buffer, dst: &mut D::Buf, stage: &mut [u8]) -> Result<()> {
    let mut i = 0;
    while i < b.pieces.len() {
        let len = b.pieces[i].len(ck) as usize;
        if len > stage.len() {
            // A tensor larger than the staging buffer goes in parts.
            let Piece::Tensor(_) = &b.pieces[i] else { bail!("piece {i} of {} ({len} bytes) does not fit the {}-byte staging buffer", b.name, stage.len()) };
            let mut done = 0;
            while done < len {
                let n = (len - done).min(stage.len());
                fill(ck, &b.pieces[i], done as u64, &mut stage[..n])?;
                dev.write(dst, b.offsets[i] as usize + done, &stage[..n])?;
                done += n;
            }
            i += 1;
            continue;
        }
        // As many consecutive pieces as fit, produced in parallel into disjoint parts of the stage.
        let start = i;
        let mut used = 0;
        while i < b.pieces.len() && used + b.pieces[i].len(ck) as usize <= stage.len() {
            used += b.pieces[i].len(ck) as usize;
            i += 1;
        }
        let mut slots: Vec<(&Piece, &mut [u8])> = vec![];
        let mut rest = &mut stage[..used];
        for p in &b.pieces[start..i] {
            let (head, tail) = rest.split_at_mut(p.len(ck) as usize);
            slots.push((p, head));
            rest = tail;
        }
        fill_parallel(ck, slots)?;
        dev.write(dst, b.offsets[start] as usize, &stage[..used])?;
    }
    Ok(())
}

fn fill_parallel(ck: &Checkpoint, slots: Vec<(&Piece, &mut [u8])>) -> Result<()> {
    run_parallel(slots, FILL_THREADS, |(p, out)| fill(ck, p, 0, out), |_| true)
}

/// Runs `work` on every item, on the calling thread and up to `threads - 1` helpers drawing from
/// one queue, so at most `threads` items are in progress at once. Helper `i` is started only if
/// `may_spawn(i)`; a helper the OS refuses to start (or `may_spawn` declines) just leaves its
/// share to the threads that did start. Returns the first error, after every thread has stopped.
fn run_parallel<T: Send>(items: Vec<T>, threads: usize, work: impl Fn(T) -> Result<()> + Sync, may_spawn: impl Fn(usize) -> bool) -> Result<()> {
    let helpers = threads.min(items.len()).saturating_sub(1);
    let queue = std::sync::Mutex::new(items.into_iter());
    let failed = std::sync::atomic::AtomicBool::new(false);
    let run = || -> Result<()> {
        loop {
            if failed.load(std::sync::atomic::Ordering::Relaxed) {
                return Ok(());
            }
            let Some(item) = queue.lock().map_err(|_| anyhow::anyhow!("a fill thread panicked"))?.next() else { return Ok(()) };
            if let Err(e) = work(item) {
                failed.store(true, std::sync::atomic::Ordering::Relaxed);
                return Err(e);
            }
        }
    };
    std::thread::scope(|sc| {
        let mut handles = vec![];
        for i in 1..=helpers {
            if !may_spawn(i) {
                break;
            }
            match std::thread::Builder::new().name(format!("fill-{i}")).spawn_scoped(sc, &run) {
                Ok(h) => handles.push(h),
                Err(_) => break,
            }
        }
        let mut first = run().err();
        for h in handles {
            let r = h.join().map_err(|_| anyhow::anyhow!("a fill thread panicked")).and_then(|r| r);
            if first.is_none() {
                first = r.err();
            }
        }
        first.map_or(Ok(()), Err)
    })
}

/// Host bytes a load and its read-back of `samples` hold at their peak, besides the copied weights
/// and the gather: the larger of the copy phase (the `staging` buffer plus one stacked checkpoint
/// copy of a packed or scale piece per fill thread) and the read-back phase (the GPU bytes and the
/// checkpoint bytes of the largest sample, plus the unpacked rows for a packed one).
pub fn host_scratch_bytes(plan: &LoadPlan, ck: &Checkpoint, staging: u64, samples: &[(usize, usize, u64, u64)]) -> u64 {
    let largest_built = plan.staging_bytes(ck, 0);
    let copy = staging + FILL_THREADS as u64 * largest_built;
    let read_back = samples.iter().map(|&(bi, pi, _, len)| match plan.buffers[bi].pieces[pi] {
        Piece::PackedFp4 { .. } => 3 * len,
        _ => 2 * len,
    }).max().unwrap_or(0);
    copy.max(read_back)
}

/// Host bytes `load-weights` reserves besides the copied weights: the largest of its phases, which
/// run one after another and free their buffers before the next starts. `scratch` is the copy and
/// read-back peak (`host_scratch_bytes`); in disk mode the timed gather of `gather_tokens` tokens
/// follows (`PleTable::gather_peak_bytes`). Summing the phases instead would count the scratch
/// while the gather runs, when it is already freed, and still miss the row IDs whenever they
/// outgrow the scratch (codex-review fp:44b41fc11198 on #19).
pub fn hybrid_overhead_bytes(plan: &LoadPlan, scratch: u64, gather_tokens: usize) -> Result<u64> {
    let gather = match plan.mode {
        PleMode::Disk => plan.ple_table.gather_peak_bytes(gather_tokens)?,
        PleMode::Resident => 0,
    };
    Ok(scratch.max(gather))
}

/// Buffers up to this size are read back whole; larger ones by their first and last piece, and a
/// stored tensor among those by its two ends.
pub const VERIFY_WHOLE_BYTES: u64 = 64 << 20;
/// A stored tensor longer than this is read back by its first and last this many bytes.
pub const VERIFY_EDGE_BYTES: u64 = 4 << 20;

/// What `verify` read back and compared.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Verified {
    pub buffers: usize,
    pub pieces: usize,
    pub bytes: u64,
}

/// The ranges `verify` reads back: `(buffer, piece, offset in the piece, length)`. Every byte of
/// a buffer up to `VERIFY_WHOLE_BYTES`; the first and last piece of a larger one (experts 0 and
/// E-1 of a grouped expert buffer), with a long stored tensor there read by its two ends.
pub fn samples(plan: &LoadPlan, ck: &Checkpoint) -> Vec<(usize, usize, u64, u64)> {
    samples_with(plan, ck, VERIFY_WHOLE_BYTES, VERIFY_EDGE_BYTES)
}

/// `samples` with the two thresholds given.
pub fn samples_with(plan: &LoadPlan, ck: &Checkpoint, whole: u64, edge: u64) -> Vec<(usize, usize, u64, u64)> {
    let mut out = vec![];
    for (bi, b) in plan.buffers.iter().enumerate() {
        let sampled = b.bytes > whole;
        let pieces: Vec<usize> = if !sampled || b.pieces.len() == 1 {
            (0..b.pieces.len()).collect()
        } else {
            vec![0, b.pieces.len() - 1]
        };
        for pi in pieces {
            let p = &b.pieces[pi];
            let len = p.len(ck);
            match p {
                Piece::Tensor(_) if sampled && len > 2 * edge => {
                    out.push((bi, pi, 0, edge));
                    out.push((bi, pi, len - edge, edge));
                }
                _ => out.push((bi, pi, 0, len)),
            }
        }
    }
    out
}

/// Reads back the `samples` of every buffer and compares them with the checkpoint: a stored tensor
/// byte for byte; packed weights after unpacking them back to the checkpoint's `[n, k/2]` rows;
/// scales against the S0E5M3 conversion of the checkpoint's scale bytes.
pub fn verify<D: Device>(dev: &mut D, ck: &Checkpoint, plan: &LoadPlan, bufs: &[D::Buf]) -> Result<Verified> {
    verify_samples(dev, ck, plan, bufs, &samples(plan, ck))
}

/// `verify` over the given samples (see `samples_with`).
pub fn verify_samples<D: Device>(dev: &mut D, ck: &Checkpoint, plan: &LoadPlan, bufs: &[D::Buf], samples: &[(usize, usize, u64, u64)]) -> Result<Verified> {
    let mut v = Verified::default();
    let mut last_buffer = None;
    for &(bi, pi, off, len) in samples {
        let b = &plan.buffers[bi];
        let p = &b.pieces[pi];
        // Buffers of exactly this sample's length, freed before the next (`host_scratch_bytes`).
        let mut got = vec![0u8; usize::try_from(len)?];
        dev.read(&bufs[bi], (b.offsets[pi] + off) as usize, &mut got)?;
        let want;
        match p {
            Piece::Tensor(_) | Piece::Scales { .. } => {
                let mut w = vec![0u8; got.len()];
                fill(ck, p, off, &mut w)?;
                want = w;
            }
            Piece::PackedFp4 { parts, n, k } => {
                // Compared in the checkpoint's own layout, so a repack error shows here too.
                want = stacked(ck, parts)?;
                let mut unpacked = vec![0u8; got.len()];
                unpack_fp4(&got, *n, *k, &mut unpacked)?;
                got = unpacked;
            }
        }
        if let Some(at) = got.iter().zip(&want).position(|(a, b)| a != b) {
            bail!("{} piece {pi} differs from the checkpoint at byte {} of the compared range (GPU {:#04x}, checkpoint {:#04x})", b.name, off + at as u64, got[at], want[at]);
        }
        v.pieces += 1;
        v.bytes += len;
        if last_buffer != Some(bi) {
            v.buffers += 1;
            last_buffer = Some(bi);
        }
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::run_parallel;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[test]
    fn work_finishes_when_helper_threads_cannot_start() {
        // No helper, one helper then a refusal, and every helper: each item runs exactly once.
        for allowed in [0, 1, 15] {
            let done = Mutex::new(vec![0u32; 200]);
            run_parallel((0..200).collect(), 16, |i: usize| {
                done.lock().unwrap()[i] += 1;
                Ok(())
            }, |h| h <= allowed).unwrap();
            assert!(done.into_inner().unwrap().iter().all(|&n| n == 1), "{allowed} helpers allowed");
        }
    }

    #[test]
    fn at_most_threads_items_run_at_once_and_the_first_error_is_returned() {
        let (now, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        run_parallel((0..64).collect(), 4, |_: usize| {
            peak.fetch_max(now.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(2));
            now.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }, |_| true).unwrap();
        assert!((1..=4).contains(&peak.load(Ordering::SeqCst)), "peak {}", peak.load(Ordering::SeqCst));
        let err = run_parallel((0..64).collect(), 4, |i: usize| {
            anyhow::ensure!(i != 7, "item {i} failed");
            Ok(())
        }, |_| true).unwrap_err();
        assert_eq!(err.to_string(), "item 7 failed");
    }
}

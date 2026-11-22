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
    let ple_table = ple_table(&names, &ple.table(), h.ple.split_parts.unwrap_or(0) as usize, check.table_rows)?;
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

fn ple_table(names: &HashMap<&str, &TensorInfo>, table: &str, parts: usize, rows: usize) -> Result<PleTable> {
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
    Ok(PleTable { rows_per_shard: rows, shards, row_bytes })
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

/// The parts' bytes back to back: their rows stacked in order.
fn stacked(ck: &Checkpoint, parts: &[String]) -> Result<Vec<u8>> {
    let mut out = vec![];
    let mut buf = vec![];
    for p in parts {
        ck.read_into(&ck.tensors[p], &mut buf)?;
        out.extend_from_slice(&buf);
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
pub fn load<D: Device>(dev: &mut D, ck: &Checkpoint, plan: &LoadPlan, staging: usize) -> Result<Vec<D::Buf>> {
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
    if slots.len() == 1 {
        let (p, out) = slots.into_iter().next().expect("one slot");
        return fill(ck, p, 0, out);
    }
    let per = slots.len().div_ceil(FILL_THREADS);
    let mut chunks: Vec<Vec<(&Piece, &mut [u8])>> = vec![];
    let mut it = slots.into_iter().peekable();
    while it.peek().is_some() {
        chunks.push(it.by_ref().take(per).collect());
    }
    std::thread::scope(|sc| {
        let handles: Vec<_> = chunks.into_iter().map(|c| sc.spawn(move || c.into_iter().try_for_each(|(p, out)| fill(ck, p, 0, out)))).collect();
        handles.into_iter().try_for_each(|h| h.join().map_err(|_| anyhow::anyhow!("a fill thread panicked"))?)
    })
}

/// Buffers up to this size are read back whole; larger ones by their first and last piece.
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

/// The ranges `verify` reads back: `(buffer, piece, offset in the piece, length)`. Every piece of
/// a buffer up to `VERIFY_WHOLE_BYTES`; the first and last piece of a larger one (experts 0 and
/// E-1 of a grouped expert buffer); a long tensor piece by its two ends.
pub fn samples(plan: &LoadPlan, ck: &Checkpoint) -> Vec<(usize, usize, u64, u64)> {
    samples_with(plan, ck, VERIFY_WHOLE_BYTES, VERIFY_EDGE_BYTES)
}

/// `samples` with the two thresholds given.
pub fn samples_with(plan: &LoadPlan, ck: &Checkpoint, whole: u64, edge: u64) -> Vec<(usize, usize, u64, u64)> {
    let mut out = vec![];
    for (bi, b) in plan.buffers.iter().enumerate() {
        let pieces: Vec<usize> = if b.bytes <= whole || b.pieces.len() == 1 {
            (0..b.pieces.len()).collect()
        } else {
            vec![0, b.pieces.len() - 1]
        };
        for pi in pieces {
            let p = &b.pieces[pi];
            let len = p.len(ck);
            match p {
                Piece::Tensor(_) if len > 2 * edge => {
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
    let mut got = vec![];
    let mut want = vec![];
    for &(bi, pi, off, len) in samples {
        let b = &plan.buffers[bi];
        let p = &b.pieces[pi];
        got.resize(len as usize, 0);
        dev.read(&bufs[bi], (b.offsets[pi] + off) as usize, &mut got)?;
        match p {
            Piece::Tensor(_) | Piece::Scales { .. } => {
                want.resize(len as usize, 0);
                fill(ck, p, off, &mut want)?;
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

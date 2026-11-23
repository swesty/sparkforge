//! Make a validated checkpoint readable by the generated kernels on GB10 (`--features cuda`).
//!
//! Every linear is copied into one GPU allocation; fused linears are packed in kernel row order
//! (q|k|v, gate|up). Reading the memory-mapped file in place through ATS was measured and
//! rejected: see openspec/changes/add-weight-loading/design.md, decision 2.
use crate::weights::{bytes_to_copy, load_overhead_bytes, mem_available, Admission, Binding, BoundLinear, Checkpoint, STAGING_BYTES};
use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

pub struct DeviceTensor {
    /// Device address kernels receive.
    pub ptr: u64,
    pub bytes: usize,
    _owned: CudaSlice<u8>,
}

pub struct DeviceWeights {
    /// Keyed by `BoundLinear::key()` (for example `layers.0.o_proj`, `lm_head`).
    pub linears: HashMap<String, DeviceTensor>,
    pub bytes_copied: u64,
    pub load_secs: f64,
    /// `MemAvailable` sampled right after the copy, while every allocation above is live.
    pub mem_available_after: Option<u64>,
}

/// Copies every linear of `b` into GPU memory. `admission` comes from `weights::admit_load`,
/// taken before the CUDA context was created. Before allocating, `upload` confirms the admission
/// covers this binding in this mode (`verifying` adds the reference's scratch) and that CUDA
/// startup left the headroom. It only borrows the admission: the caller holds it until every
/// admitted allocation is made, including the verification scratch allocated after the copy, so a
/// concurrent load's guard sees all of this one's memory as used (codex-review round 3 on #13).
pub fn upload(stream: &Arc<CudaStream>, ck: &Checkpoint, b: &Binding, admission: &Admission, verifying: bool) -> Result<DeviceWeights> {
    admission.confirm(bytes_to_copy(ck, b), load_overhead_bytes(b, verifying), mem_available)?;
    let t0 = Instant::now();
    let mut out = DeviceWeights { linears: HashMap::new(), bytes_copied: 0, load_secs: 0.0, mem_available_after: None };
    let mut buf = vec![0u8; STAGING_BYTES as usize];
    for l in &b.linears {
        let t = copy_linear(stream, ck, l, &mut buf).with_context(|| format!("copying {}", l.key()))?;
        out.bytes_copied += t.bytes as u64;
        out.linears.insert(l.key(), t);
    }
    stream.synchronize()?;
    out.load_secs = t0.elapsed().as_secs_f64();
    out.mem_available_after = mem_available();
    Ok(out)
}

fn copy_linear(stream: &Arc<CudaStream>, ck: &Checkpoint, l: &BoundLinear, buf: &mut [u8]) -> Result<DeviceTensor> {
    let total: usize = l.parts.iter().map(|p| ck.tensors[p].len() as usize).sum();
    let mut dst = unsafe { stream.alloc::<u8>(total)? };
    let mut off = 0;
    for p in &l.parts {
        let t = &ck.tensors[p];
        let mut done = 0u64;
        while done < t.len() {
            let n = (t.len() - done).min(buf.len() as u64) as usize;
            ck.read_range(t, done, &mut buf[..n])?;
            // A copy from pageable host memory returns once the source has been staged, so `buf` can
            // be reused for the next piece immediately.
            stream.memcpy_htod(&buf[..n], &mut dst.slice_mut(off..off + n))?;
            off += n;
            done += n as u64;
        }
    }
    let ptr = {
        let (p, _sync) = dst.device_ptr(stream);
        p
    };
    Ok(DeviceTensor { ptr, bytes: total, _owned: dst })
}

/// The GPU as a `load_plan::Device`: one stream-ordered allocation per buffer.
pub struct Gpu(pub Arc<CudaStream>);

impl crate::load_plan::Device for Gpu {
    type Buf = CudaSlice<u8>;
    fn alloc(&mut self, bytes: usize) -> Result<CudaSlice<u8>> {
        Ok(unsafe { self.0.alloc::<u8>(bytes)? })
    }
    fn write(&mut self, buf: &mut CudaSlice<u8>, off: usize, src: &[u8]) -> Result<()> {
        // From pageable memory the copy returns once `src` is staged, so it can be reused at once.
        self.0.memcpy_htod(src, &mut buf.slice_mut(off..off + src.len()))?;
        Ok(())
    }
    fn read(&mut self, buf: &CudaSlice<u8>, off: usize, dst: &mut [u8]) -> Result<()> {
        self.0.memcpy_dtoh(&buf.slice(off..off + dst.len()), dst)?;
        self.0.synchronize()?;
        Ok(())
    }
}

/// Host scratch a hybrid load needs beyond the staging buffer: the fill threads' stacked expert
/// parts (16 threads x about 3.3 MB in the reference) and the read-back comparison buffers (at
/// most three of `VERIFY_EDGE_BYTES` or one expert piece). Rounded up generously.
pub const HYBRID_SCRATCH_BYTES: u64 = 256 << 20;

/// A hybrid checkpoint resident in GPU memory.
pub struct HybridWeights {
    /// One allocation per `LoadPlan` buffer, in its order.
    pub buffers: Vec<CudaSlice<u8>>,
    pub load_secs: f64,
    /// `MemAvailable` right after the copy, with every buffer live.
    pub mem_available_after: Option<u64>,
    pub verified: crate::load_plan::Verified,
    pub verify_secs: f64,
}

/// Copies every buffer of `plan` into GPU memory, then reads samples back and compares them with
/// the checkpoint (`load_plan::verify`). `admission` must cover `plan.bytes` plus `overhead`; it is
/// confirmed against `MemAvailable` before the first allocation.
/// `staging` is the host staging size (`LoadPlan::staging_bytes`), which `overhead` must include.
pub fn load_hybrid(stream: &Arc<CudaStream>, ck: &Checkpoint, plan: &crate::load_plan::LoadPlan, admission: &Admission, overhead: u64, staging: u64) -> Result<HybridWeights> {
    admission.confirm(plan.bytes, overhead, mem_available)?;
    let mut gpu = Gpu(stream.clone());
    let t0 = Instant::now();
    let buffers = crate::load_plan::load(&mut gpu, ck, plan, usize::try_from(staging)?)?;
    stream.synchronize()?;
    let load_secs = t0.elapsed().as_secs_f64();
    let mem_available_after = mem_available();
    let t1 = Instant::now();
    let verified = crate::load_plan::verify(&mut gpu, ck, plan, &buffers)?;
    Ok(HybridWeights { buffers, load_secs, mem_available_after, verified, verify_secs: t1.elapsed().as_secs_f64() })
}

/// One timed run of reading PLE rows from the checkpoint into pinned memory and copying them to
/// the GPU, as disk mode does per token.
pub struct GatherTiming {
    pub tokens: usize,
    pub rows: usize,
    pub bytes: usize,
    /// Reading the rows into pinned host memory.
    pub gather_secs: f64,
    /// The copy of the gathered rows to the GPU.
    pub copy_secs: f64,
}

/// Times reading `tokens` x `heads` PLE rows (one per n-gram head per token) into pinned memory and
/// copying them to the GPU.
pub fn time_ple_gather(ctx: &Arc<cudarc::driver::CudaContext>, stream: &Arc<CudaStream>, ck: &Checkpoint, table: &crate::load_plan::PleTable,
    tokens: usize, seed: u64) -> Result<GatherTiming> {
    // Each token reads one row per head, from that head's own range of the table.
    let rows = table.sample_rows(tokens, seed)?;
    let bytes = usize::try_from(table.gather_size(tokens)?.1)?;
    let mut pinned = unsafe { ctx.alloc_pinned::<u8>(bytes)? };
    let mut dst = unsafe { stream.alloc::<u8>(bytes)? };
    let t0 = Instant::now();
    table.gather(ck, &rows, pinned.as_mut_slice()?)?;
    let gather_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    stream.memcpy_htod(&pinned, &mut dst)?;
    stream.synchronize()?;
    Ok(GatherTiming { tokens, rows: rows.len(), bytes, gather_secs, copy_secs: t1.elapsed().as_secs_f64() })
}

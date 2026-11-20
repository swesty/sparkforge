//! Make a validated checkpoint readable by the generated kernels on GB10 (`--features cuda`).
//!
//! Mapped mode hands kernels the memory-mapped address itself: GB10's GPU translates through the
//! CPU page tables (ATS), so no copy and no pinning are needed. Copied mode copies every tensor
//! into a GPU allocation. Fused linears are packed into one allocation in both modes.
use crate::weights::{check_headroom, mem_available, Binding, BoundLinear, Checkpoint};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Mapped,
    Copied,
}

pub struct DeviceTensor {
    /// Address kernels receive. For mapped tensors this is the mmap address.
    pub ptr: u64,
    pub bytes: usize,
    pub mapped: bool,
    _owned: Option<CudaSlice<u8>>,
}

pub struct DeviceWeights {
    pub mode: Mode,
    /// Keyed by `BoundLinear::key()` (for example `layers.0.o_proj`, `lm_head`).
    pub linears: HashMap<String, DeviceTensor>,
    pub bytes_mapped: u64,
    pub bytes_copied: u64,
    /// Tensors copied in mapped mode because their data was not 16-byte aligned.
    pub misaligned: usize,
    pub load_secs: f64,
}

fn aligned(ck: &Checkpoint, name: &str) -> bool {
    let t = &ck.tensors[name];
    (ck.bytes(t).as_ptr() as usize) % 16 == 0
}

/// Bytes the load will copy into GPU allocations, before allocating anything.
pub fn bytes_to_copy(ck: &Checkpoint, b: &Binding, mode: Mode) -> u64 {
    b.linears
        .iter()
        .filter(|l| mode == Mode::Copied || l.fused() || !aligned(ck, &l.parts[0]))
        .flat_map(|l| l.parts.iter())
        .map(|p| ck.tensors[p].end as u64 - ck.tensors[p].start as u64)
        .sum()
}

pub fn upload(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, ck: &Checkpoint, b: &Binding, mode: Mode) -> Result<DeviceWeights> {
    if mode == Mode::Mapped {
        let pageable = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)?;
        ensure!(pageable == 1, "mapped mode needs pageable memory access (ATS); this device reports {pageable}");
    }
    let copy = bytes_to_copy(ck, b, mode);
    match mem_available() {
        Some(avail) => check_headroom(copy, avail)?,
        None => bail!("cannot read MemAvailable from /proc/meminfo; refusing to allocate without the memory guard"),
    }
    let t0 = Instant::now();
    let mut out = DeviceWeights { mode, linears: HashMap::new(), bytes_mapped: 0, bytes_copied: 0, misaligned: 0, load_secs: 0.0 };
    for l in &b.linears {
        let t = bind_linear(stream, ck, l, mode, &mut out).with_context(|| format!("binding {}", l.key()))?;
        out.linears.insert(l.key(), t);
    }
    stream.synchronize()?;
    out.load_secs = t0.elapsed().as_secs_f64();
    Ok(out)
}

fn bind_linear(stream: &Arc<CudaStream>, ck: &Checkpoint, l: &BoundLinear, mode: Mode, out: &mut DeviceWeights) -> Result<DeviceTensor> {
    let total: usize = l.parts.iter().map(|p| ck.tensors[p].end - ck.tensors[p].start).sum();
    if mode == Mode::Mapped && !l.fused() {
        if aligned(ck, &l.parts[0]) {
            out.bytes_mapped += total as u64;
            let ptr = ck.bytes(&ck.tensors[&l.parts[0]]).as_ptr() as u64;
            return Ok(DeviceTensor { ptr, bytes: total, mapped: true, _owned: None });
        }
        out.misaligned += 1;
    }
    // Copy (or pack: q|k|v, gate|up rows in order) into one GPU allocation.
    let mut dst = unsafe { stream.alloc::<u8>(total)? };
    let mut off = 0;
    for p in &l.parts {
        let src = ck.bytes(&ck.tensors[p]);
        stream.memcpy_htod(src, &mut dst.slice_mut(off..off + src.len()))?;
        off += src.len();
    }
    out.bytes_copied += total as u64;
    let ptr = {
        let (p, _sync) = dst.device_ptr(stream);
        p
    };
    Ok(DeviceTensor { ptr, bytes: total, mapped: false, _owned: Some(dst) })
}

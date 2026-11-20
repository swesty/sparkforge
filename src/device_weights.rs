//! Make a validated checkpoint readable by the generated kernels on GB10 (`--features cuda`).
//!
//! Every linear is copied into one GPU allocation; fused linears are packed in kernel row order
//! (q|k|v, gate|up). Reading the memory-mapped file in place through ATS was measured and
//! rejected: see openspec/changes/add-weight-loading/design.md, decision 2.
use crate::weights::{bytes_to_copy, mem_available, Admission, Binding, BoundLinear, Checkpoint, STAGING_BYTES};
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
/// taken before the CUDA context was created; it is held until every allocation is made and
/// filled, so a concurrent load's guard sees this one's memory as used. Before allocating, it
/// confirms the admission covers this binding and that CUDA startup left the headroom.
pub fn upload(stream: &Arc<CudaStream>, ck: &Checkpoint, b: &Binding, admission: Admission) -> Result<DeviceWeights> {
    admission.confirm(bytes_to_copy(ck, b), mem_available)?;
    let _admission = admission;
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

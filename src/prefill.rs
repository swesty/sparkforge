//! Host launch of a generated prefill GEMM (`gemm_<linear>`), with the checks the kernel can't
//! make itself: buffer lengths for the run-time prompt length, the operand alignment its 16-byte
//! `cp.async` copies need, and a grid sized from that prompt length rather than the manifest's
//! (which is for the planned M).
use crate::codegen::KernelEntry;
use crate::plan::GemmTile;
use anyhow::{ensure, Result};
use cudarc::driver::{CudaFunction, CudaStream, CudaView, CudaViewMut, DevicePtr, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Grid of a prefill GEMM for `m` prompt rows: one block per BM x BN tile.
pub fn grid(t: &GemmTile, m: usize) -> u32 {
    (m.div_ceil(t.bm as usize) * (t.n / t.bn) as usize) as u32
}

/// C[m][N] = A[m][K] . W[N][K]^T for the `m` rows of `a`. Fails before launching unless the kernel
/// is generated, 1 <= m, `a` holds at least m x K elements, `w` exactly N x K, `c` at least m x N,
/// A and W are 16-byte aligned and C is 4-byte aligned.
#[allow(clippy::too_many_arguments)]
pub fn gemm(s: &Arc<CudaStream>, t: &GemmTile, entry: &KernelEntry, f: &CudaFunction, a: &CudaView<u16>, w: &CudaView<u16>, c: &mut CudaViewMut<u16>, m: usize) -> Result<()> {
    let (pa, pw, pc) = (a.device_ptr(s).0, w.device_ptr(s).0, c.device_ptr(s).0);
    gemm_at(s, t, entry, f, Buf { ptr: pa, len: a.len() }, Buf { ptr: pw, len: w.len() }, Buf { ptr: pc, len: c.len() }, m)
}

/// A device buffer of `len` bf16 elements at `ptr`, for operands held as raw addresses (the weights
/// a `forward::Runtime` borrows).
#[derive(Clone, Copy)]
pub struct Buf {
    pub ptr: u64,
    pub len: usize,
}

/// `gemm` on raw device addresses, with the same checks. The caller vouches that each address is a
/// live allocation of at least `len` elements.
#[allow(clippy::too_many_arguments)]
pub fn gemm_at(s: &Arc<CudaStream>, t: &GemmTile, entry: &KernelEntry, f: &CudaFunction, a: Buf, w: Buf, c: Buf, m: usize) -> Result<()> {
    let (n, k) = (t.n as usize, t.k as usize);
    ensure!(entry.status == "generated" && entry.entry == format!("gemm_{}", t.linear), "{} is not a generated prefill GEMM for {}", entry.entry, t.linear);
    ensure!((1..=i32::MAX as usize).contains(&m), "a prompt of {m} rows");
    ensure!(a.len >= m * k, "A has {} elements, {m} rows of {k} need {}", a.len, m * k);
    ensure!(w.len == n * k, "W has {} elements, the weights are {n} x {k}", w.len);
    ensure!(c.len >= m * n, "C has {} elements, {m} rows of {n} need {}", c.len, m * n);
    let addr = |p: u64, align: u64, what: &str| -> Result<()> {
        ensure!(p % align == 0, "{what} at {p:#x} is not {align}-byte aligned (the kernel's copies need it)");
        Ok(())
    };
    addr(a.ptr, 16, "A")?;
    addr(w.ptr, 16, "W")?;
    addr(c.ptr, 4, "C")?;
    let m32 = m as i32;
    let cfg = LaunchConfig { grid_dim: (grid(t, m), 1, 1), block_dim: (entry.block, 1, 1), shared_mem_bytes: entry.dyn_smem };
    unsafe { s.launch_builder(f).arg(&a.ptr).arg(&w.ptr).arg(&c.ptr).arg(&m32).launch(cfg)? };
    Ok(())
}

//! Host launch of a generated prefill GEMM (`gemm_<linear>`), with the checks the kernel can't
//! make itself: buffer lengths for the run-time prompt length, the operand alignment its 16-byte
//! `cp.async` copies need, and a grid sized from that prompt length rather than the manifest's
//! (which is for the planned M).
use crate::codegen::{Generated, KernelEntry};
use crate::plan::{GemmTile, Plan};
use crate::runtime::Engine;
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, CudaView, CudaViewMut, DevicePtr, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Grid of a prefill GEMM for `m` prompt rows: one block per BM x BN tile.
pub fn grid(t: &GemmTile, m: usize) -> u32 {
    (m.div_ceil(t.bm as usize) * (t.n / t.bn) as usize) as u32
}

/// One generated prefill GEMM: its plan tile, its manifest entry and its loaded function, all found
/// by the one name `gemm_<linear>`, so a launch can't check one projection's shapes and run another
/// projection's kernel (which uses its own compiled N stride). Built only by `Gemm::new`.
pub struct Gemm<'a> {
    tile: &'a GemmTile,
    entry: &'a KernelEntry,
    func: &'a CudaFunction,
    ctx: &'a Arc<CudaContext>,
}

impl<'a> Gemm<'a> {
    /// The GEMM of `linear` in `p`; fails unless the plan has its tile and `g`/`e` its generated
    /// kernel.
    pub fn new(p: &'a Plan, g: &'a Generated, e: &'a Engine, linear: &str) -> Result<Self> {
        let name = format!("gemm_{linear}");
        let tile = p.gemm.iter().find(|t| t.linear == linear).with_context(|| format!("the plan has no {name} tile"))?;
        let entry = g.kernels.iter().find(|k| k.entry == name && k.status == "generated").with_context(|| format!("{name} was not generated"))?;
        let func = e.funcs.get(&name).with_context(|| format!("{name} is not loaded"))?;
        Ok(Self { tile, entry, func, ctx: &e.ctx })
    }
    pub fn tile(&self) -> &GemmTile {
        self.tile
    }
    pub fn entry(&self) -> &KernelEntry {
        self.entry
    }

    /// The launch for `m` rows with these operand lengths and addresses, or why it would be unsafe.
    fn check(&self, s: &CudaStream, a: (u64, usize), w: (u64, usize), c: (u64, usize), m: usize) -> Result<LaunchConfig> {
        let (n, k) = (self.tile.n as usize, self.tile.k as usize);
        ensure!(Arc::ptr_eq(s.context(), self.ctx), "{} was loaded in another CUDA context than the stream's", self.entry.entry);
        ensure!((1..=i32::MAX as usize).contains(&m), "a prompt of {m} rows");
        ensure!(a.1 >= m * k, "A has {} elements, {m} rows of {k} need {}", a.1, m * k);
        ensure!(w.1 == n * k, "W has {} elements, the weights are {n} x {k}", w.1);
        ensure!(c.1 >= m * n, "C has {} elements, {m} rows of {n} need {}", c.1, m * n);
        let aligned = |p: u64, align: u64, what: &str| -> Result<()> {
            ensure!(p.is_multiple_of(align), "{what} at {p:#x} is not {align}-byte aligned (the kernel's copies need it)");
            Ok(())
        };
        aligned(a.0, 16, "A")?;
        aligned(w.0, 16, "W")?;
        aligned(c.0, 4, "C")?;
        Ok(LaunchConfig { grid_dim: (grid(self.tile, m), 1, 1), block_dim: (self.entry.block, 1, 1), shared_mem_bytes: self.entry.dyn_smem })
    }
}

/// C[m][N] = A[m][K] . W[N][K]^T for the `m` rows of `a`. Fails before launching unless 1 <= m, `a`
/// holds at least m x K elements, `w` exactly N x K, `c` at least m x N, A and W are 16-byte
/// aligned, C is 4-byte aligned, and `s` is in the context the kernel was loaded in. The views are
/// passed to the launch, so with cudarc's event tracking a later reader of `c` on another stream
/// waits for this write.
pub fn gemm(s: &Arc<CudaStream>, gm: &Gemm, a: &CudaView<u16>, w: &CudaView<u16>, c: &mut CudaViewMut<u16>, m: usize) -> Result<()> {
    // Addresses for the alignment check only; the launch below takes the views themselves.
    let (pa, pw, pc) = (a.device_ptr(s).0, w.device_ptr(s).0, c.device_ptr(s).0);
    let cfg = gm.check(s, (pa, a.len()), (pw, w.len()), (pc, c.len()), m)?;
    let m32 = m as i32;
    unsafe { s.launch_builder(gm.func).arg(a).arg(w).arg(c).arg(&m32).launch(cfg)? };
    Ok(())
}

/// A device buffer of `len` bf16 elements at `ptr`, for operands held as raw addresses (the weights
/// a `forward::Runtime` borrows).
#[derive(Clone, Copy)]
pub struct Buf {
    pub ptr: u64,
    pub len: usize,
}

/// `gemm` on raw device addresses, with the same checks; nothing tracks these buffers.
///
/// # Safety
///
/// Each `Buf` must name a live allocation of at least `len` elements in the stream's context, and
/// stay allocated until the launch has finished on `s`. Other streams that read `c` or write `a`,
/// `w` or `c` must be ordered against `s` by the caller (no event tracking covers raw addresses).
pub unsafe fn gemm_raw(s: &Arc<CudaStream>, gm: &Gemm, a: Buf, w: Buf, c: Buf, m: usize) -> Result<()> {
    let cfg = gm.check(s, (a.ptr, a.len), (w.ptr, w.len), (c.ptr, c.len), m)?;
    let m32 = m as i32;
    unsafe { s.launch_builder(gm.func).arg(&a.ptr).arg(&w.ptr).arg(&c.ptr).arg(&m32).launch(cfg)? };
    Ok(())
}

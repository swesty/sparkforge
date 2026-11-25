//! On-GPU check of every generated kernel against a CPU reference, plus optional timing.
//! Needs `--features cuda` and a GB10. Inputs are random bf16 in [-1, 1).
//!
//! Error metric, per element: |gpu - ref| / (|ref| + 0.01 * max|ref|). A kernel passes when the
//! worst element is below 1/64 (about two bf16 ulps). References are computed in f64 from the
//! same bf16 inputs the GPU sees, so input rounding is not counted as kernel error.
use crate::codegen::{Generated, KernelEntry};
use crate::plan::Plan;
use crate::runtime::Engine;
use anyhow::{Context, Result};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

pub const TOL: f64 = 1.0 / 64.0;

pub fn f2bf(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
pub fn bf2f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn bf16(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| f2bf(self.unif())).collect()
    }
}

/// Worst per-element error; NaN or inf on the GPU side counts as infinite error.
pub fn max_err(gpu: &[f32], r: &[f64]) -> f64 {
    assert_eq!(gpu.len(), r.len());
    let amax = r.iter().fold(0f64, |a, &b| a.max(b.abs())).max(1e-30);
    gpu.iter()
        .zip(r)
        .map(|(&g, &r)| {
            let d = (g as f64 - r).abs() / (r.abs() + 1e-2 * amax);
            if d.is_finite() { d } else { f64::INFINITY }
        })
        .fold(0.0, f64::max)
}

pub use crate::estimate::{token_estimate_us, Row};

pub struct Opts {
    pub bench: bool,
    pub seed: u64,
}

fn cfg(grid: [u32; 3], block: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid[0], grid[1], grid[2]), block_dim: (block, 1, 1), shared_mem_bytes: smem }
}

fn time_us(s: &Arc<CudaStream>, iters: u32, mut f: impl FnMut(u32) -> Result<()>) -> Result<f64> {
    for i in 0..3 {
        f(i)?;
    }
    let a = s.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    for i in 0..iters {
        f(i)?;
    }
    let b = s.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    b.synchronize()?;
    Ok(a.elapsed_ms(&b)? as f64 * 1e3 / iters as f64)
}

fn to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&h| bf2f(h)).collect()
}

struct Ctx<'a> {
    p: &'a Plan,
    g: &'a Generated,
    e: &'a Engine,
    o: &'a Opts,
    rng: Rng,
    rows: Vec<Row>,
}

/// Looks up a generated kernel and its loaded function. Borrows `g`/`e`, not the `Ctx`, so the
/// result can be held across `&mut self` calls.
fn kernel<'a>(g: &'a Generated, e: &'a Engine, entry: &str) -> Option<(&'a KernelEntry, &'a CudaFunction)> {
    let k = g.kernels.iter().find(|k| k.entry == entry && k.status == "generated")?;
    Some((k, e.funcs.get(entry)?))
}

impl Ctx<'_> {

    fn push(&mut self, kernel: &str, case: String, err: f64, us: Option<f64>, bytes: f64) {
        let gbs = us.map(|t| bytes / (t * 1e3));
        self.rows.push(Row { kernel: kernel.into(), case, err, us, gbs });
    }

    fn rmsnorm(&mut self) -> Result<()> {
        let Some((k, f)) = kernel(self.g, self.e, "rmsnorm") else { return Ok(()) };
        let s = self.e.stream.clone();
        let h = self.p.model.hidden as usize;
        let rows = 4;
        let x = self.rng.bf16(rows * h);
        let w = self.rng.bf16(h);
        let eps = self.p.model.rms_eps as f32;
        let (dx, dw) = (s.memcpy_stod(&x)?, s.memcpy_stod(&w)?);
        let mut dy = s.alloc_zeros::<u16>(rows * h)?;
        unsafe { s.launch_builder(f).arg(&dx).arg(&dw).arg(&mut dy).arg(&eps).launch(cfg([rows as u32, 1, 1], k.block, 0))? };
        let y = to_f32(&s.memcpy_dtov(&dy)?);
        let mut r = vec![0f64; rows * h];
        for row in 0..rows {
            let xs = &x[row * h..(row + 1) * h];
            let ss: f64 = xs.iter().map(|&v| (bf2f(v) as f64).powi(2)).sum();
            let inv = 1.0 / (ss / h as f64 + eps as f64).sqrt();
            for i in 0..h {
                r[row * h + i] = bf2f(xs[i]) as f64 * inv * bf2f(w[i]) as f64;
            }
        }
        let us = if self.o.bench {
            Some(time_us(&s, 200, |_| {
                unsafe { s.launch_builder(f).arg(&dx).arg(&dw).arg(&mut dy).arg(&eps).launch(cfg([1, 1, 1], k.block, 0))? };
                Ok(())
            })?)
        } else { None };
        self.push("rmsnorm", format!("{rows} rows x {h}"), max_err(&y, &r), us, (3 * h * 2) as f64);
        Ok(())
    }

    fn rope(&mut self) -> Result<()> {
        let Some((k, f)) = kernel(self.g, self.e, "rope_kv_append") else { return Ok(()) };
        let s = self.e.stream.clone();
        let m = &self.p.model;
        let (hd, heads, kvh) = (m.head_dim as usize, m.heads as usize, m.kv_heads as usize);
        let n = (heads + 2 * kvh) * hd;
        // inv_freq comes from the generator (host-side, unit-tested in tests/plan.rs); this check
        // covers the kernel: precise sin/cos at large angles, the rotation, and the cache writes.
        let inv_freq = crate::codegen::rope_inv_freq(m);
        let mut positions: Vec<usize> = vec![1, 4095, 32767, m.max_pos as usize - 1];
        positions.retain(|&p| p < m.max_pos as usize);
        positions.dedup();
        for pos in positions {
            let qkv = self.rng.bf16(n);
            let mut dq = s.memcpy_stod(&qkv)?;
            let slots = (pos + 1) * kvh * hd;
            let mut kc = s.alloc_zeros::<u16>(slots)?;
            let mut vc = s.alloc_zeros::<u16>(slots)?;
            let p32 = pos as i32;
            unsafe {
                s.launch_builder(f).arg(&mut dq).arg(&mut kc).arg(&mut vc).arg(&p32)
                    .launch(cfg(k.grid, k.block, 0))?
            };
            let off = pos * kvh * hd;
            let mut got = to_f32(&s.memcpy_dtov(&dq)?);
            got.extend(to_f32(&s.memcpy_dtov(&kc.slice(off..off + kvh * hd))?));
            got.extend(to_f32(&s.memcpy_dtov(&vc.slice(off..off + kvh * hd))?));
            // Reference follows HF: inv_freq and pos*inv_freq in f32, then an exact sin/cos.
            let mut r: Vec<f64> = qkv.iter().map(|&v| bf2f(v) as f64).collect();
            for h in 0..heads + kvh {
                for i in 0..hd / 2 {
                    let ang = (pos as f32 * inv_freq[i]) as f64;
                    let (x1, x2) = (bf2f(qkv[h * hd + i]) as f64, bf2f(qkv[h * hd + i + hd / 2]) as f64);
                    r[h * hd + i] = x1 * ang.cos() - x2 * ang.sin();
                    r[h * hd + i + hd / 2] = x2 * ang.cos() + x1 * ang.sin();
                }
            }
            let kref: Vec<f64> = r[heads * hd..(heads + kvh) * hd].to_vec();
            let vref: Vec<f64> = r[(heads + kvh) * hd..].to_vec();
            r.extend(kref);
            r.extend(vref);
            let err = max_err(&got, &r);
            let us = if self.o.bench && pos == 4095 {
                Some(time_us(&s, 200, |_| {
                    unsafe { s.launch_builder(f).arg(&mut dq).arg(&mut kc).arg(&mut vc).arg(&p32).launch(cfg(k.grid, k.block, 0))? };
                    Ok(())
                })?)
            } else { None };
            self.push("rope_kv_append", format!("pos {pos}"), err, us, (n * 4) as f64);
        }
        Ok(())
    }

    fn attention(&mut self) -> Result<()> {
        let (Some((ks, fs)), Some((kc, fc))) = (kernel(self.g, self.e, "attn_decode_split"), kernel(self.g, self.e, "attn_decode_combine")) else {
            return Ok(());
        };
        let s = self.e.stream.clone();
        let a = &self.p.attn;
        let (hd, kvh, group, splits, heads) = (a.head_dim as usize, a.kv_heads as usize, a.group as usize, a.splits as usize, a.heads as usize);
        let ctx = a.ref_ctx as usize;
        let scale = 1.0 / (hd as f32).sqrt();
        let kh = self.rng.bf16(ctx * kvh * hd);
        let vh = self.rng.bf16(ctx * kvh * hd);
        let q = self.rng.bf16(heads * hd);
        let (dk, dv, dq) = (s.memcpy_stod(&kh)?, s.memcpy_stod(&vh)?, s.memcpy_stod(&q)?);
        let mut ml = s.alloc_zeros::<f32>(heads * splits * 2)?;
        let mut acc = s.alloc_zeros::<f32>(heads * splits * hd)?;
        let mut out = s.alloc_zeros::<u16>(heads * hd)?;
        let mut lens = vec![1usize, 37, 1000, ctx];
        lens.retain(|&l| l <= ctx);
        lens.dedup();
        for len in lens {
            let l32 = len as i32;
            let run = |ml: &mut cudarc::driver::CudaSlice<f32>, acc: &mut cudarc::driver::CudaSlice<f32>, out: &mut cudarc::driver::CudaSlice<u16>| -> Result<()> {
                unsafe {
                    s.launch_builder(fs).arg(&dq).arg(&dk).arg(&dv).arg(&mut *ml).arg(&mut *acc).arg(&l32).arg(&scale)
                        .launch(cfg(ks.grid, ks.block, ks.dyn_smem))?;
                    s.launch_builder(fc).arg(&*ml).arg(&*acc).arg(&mut *out).launch(cfg(kc.grid, kc.block, 0))?;
                }
                Ok(())
            };
            run(&mut ml, &mut acc, &mut out)?;
            let got = to_f32(&s.memcpy_dtov(&out)?);
            let mut r = vec![0f64; heads * hd];
            for h in 0..heads {
                let kv = h / group;
                let qs: Vec<f64> = (0..hd).map(|d| (bf2f(q[h * hd + d]) * scale) as f64).collect();
                let sc: Vec<f64> = (0..len)
                    .map(|t| (0..hd).map(|d| qs[d] * bf2f(kh[(t * kvh + kv) * hd + d]) as f64).sum())
                    .collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
                let z: f64 = p.iter().sum();
                for d in 0..hd {
                    r[h * hd + d] = (0..len).map(|t| p[t] * bf2f(vh[(t * kvh + kv) * hd + d]) as f64).sum::<f64>() / z;
                }
            }
            let err = max_err(&got, &r);
            let us = if self.o.bench && len == ctx {
                Some(time_us(&s, 50, |_| run(&mut ml, &mut acc, &mut out))?)
            } else { None };
            self.push("attn_decode (split+combine)", format!("seq {len}, {kvh}x{splits} CTAs"), err, us, (2 * len * kvh * hd * 2) as f64);
        }
        Ok(())
    }

    fn silu(&mut self) -> Result<()> {
        let Some((k, f)) = kernel(self.g, self.e, "silu_mul") else { return Ok(()) };
        let s = self.e.stream.clone();
        let inter = self.p.model.moe.as_ref().map_or(self.p.model.intermediate, |e| e.expert_intermediate) as usize;
        let gu = self.rng.bf16(2 * inter);
        let dg = s.memcpy_stod(&gu)?;
        let mut dout = s.alloc_zeros::<u16>(inter)?;
        unsafe { s.launch_builder(f).arg(&dg).arg(&mut dout).launch(cfg(k.grid, k.block, 0))? };
        let got = to_f32(&s.memcpy_dtov(&dout)?);
        let r: Vec<f64> = (0..inter)
            .map(|i| {
                let (g, u) = (bf2f(gu[i]) as f64, bf2f(gu[inter + i]) as f64);
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        let us = if self.o.bench {
            Some(time_us(&s, 200, |_| {
                unsafe { s.launch_builder(f).arg(&dg).arg(&mut dout).launch(cfg(k.grid, k.block, 0))? };
                Ok(())
            })?)
        } else { None };
        self.push("silu_mul", format!("inter {inter}"), max_err(&got, &r), us, (inter * 6) as f64);
        Ok(())
    }

    /// One NVFP4 GEMV (gpu-verification: checkpoint-format inputs against an independent f64
    /// reference, edge scales, distinct segment scales, unordered expert ids, bitwise replay), then
    /// with `--bench` its timing over DRAM-resident weights.
    fn fp4_gemv(&mut self, entry: &str, gp: &crate::plan::GemvPlan, fp: &crate::plan::Fp4Gemv, k: &KernelEntry, f: &CudaFunction) -> Result<()> {
        use crate::fp4::{Operands, Rng as FpRng};
        let s = self.e.stream.clone();
        let (n, kk, slots, splits) = (gp.n as usize, gp.k as usize, fp.slots as usize, fp.splits as usize);
        let grouped = fp.experts > 0;
        // Twice the slots in copies, read by distinct ids that are neither contiguous nor sorted:
        // slot s reads (step * s + 3) % copies, with step odd, >= 7 and coprime with copies.
        let copies = if grouped { (2 * slots).min(fp.experts as usize) } else { 1 };
        let step = (7..).step_by(2).find(|&st| gcd(st, copies) == 1).unwrap_or(1);
        let ids: Vec<u32> = if grouped { (0..slots).map(|i| ((step * i + 3) % copies) as u32).collect() } else { vec![] };
        let mut frng = FpRng(self.rng.next() | 1);
        let ops = Operands::random(fp, n, kk, copies, &mut frng);
        let (w, sf) = ops.packed()?;
        // Several activation vectors, each checked against its own reference on the same workspace
        // and counters (no clearing between launches), so a stale partial or counter shows up as a
        // wrong round, not only as a replay mismatch.
        const ROUNDS: usize = 4;
        let xs: Vec<Vec<u16>> = (0..ROUNDS).map(|_| self.rng.bf16((slots - 1) * fp.x_stride as usize + kk)).collect();
        let refs = xs.iter().map(|x| ops.reference(fp, &ids, &x.iter().map(|&v| bf2f(v) as f64).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?;
        let tiles = n / crate::plan::FP4_TILE_N as usize;
        let (dw, dsf, dgs) = (s.memcpy_stod(&w)?, s.memcpy_stod(&sf)?, s.memcpy_stod(&ops.gscale)?);
        let ids_i32: Vec<i32> = if grouped { ids.iter().map(|&i| i as i32).collect() } else { vec![0] };
        let did = s.memcpy_stod(&ids_i32)?;
        let dxs = xs.iter().map(|x| s.memcpy_stod(x)).collect::<Result<Vec<_>, _>>()?;
        let mut dy = s.alloc_zeros::<u16>(slots * n)?;
        // Zeroed once: the last split of each tile resets its counter, so no clearing between launches.
        let mut part = s.alloc_zeros::<f32>((slots * splits * n).max(1))?;
        let mut cnt = s.alloc_zeros::<u32>((slots * tiles).max(1))?;
        let launch = |dx: &cudarc::driver::CudaSlice<u16>, dy: &mut cudarc::driver::CudaSlice<u16>, part: &mut cudarc::driver::CudaSlice<f32>, cnt: &mut cudarc::driver::CudaSlice<u32>| -> Result<()> {
            unsafe {
                s.launch_builder(f).arg(&dw).arg(&dsf).arg(&dgs).arg(&did).arg(dx).arg(&mut *dy).arg(&mut *part).arg(&mut *cnt)
                    .launch(cfg(k.grid, k.block, k.dyn_smem))?
            };
            Ok(())
        };
        let mut err = 0f64;
        let mut first = vec![];
        for (r, dx) in dxs.iter().enumerate() {
            launch(dx, &mut dy, &mut part, &mut cnt)?;
            let out = s.memcpy_dtov(&dy)?;
            err = err.max(max_err(&to_f32(&out), &refs[r]));
            if r == 0 {
                first = out;
            }
        }
        let mut case = format!("N {n} K {kk}, {splits} split(s), {slots} slot(s), {ROUNDS} inputs");
        if splits > 1 {
            // 63 more launches alternating the inputs, then the first input once more over a NaN
            // sentinel: a split that never writes (a stale counter) or a stale partial cannot pass.
            for i in 0..63 {
                launch(&dxs[1 + i % (ROUNDS - 1)], &mut dy, &mut part, &mut cnt)?;
            }
            s.memcpy_htod(&vec![0xffffu16; slots * n], &mut dy)?;
            launch(&dxs[0], &mut dy, &mut part, &mut cnt)?;
            let again = s.memcpy_dtov(&dy)?;
            let same = first == again;
            case += if same { ", replay bitwise identical" } else { ", REPLAY DIFFERS" };
            if !same {
                err = f64::INFINITY;
            }
        }
        // An activation at the BF16 limit times a small block scale (`Operands::overflow_case`):
        // the rows `OVERFLOW_ROWS` of each slot must be finite and match the reference.
        {
            use crate::fp4::OVERFLOW_ROWS;
            let (big, bx) = ops.overflow_case(fp, slots);
            let (bw, bsf) = big.packed()?;
            let br = big.reference(fp, &ids, &bx.iter().map(|&v| bf2f(v) as f64).collect::<Vec<_>>())?;
            let (dbw, dbsf, dbx, dbgs) = (s.memcpy_stod(&bw)?, s.memcpy_stod(&bsf)?, s.memcpy_stod(&bx)?, s.memcpy_stod(&big.gscale)?);
            unsafe {
                s.launch_builder(f).arg(&dbw).arg(&dbsf).arg(&dbgs).arg(&did).arg(&dbx).arg(&mut dy).arg(&mut part).arg(&mut cnt)
                    .launch(cfg(k.grid, k.block, k.dyn_smem))?
            };
            let out = to_f32(&s.memcpy_dtov(&dy)?);
            let at: Vec<usize> = (0..slots).flat_map(|sl| OVERFLOW_ROWS.map(|r| sl * n + r)).collect();
            let e = max_err(&at.iter().map(|&i| out[i]).collect::<Vec<_>>(), &at.iter().map(|&i| br[i]).collect::<Vec<_>>());
            case += if e <= 1.0 / 64.0 { ", BF16-max activation finite" } else { ", BF16-MAX ACTIVATION WRONG" };
            err = err.max(e);
        }
        let dx = &dxs[0];
        if grouped {
            case += &format!(", experts {ids:?}");
        }
        drop((dw, dsf));
        let bytes_one = n * kk / 2 + n * kk / 16;
        let us = if self.o.bench { Some(self.fp4_time(fp, k, f, n, kk, bytes_one, dx)?) } else { None };
        self.push(entry, case, err, us, (bytes_one * slots) as f64);
        Ok(())
    }

    /// Times an NVFP4 GEMV over weights that cannot sit in L2: a grouped one over banks of the whole
    /// expert table with each launch's experts unread for the last 3 x L2 bytes, a dense one rotating
    /// through enough copies to make 3 x L2. Weights are zero: the time does not depend on them.
    #[allow(clippy::too_many_arguments)]
    fn fp4_time(&mut self, fp: &crate::plan::Fp4Gemv, k: &KernelEntry, f: &CudaFunction, n: usize, kk: usize, bytes_one: usize,
        dx: &cudarc::driver::CudaSlice<u16>) -> Result<f64> {
        let s = self.e.stream.clone();
        let (slots, splits) = (fp.slots as usize, fp.splits as usize);
        let tiles = n / crate::plan::FP4_TILE_N as usize;
        let l2 = self.p.target.l2_bytes as usize;
        let iters = 64usize;
        // Grouped: banks of the whole expert table, each launch's experts disjoint from every launch
        // in the last 3 x L2 bytes (`fp4::bench_sets`). Dense: enough copies to make 3 x L2.
        let experts = fp.experts as usize;
        let (banks, sets) = if experts > 0 {
            crate::fp4::bench_sets(experts, slots, bytes_one, l2, iters, &mut crate::fp4::Rng(self.rng.next() | 1))
        } else {
            (0, vec![])
        };
        let copies = if experts > 0 { banks * experts } else if bytes_one < 3 * l2 { (3 * l2).div_ceil(bytes_one) } else { 1 };
        let (wb, sb) = (n * kk / 2, n * kk / 16);
        let dw = s.alloc_zeros::<u8>(copies * wb)?;
        let dsf = s.alloc_zeros::<u8>(copies * sb)?;
        let dgs = s.alloc_zeros::<f32>(copies.max(1) * 3)?;
        let mut dy = s.alloc_zeros::<u16>(slots * n)?;
        let mut part = s.alloc_zeros::<f32>((slots * splits * n).max(1))?;
        let mut cnt = s.alloc_zeros::<u32>((slots * tiles).max(1))?;
        if experts > 0 {
            let flat: Vec<i32> = sets.iter().flat_map(|(_, ids)| ids.iter().map(|&i| i as i32)).collect();
            let dsets = s.memcpy_stod(&flat)?;
            time_us(&s, iters as u32, |i| {
                let i = i as usize % iters;
                let b = sets[i].0;
                let ids = dsets.slice(i * slots..(i + 1) * slots);
                let (w, sf) = (dw.slice(b * experts * wb..(b + 1) * experts * wb), dsf.slice(b * experts * sb..(b + 1) * experts * sb));
                unsafe {
                    s.launch_builder(f).arg(&w).arg(&sf).arg(&dgs).arg(&ids).arg(dx).arg(&mut dy).arg(&mut part).arg(&mut cnt)
                        .launch(cfg(k.grid, k.block, k.dyn_smem))?
                };
                Ok(())
            })
        } else {
            let did = s.alloc_zeros::<i32>(1)?;
            time_us(&s, 30.max(copies as u32), |i| {
                let c = i as usize % copies;
                let (w, sf) = (dw.slice(c * wb..(c + 1) * wb), dsf.slice(c * sb..(c + 1) * sb));
                unsafe {
                    s.launch_builder(f).arg(&w).arg(&sf).arg(&dgs).arg(&did).arg(dx).arg(&mut dy).arg(&mut part).arg(&mut cnt)
                        .launch(cfg(k.grid, k.block, k.dyn_smem))?
                };
                Ok(())
            })
        }
    }

    fn gemvs(&mut self) -> Result<()> {
        let s = self.e.stream.clone();
        let l2 = self.p.target.l2_bytes as usize;
        for gp in self.p.gemv.clone() {
            let entry = format!("gemv_{}", gp.linear);
            let Some((k, f)) = kernel(self.g, self.e, &entry) else { continue };
            if let Some(fp) = &gp.fp4 {
                self.fp4_gemv(&entry, &gp, fp, k, f)?;
                continue;
            }
            let (n, kk) = (gp.n as usize, gp.k as usize);
            let w = self.rng.bf16(n * kk);
            let x = self.rng.bf16(kk);
            let (dw, dx) = (s.memcpy_stod(&w)?, s.memcpy_stod(&x)?);
            let mut dy = s.alloc_zeros::<u16>(n)?;
            unsafe { s.launch_builder(f).arg(&dw).arg(&dx).arg(&mut dy).launch(cfg(k.grid, k.block, k.dyn_smem))? };
            let got = to_f32(&s.memcpy_dtov(&dy)?);
            let xf: Vec<f64> = x.iter().map(|&v| bf2f(v) as f64).collect();
            let mut r = vec![0f64; n];
            let chunk = n.div_ceil(16);
            std::thread::scope(|sc| {
                for (ci, out) in r.chunks_mut(chunk).enumerate() {
                    let (w, xf) = (&w, &xf);
                    sc.spawn(move || {
                        for (j, o) in out.iter_mut().enumerate() {
                            let row = (ci * chunk + j) * kk;
                            *o = (0..kk).map(|c| bf2f(w[row + c]) as f64 * xf[c]).sum();
                        }
                    });
                }
            });
            drop(w);
            drop(dw);
            let err = max_err(&got, &r);
            let bytes = n * kk * 2;
            let us = if self.o.bench {
                // Rotate through enough copies that the weights cannot stay resident in the 24 MB L2.
                let copies = if bytes < 3 * l2 { (3 * l2).div_ceil(bytes) } else { 1 };
                let big = s.alloc_zeros::<u16>(copies * n * kk)?;
                Some(time_us(&s, 30.max(copies as u32), |i| {
                    let off = (i as usize % copies) * n * kk;
                    let v = big.slice(off..off + n * kk);
                    unsafe { s.launch_builder(f).arg(&v).arg(&dx).arg(&mut dy).launch(cfg(k.grid, k.block, k.dyn_smem))? };
                    Ok(())
                })?)
            } else { None };
            self.push(&entry, format!("N {n} K {kk}, rpw {}", gp.rows_per_warp), err, us, bytes as f64);
        }
        Ok(())
    }
}

/// Runs every check; returns the result rows. Callers decide pass/fail with `TOL`.
pub fn run(p: &Plan, g: &Generated, e: &Engine, o: &Opts) -> Result<Vec<Row>> {
    crate::codegen::require_generated(g, &p.model.arch)?;
    // One stream, no cross-stream hazards: skip cudarc's per-argument event bookkeeping.
    unsafe { e.ctx.disable_event_tracking() };
    let mut c = Ctx { p, g, e, o, rng: Rng(o.seed | 1), rows: vec![] };
    c.rmsnorm().context("rmsnorm")?;
    c.rope().context("rope_kv_append")?;
    c.attention().context("attention")?;
    c.silu().context("silu_mul")?;
    c.gemvs().context("gemv")?;
    Ok(c.rows)
}

/// One `verify-weights` result row.
pub struct WeightRow {
    pub linear: String,
    pub case: String,
    pub err: Option<f64>,
    pub us: Option<f64>,
    pub gbs: Option<f64>,
}

pub struct WeightReport {
    pub rows: Vec<WeightRow>,
    pub load_secs: f64,
    pub bytes_copied: u64,
    /// `MemAvailable` sampled by `upload` while the weights were resident. The weights are freed
    /// when `verify_weights` returns, so a sample taken by the caller would overstate headroom.
    pub mem_available_after_load: Option<u64>,
}

/// GEMVs on real checkpoint weights against an f64 reference, timed across all layers. The caller
/// holds `admission` until this returns: the reference's scratch is allocated after the upload, and
/// the admission covers it (`load_overhead_bytes` with `verifying`).
pub fn verify_weights(
    p: &Plan,
    g: &Generated,
    e: &Engine,
    ck: &crate::weights::Checkpoint,
    b: &crate::weights::Binding,
    seed: u64,
    admission: &crate::weights::Admission,
) -> Result<WeightReport> {
    use crate::device_weights::upload;
    unsafe { e.ctx.disable_event_tracking() };
    let s = e.stream.clone();
    let mut rng = Rng(seed | 1);
    let last = p.model.layers.saturating_sub(1);
    let dw = upload(&s, ck, b, admission, true)?;
    let mut rep = WeightReport {
        rows: vec![],
        load_secs: dw.load_secs,
        bytes_copied: dw.bytes_copied,
        mem_available_after_load: dw.mem_available_after,
    };
    {
        for linear in ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"] {
            let entry = format!("gemv_{linear}");
            let (k, f) = kernel(g, e, &entry).with_context(|| format!("{entry} was not generated"))?;
            let bound: Vec<&crate::weights::BoundLinear> = b.linears.iter().filter(|l| l.linear == linear).collect();
            let (n, kk) = (bound[0].n, bound[0].k);
            let x = rng.bf16(kk);
            let xf: Vec<f64> = x.iter().map(|&v| bf2f(v) as f64).collect();
            let dx = s.memcpy_stod(&x)?;
            let mut dy = s.alloc_zeros::<u16>(n)?;
            // Correctness on the first and last layer (or lm_head).
            for bl in bound.iter().filter(|l| l.layer.is_none_or(|i| i == 0 || i == last)) {
                let w = dw.linears[&bl.key()].ptr;
                unsafe { s.launch_builder(f).arg(&w).arg(&dx).arg(&mut dy).launch(cfg(k.grid, k.block, k.dyn_smem))? };
                let got = to_f32(&s.memcpy_dtov(&dy)?);
                // Streams the weights from the checkpoint in bounded chunks (no full f64 copy).
                let r = crate::weights::reference_gemv(ck, bl, &xf)?;
                let case = match bl.layer { Some(i) => format!("layer {i}"), None => "-".into() };
                rep.rows.push(WeightRow { linear: linear.into(), case, err: Some(max_err(&got, &r)), us: None, gbs: None });
            }
            // Timing: rotate through every layer's copy of this linear (naturally larger than L2).
            let ptrs: Vec<u64> = bound.iter().map(|l| dw.linears[&l.key()].ptr).collect();
            let bytes = (n * kk * 2) as f64;
            let iters = (ptrs.len() as u32).max(20);
            let us = time_us(&s, iters, |i| {
                let w = ptrs[i as usize % ptrs.len()];
                unsafe { s.launch_builder(f).arg(&w).arg(&dx).arg(&mut dy).launch(cfg(k.grid, k.block, k.dyn_smem))? };
                Ok(())
            })?;
            rep.rows.push(WeightRow { linear: linear.into(), case: format!("all {} copies", ptrs.len()), err: None, us: Some(us), gbs: Some(bytes / (us * 1e3)) });
        }
    }
    // Give the weights back before the caller drops the admission. cudarc frees into the stream's
    // memory pool (`cuMemFreeAsync`), which returns memory to the system only at a synchronize, so
    // without this a queued load would be admitted while `MemAvailable` still counts these weights
    // and be refused for memory that is about to be free (measured on spark-5).
    drop(dw);
    s.synchronize()?;
    Ok(rep)
}

//! Host side of the NVFP4 decode GEMV check (no CUDA): random operands in the checkpoint's format,
//! their repack for the GPU, and an f64 reference that never reads the repacked bytes.
//!
//! The reference decodes E2M1 codes and E4M3 scales with decoders of its own, so a check against it
//! covers the repack (`repack.rs`) and the kernel together. `verify` runs it against the GPU, and
//! `tests/repack.rs` against a host emulation of the kernel's indexing.
use crate::plan::{Fp4Gemv, Segment};
use crate::repack::{pack_fp4, scales_s0e5m3};
use anyhow::{ensure, Result};

/// E2M1 code (4 bits) to its value: sign, 2 exponent bits, 1 mantissa bit (0, 0.5, 1, ... 6).
pub fn e2m1(code: u8) -> f64 {
    const MAG: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

/// E4M3 byte to its value (bias 7, subnormals at exponent 0; 0x7f and 0xff are NaN).
pub fn e4m3(b: u8) -> f64 {
    let (e, m) = (((b >> 3) & 0xf) as i32, (b & 7) as f64);
    if b & 0x7f == 0x7f {
        return f64::NAN;
    }
    let v = if e == 0 { m / 8.0 * 2f64.powi(-6) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) };
    if b & 0x80 != 0 { -v } else { v }
}

/// What the kernel computes from an S0E5M3 byte: the FP16 number with bits `byte << 7`.
pub fn s0e5m3(b: u8) -> f32 {
    let h = (b as u32) << 7;
    let (e, m) = ((h >> 10) & 0x1f, h & 0x3ff);
    if e == 0 { m as f32 * 2f32.powi(-24) } else { (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15) }
}

/// Global-scale column of row `row` under `segments`.
pub fn segment_col(segments: &[Segment], row: u32) -> u32 {
    segments.iter().find(|s| row < s.end).map_or(segments.last().map_or(0, |s| s.col), |s| s.col)
}

/// Random NVFP4 operands of one GEMV in checkpoint format, for `experts` copies (1 for a dense
/// GEMV): `[E][n][k/2]` code bytes, `[E][n][k/16]` E4M3 scales and the global scale table.
#[derive(Clone)]
pub struct Operands {
    pub n: usize,
    pub k: usize,
    pub experts: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    /// `[E][stride]` for a grouped GEMV (`stride` 3), else one value per column used.
    pub gscale: Vec<f32>,
}

/// The scale bytes every case carries at fixed positions: zero and the subnormals 2^-9 and 7 x 2^-9.
pub const EDGE_SCALES: [u8; 3] = [0x00, 0x01, 0x07];

/// The output rows of every slot that `Operands::overflow_case` makes overflow-prone.
pub const OVERFLOW_ROWS: [usize; 2] = [1, 2];

/// The BF16 bits of the largest finite BF16 (about 3.39e38).
pub const BF16_MAX: u16 = 0x7f7f;

/// xorshift64 (the generator `verify` uses for its BF16 inputs).
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

impl Operands {
    /// Codes uniform over all 256 byte values (E2M1 has no NaN or Inf); scales normal E4M3 with an
    /// exponent field of 5..=9 (0.25 to 7.5), with `EDGE_SCALES` at flat positions 0, 1 and the
    /// last of every copy; global scale column `c` of copy `e` is `0.75 (c + 1) (1 + e % 7 / 16)`,
    /// so every segment and every expert has its own value.
    pub fn random(g: &Fp4Gemv, n: usize, k: usize, experts: usize, rng: &mut Rng) -> Operands {
        let (cb, sb) = (n * k / 2, n * k / 16);
        let mut codes = vec![0u8; experts * cb];
        for c in codes.iter_mut() {
            *c = (rng.next() >> 56) as u8;
        }
        let mut scales = vec![0u8; experts * sb];
        for s in scales.iter_mut() {
            let r = rng.next() >> 32;
            *s = ((5 + (r % 5) as u8) << 3) | ((r >> 8) & 7) as u8;
        }
        for e in 0..experts {
            let s = &mut scales[e * sb..(e + 1) * sb];
            let last = s.len() - 1;
            s[0] = EDGE_SCALES[0];
            s[1.min(last)] = EDGE_SCALES[1];
            s[last] = EDGE_SCALES[2];
        }
        let cols = if g.gscale_stride > 0 { g.gscale_stride as usize } else { g.segments.iter().map(|s| s.col as usize + 1).max().unwrap_or(1) };
        let copies = if g.gscale_stride > 0 { experts } else { 1 };
        let gscale = (0..copies).flat_map(|e| (0..cols).map(move |c| 0.75 * (c + 1) as f32 * (1.0 + (e % 7) as f32 / 16.0))).collect();
        Operands { n, k, experts, codes, scales, gscale }
    }

    /// Cases where applying a scale only after the activation multiply overflows f32: a copy of these
    /// operands with BF16 activations that are `BF16_MAX` (about 3.39e38) at each slot's K 0 and
    /// zero elsewhere, in which, for every copy,
    /// - row 1 has code +6 at K 0 with block scale byte 0x01 (2^-9): its output is about 4e36 times
    ///   its global scale, but 6 x 3.39e38 does not fit in f32;
    /// - row 2 has code +6 at K 0 with block scale 1.0 (0x38), and the global scale of its segment is
    ///   1/8: its output is about 2.54e38, which fits in BF16, but the block-scaled sum does not fit
    ///   in f32.
    ///
    /// Other rows may overflow legitimately, so compare only `OVERFLOW_ROWS`.
    pub fn overflow_case(&self, g: &Fp4Gemv, slots: usize) -> (Operands, Vec<u16>) {
        let mut o = self.clone();
        let (kh, ks) = (self.k / 2, self.k / 16);
        let col = segment_col(&g.segments, OVERFLOW_ROWS[1] as u32) as usize;
        assert_eq!(col, segment_col(&g.segments, OVERFLOW_ROWS[0] as u32) as usize, "both rows share a segment");
        for e in 0..self.experts {
            for (row, scale) in OVERFLOW_ROWS.into_iter().zip([0x01u8, 0x38]) {
                o.scales[(e * self.n + row) * ks] = scale;
                let c = &mut o.codes[(e * self.n + row) * kh];
                *c = (*c & 0xf0) | 0x07;
            }
        }
        // The global scale of the rows' segment: one per copy when grouped, else one in all.
        if g.gscale_stride > 0 {
            for e in 0..self.experts {
                o.gscale[e * g.gscale_stride as usize + col] = 0.125;
            }
        } else {
            o.gscale[col] = 0.125;
        }
        let xs = g.x_stride as usize;
        let mut x = vec![0u16; (slots - 1) * xs + self.k];
        for s in 0..slots {
            x[s * xs] = BF16_MAX;
        }
        (o, x)
    }

    /// The GPU operands: every copy's weights tile-packed and its scales as `(k/16, n)` S0E5M3, in
    /// copy order (the weight loader's grouped buffer layout).
    pub fn packed(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        let (cb, sb) = (self.n * self.k / 2, self.n * self.k / 16);
        let mut w = vec![0u8; self.codes.len()];
        let mut sf = vec![0u8; self.scales.len()];
        for e in 0..self.experts {
            pack_fp4(&self.codes[e * cb..(e + 1) * cb], self.n, self.k, &mut w[e * cb..(e + 1) * cb])?;
            scales_s0e5m3(&self.scales[e * sb..(e + 1) * sb], self.n, self.k, &mut sf[e * sb..(e + 1) * sb])?;
        }
        Ok((w, sf))
    }

    /// f64 reference `[slots][n]`: slot `s` uses copy `ids[s]` (copy 0 for a dense GEMV, `ids`
    /// empty) and activations `x[s * x_stride ..][..k]`.
    pub fn reference(&self, g: &Fp4Gemv, ids: &[u32], x: &[f64]) -> Result<Vec<f64>> {
        let slots: Vec<usize> = if ids.is_empty() { vec![0] } else { ids.iter().map(|&i| i as usize).collect() };
        let (n, k) = (self.n, self.k);
        let xs = g.x_stride as usize;
        ensure!(x.len() >= (slots.len() - 1) * xs + k, "{} activations for {} slots of {k}", x.len(), slots.len());
        let mut out = vec![0f64; slots.len() * n];
        // Rows r0.. of slot s (copy e) into `out`.
        let rows = |s: usize, e: usize, r0: usize, out: &mut [f64]| {
            let xv = &x[s * xs..s * xs + k];
            for (i, o) in out.iter_mut().enumerate() {
                let r = r0 + i;
                let codes = &self.codes[(e * n + r) * k / 2..][..k / 2];
                let sf = &self.scales[(e * n + r) * k / 16..][..k / 16];
                let mut acc = 0f64;
                for (c, &byte) in codes.iter().enumerate() {
                    let scale = e4m3(sf[2 * c / 16]);
                    acc += e2m1(byte & 0xf) * scale * xv[2 * c] + e2m1(byte >> 4) * scale * xv[2 * c + 1];
                }
                let col = segment_col(&g.segments, r as u32) as usize;
                let gs = if g.gscale_stride > 0 { self.gscale[e * g.gscale_stride as usize + col] } else { self.gscale[col] };
                *o = acc * gs as f64;
            }
        };
        std::thread::scope(|sc| {
            for ((s, &e), chunk) in slots.iter().enumerate().zip(out.chunks_mut(n)) {
                let per = n.div_ceil(4);
                for (part, piece) in chunk.chunks_mut(per).enumerate() {
                    let rows = &rows;
                    sc.spawn(move || rows(s, e, part * per, piece));
                }
            }
        });
        Ok(out)
    }
}

/// Expert sets for timing a grouped GEMV from DRAM: `launches` sets of `slots` distinct ids, each
/// with the bank it reads (an independent copy of the whole `experts`-entry table). Each bank's
/// shuffled order is cut into `experts / slots` whole batches (no batch wraps around the table), and
/// launch `i` reads batch `(i / banks) % batches` of bank `i % banks`. A (bank, expert) pair then
/// recurs only after `banks * batches - 1` other launches, which `bench_banks` sizes to read at least
/// 3 x L2 bytes, so no launch is served from L2 a recent one left behind (random draws from one table
/// are, in proportion to how much of it L2 holds).
pub fn bench_sets(experts: usize, slots: usize, bytes_one: usize, l2: usize, launches: usize, rng: &mut Rng) -> (usize, Vec<(usize, Vec<u32>)>) {
    assert!(slots >= 1 && slots <= experts, "{slots} slots of {experts} experts");
    let banks = bench_banks(experts, slots, bytes_one, l2);
    let batches = experts / slots;
    let orders: Vec<Vec<u32>> = (0..banks).map(|_| {
        let mut o: Vec<u32> = (0..experts as u32).collect();
        for i in (1..o.len()).rev() {
            o.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        o
    }).collect();
    let sets = (0..launches).map(|i| {
        let (b, batch) = (i % banks, (i / banks) % batches);
        (b, orders[b][batch * slots..(batch + 1) * slots].to_vec())
    }).collect();
    (banks, sets)
}

/// Copies of an `experts`-entry table (`bytes_one` per expert) that `bench_sets` rotates through:
/// enough that the `banks * (experts / slots) - 1` launches between two reads of one expert copy
/// read at least 3 x L2 bytes.
pub fn bench_banks(experts: usize, slots: usize, bytes_one: usize, l2: usize) -> usize {
    let between = (3 * l2).div_ceil(slots * bytes_one);
    (between + 1).div_ceil(experts / slots).max(1)
}

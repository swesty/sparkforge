//! Load-time repacking of NVFP4 weights for the decode GEMV (host only, no CUDA).
//!
//! The layout is the one FlashInfer's SM12x BF16 x FP4 GEMV reads (`GemvBf16Fp4Sm12x`, prepared
//! by `_prepare_cute_dsl_sm12x` in `flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py` at
//! flashinfer@db2373490172, Apache-2.0), re-derived here from that source and checked byte for
//! byte against bytes its own functions produced (`tests/golden/fp4_pack_flashinfer_db23734.json`):
//!
//! - **Weights.** A ModelOpt NVFP4 weight is `[N, K/2]` bytes, two E2M1 values per byte with the
//!   lower K index in the low nibble. It is transposed to K-major (`[K/2, N]`) and cut into tiles
//!   of 16 K x 64 N (8 packed rows x 64 columns = 512 bytes). Tiles are stored K-tile major, and
//!   the 512 bytes of each are gathered into 128 little-endian `u32`s in the order a warp's lanes
//!   load them (`tile_byte`). The result is `(K/16, N * 2)` `u32`s, the same number of bytes.
//! - **Scales.** The per-16 E4M3 scales are `[N, K/16]` in the checkpoint (linear, so FlashInfer's
//!   128x4 unswizzle doesn't apply). They are transposed to `(K/16, N)` and rewritten as S0E5M3:
//!   the top byte of the scale's FP16 encoding below the sign (`(f16_bits >> 7) & 0xFF`), which the
//!   kernel turns into a multiplier with one shift. Scales must be finite and not negative.
use anyhow::{bail, ensure, Result};

/// K values per tile (one MMA K-block).
pub const TILE_K: usize = 16;
/// N rows per tile (the kernel's tile N).
pub const TILE_N: usize = 64;
/// Bytes per tile: 16 K x 64 N values at 4 bits.
pub const TILE_BYTES: usize = TILE_K * TILE_N / 2;

/// The byte of a tile's row-major `(8, 64)` source (packed K rows x N) stored at position `i` of
/// the repacked tile. Position `i = 4 * p + j` is byte `j` of the tile's `u32` number `p`; `u32`
/// `p` belongs to lane `(p / 2) % 32` of N-warp `p / 64`, which covers N rows `base_n + {0, 16, 32,
/// 48}` and packed K rows `tc_row_half + {0, 4}`.
pub fn tile_byte(i: usize) -> usize {
    let (p, j) = (i / 4, i % 4);
    let lane = (p / 2) % 32;
    let n_warp = p / 64;
    let (tc_col, tc_row_half) = (lane / 4, lane % 4);
    let base_n = n_warp * 8 + tc_col;
    const K_HALF_OFFSET: [usize; 4] = [0, 4, 0, 4];
    const N_OFFSET: [[usize; 4]; 2] = [[0, 0, 16, 16], [32, 32, 48, 48]];
    let k_half = tc_row_half + K_HALF_OFFSET[j];
    let n = base_n + N_OFFSET[p % 2][j];
    k_half * TILE_N + n
}

/// Whether an `n` x `k` NVFP4 matrix tiles into the GEMV layout.
pub fn check_fp4_shape(n: usize, k: usize) -> Result<()> {
    ensure!(n % TILE_N == 0 && k % TILE_K == 0, "an NVFP4 matrix of {n} x {k} doesn't tile into {TILE_N} N x {TILE_K} K blocks");
    Ok(())
}

/// Repacks an `n` x `k` NVFP4 weight stored as `[n, k/2]` bytes into the GEMV tile layout, writing
/// the same number of bytes to `out`.
pub fn pack_fp4(w: &[u8], n: usize, k: usize, out: &mut [u8]) -> Result<()> {
    check_fp4_shape(n, k)?;
    let kh = k / 2;
    ensure!(w.len() == n * kh && out.len() == w.len(), "NVFP4 weight of {n} x {k} needs {} bytes in and out, got {} and {}", n * kh, w.len(), out.len());
    let order: Vec<usize> = (0..TILE_BYTES).map(tile_byte).collect();
    let n_tiles = n / TILE_N;
    for kt in 0..k / TILE_K {
        for nt in 0..n_tiles {
            let dst = &mut out[(kt * n_tiles + nt) * TILE_BYTES..][..TILE_BYTES];
            for (d, &src) in dst.iter_mut().zip(&order) {
                // Source (packed K row, N) within the tile -> checkpoint byte [n][k/2].
                let (kr, nc) = (src / TILE_N, src % TILE_N);
                *d = w[(nt * TILE_N + nc) * kh + kt * (TILE_K / 2) + kr];
            }
        }
    }
    Ok(())
}

/// The inverse of `pack_fp4`: the checkpoint's `[n, k/2]` bytes from the tile layout.
pub fn unpack_fp4(packed: &[u8], n: usize, k: usize, out: &mut [u8]) -> Result<()> {
    check_fp4_shape(n, k)?;
    let kh = k / 2;
    ensure!(packed.len() == n * kh && out.len() == packed.len(), "NVFP4 weight of {n} x {k} needs {} bytes in and out", n * kh);
    let order: Vec<usize> = (0..TILE_BYTES).map(tile_byte).collect();
    let n_tiles = n / TILE_N;
    for kt in 0..k / TILE_K {
        for nt in 0..n_tiles {
            let src = &packed[(kt * n_tiles + nt) * TILE_BYTES..][..TILE_BYTES];
            for (s, &pos) in src.iter().zip(&order) {
                let (kr, nc) = (pos / TILE_N, pos % TILE_N);
                out[(nt * TILE_N + nc) * kh + kt * (TILE_K / 2) + kr] = *s;
            }
        }
    }
    Ok(())
}

/// One E4M3 scale byte as S0E5M3: bits 14..7 of its FP16 encoding (5 exponent bits with FP16's
/// bias, then the top 3 mantissa bits). Every E4M3 value is exact in FP16. For a normal scale this
/// is the byte plus 0x40; subnormals are renormalized. Negative or NaN scales are refused.
pub fn e4m3_to_s0e5m3(b: u8) -> Result<u8> {
    ensure!(b & 0x80 == 0, "NVFP4 scale byte {b:#04x} is negative");
    ensure!(b != 0x7f, "NVFP4 scale byte 0x7f is NaN");
    let (e, m) = ((b >> 3) & 0xf, b & 0x7);
    if e > 0 {
        return Ok(b + 0x40);
    }
    if m == 0 {
        return Ok(0);
    }
    // Subnormal: m * 2^-9. Normalize m = 1.f * 2^s (s = floor(log2 m)), FP16 exponent -9 + s + 15.
    let s = 7 - m.leading_zeros() as u8; // 0..=2
    let frac = (m << (3 - s)) & 0x7; // the bits below the leading one, as 3 mantissa bits
    Ok(((6 + s) << 3) | frac)
}

/// Rewrites `[n, k/16]` E4M3 scales as `(k/16, n)` S0E5M3 bytes.
pub fn scales_s0e5m3(sf: &[u8], n: usize, k: usize, out: &mut [u8]) -> Result<()> {
    ensure!(k % TILE_K == 0, "{k} inputs aren't whole 16-element scale blocks");
    let ks = k / TILE_K;
    ensure!(sf.len() == n * ks && out.len() == sf.len(), "NVFP4 scales of {n} x {k} need {} bytes in and out", n * ks);
    for r in 0..n {
        for c in 0..ks {
            out[c * n + r] = match e4m3_to_s0e5m3(sf[r * ks + c]) {
                Ok(v) => v,
                Err(e) => bail!("row {r}, scale block {c}: {e}"),
            };
        }
    }
    Ok(())
}

//! NVFP4 repacking for the decode GEMV, checked against FlashInfer's own prep functions.
//!
//! `tests/golden/fp4_pack_flashinfer_db23734.json` holds bytes produced by
//! `_cute_dsl_pack_fp4_weight` and `_e4m3_to_s0e5m3` (flashinfer@db2373490172, torch 2.13 on the
//! CPU, run by `tools/golden/fp4_pack_flashinfer.py` in the `vllm/vllm-openai:v0.32.0` image on
//! spark-5) for random `[N, K/2]` weights and positive E4M3 scales, including zero and subnormals.
use sparkforge::repack::{e4m3_to_s0e5m3, lane_elem, lane_row, pack_fp4, scales_s0e5m3, tile_byte, unpack_fp4, TILE_BYTES};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn golden() -> Vec<(usize, usize, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/golden/fp4_pack_flashinfer_db23734.json").unwrap()).unwrap();
    v["cases"].as_array().unwrap().iter().map(|c| {
        let s = |k: &str| unhex(c[k].as_str().unwrap());
        (c["n"].as_u64().unwrap() as usize, c["k"].as_u64().unwrap() as usize, s("weight"), s("scale"), s("packed"), s("scale_s0e5m3"))
    }).collect()
}

#[test]
fn packed_weights_match_flashinfer_byte_for_byte() {
    for (n, k, w, _, packed, _) in golden() {
        let mut out = vec![0u8; w.len()];
        pack_fp4(&w, n, k, &mut out).unwrap();
        assert_eq!(out, packed, "{n} x {k}");
        let mut back = vec![0u8; w.len()];
        unpack_fp4(&out, n, k, &mut back).unwrap();
        assert_eq!(back, w, "{n} x {k}: unpack is the inverse");
    }
}

#[test]
fn scales_match_flashinfer_byte_for_byte() {
    for (n, k, _, sf, _, want) in golden() {
        let mut out = vec![0u8; sf.len()];
        scales_s0e5m3(&sf, n, k, &mut out).unwrap();
        assert_eq!(out, want, "{n} x {k}");
    }
}

#[test]
fn every_e4m3_scale_converts_like_its_fp16_encoding() {
    // All 126 finite non-negative E4M3 bytes, against FP16 bits computed from the value.
    for b in 0u8..0x7f {
        let (e, m) = ((b >> 3) & 0xf, b & 7);
        let value = if e == 0 { m as f64 * 2f64.powi(-9) } else { (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7) };
        let want = if value == 0.0 { 0 } else {
            let ex = value.log2().floor() as i32;
            let mant = ((value / 2f64.powi(ex) - 1.0) * 1024.0).round() as u32;
            ((((ex + 15) as u32) << 10 | mant) >> 7) as u8
        };
        assert_eq!(e4m3_to_s0e5m3(b).unwrap(), want, "{b:#04x}");
    }
    assert!(e4m3_to_s0e5m3(0x7f).is_err(), "NaN");
    assert!(e4m3_to_s0e5m3(0x88).is_err(), "negative");
}

#[test]
fn the_tile_order_is_a_permutation() {
    let mut seen = vec![false; TILE_BYTES];
    for i in 0..TILE_BYTES {
        assert!(!std::mem::replace(&mut seen[tile_byte(i)], true), "position {i}");
    }
}

#[test]
fn the_kernel_lane_map_inverts_the_tile_order() {
    // Lane `l` loads u32s 4l..4l+3 of the tile; byte b of u32 j sits at tile position 4(4l+j)+b.
    let mut seen = vec![false; TILE_BYTES];
    for lane in 0..32 {
        for j in 0..4 {
            for b in 0..4 {
                let (slot, k_half) = lane_elem(lane, j, b);
                let (row, pos) = (lane_row(lane, slot), 4 * (4 * lane + j) + b);
                assert!(slot < 4 && k_half < 8 && row < 64, "lane {lane} j {j} b {b}");
                assert_eq!(tile_byte(pos), k_half * 64 + row, "lane {lane} j {j} b {b}");
                assert!(!std::mem::replace(&mut seen[k_half * 64 + row], true), "lane {lane} j {j} b {b} repeats");
                // The xor-1 partner holds the same row's other K half.
                let (ps, pk) = lane_elem(lane ^ 1, j, b);
                assert_eq!((lane_row(lane ^ 1, ps), pk ^ 2), (row, k_half));
            }
        }
    }
    assert!(seen.iter().all(|&s| s), "every (row, packed K row) of the tile is covered");
}

#[test]
fn shapes_that_dont_tile_are_refused() {
    let w = vec![0u8; 32 * 16];
    assert!(pack_fp4(&w, 32, 32, &mut w.clone()).is_err(), "N not a multiple of 64");
    let w = vec![0u8; 64 * 4];
    assert!(pack_fp4(&w, 64, 8, &mut w.clone()).is_err(), "K not a multiple of 16");
    let mut sf = vec![0x38u8; 64];
    sf[5] = 0xb8;
    let err = format!("{:#}", scales_s0e5m3(&sf, 64, 16, &mut vec![0; 64]).unwrap_err());
    assert!(err.contains("row 5") && err.contains("negative"), "{err}");
}

// --- Host emulation of the NVFP4 GEMV (`codegen::GEMV_FP4`) ---
//
// Walks the kernel's lanes with the same index map (`lane_elem`, `lane_row`), decodes scales the
// way the kernel does (`fp4::s0e5m3`), sums split partials in split order and applies per-segment
// global scales, then rounds to BF16. Compared with `fp4::Operands::reference`, which decodes the
// checkpoint-format bytes and never reads the repacked ones, so a wrong lane map, scale index,
// segment, expert offset or split range fails here without a GPU.
use sparkforge::fp4::{e2m1, s0e5m3, segment_col, Operands, Rng, EDGE_SCALES};
use sparkforge::plan::{Fp4Gemv, Segment};

fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b.wrapping_add(0x7fff + ((b >> 16) & 1))) & 0xffff_0000)
}

fn gemv(splits: u32, k_tiles: u32, slots: u32, experts: u32, segments: Vec<Segment>, gscale_stride: u32, x_stride: u32) -> Fp4Gemv {
    let per = k_tiles.div_ceil(splits);
    assert_eq!(k_tiles.div_ceil(per), splits, "test shape leaves a split empty");
    Fp4Gemv { splits, kt_per_split: per, slots, experts, grid: [0, splits, slots], workspace_bytes: 0, counter_bytes: 0, segments, gscale_stride, x_stride }
}

#[allow(clippy::too_many_arguments)]
fn emulate(g: &Fp4Gemv, n: usize, k: usize, w: &[u8], sf: &[u8], gscale: &[f32], ids: &[u32], x: &[f32]) -> Vec<f32> {
    let (tiles, k_tiles, s) = (n / 64, k / 16, g.splits as usize);
    let slots = g.slots as usize;
    let mut partial = vec![0f32; slots * s * n];
    for slot in 0..slots {
        let e = if g.experts > 0 { ids[slot] as usize } else { 0 };
        let (w, sf) = (&w[e * n * k / 2..][..n * k / 2], &sf[e * n * k / 16..][..n * k / 16]);
        let xv = &x[slot * g.x_stride as usize..][..k];
        for split in 0..s {
            let kt0 = split * g.kt_per_split as usize;
            for nt in 0..tiles {
                let mut acc = [[0f32; 4]; 32];
                for kt in kt0..(kt0 + g.kt_per_split as usize).min(k_tiles) {
                    for (lane, acc) in acc.iter_mut().enumerate() {
                        let at = ((kt * tiles + nt) * 32 + lane) * 16; // one 16-byte load
                        for j in 0..4 {
                            for b in 0..4 {
                                let byte = w[at + 4 * j + b];
                                let (p, kh) = lane_elem(lane, j, b);
                                let kk = kt * 16 + 2 * kh;
                                // The kernel scales each value in f16 first (exact, so f32 here).
                                let s = s0e5m3(sf[kt * n + nt * 64 + lane_row(lane, p)]);
                                let (lo, hi) = (e2m1(byte & 0xf) as f32 * s, e2m1(byte >> 4) as f32 * s);
                                acc[p] = hi.mul_add(xv[kk + 1], lo.mul_add(xv[kk], acc[p]));
                            }
                        }
                    }
                }
                for lane in (0..32).step_by(2) {
                    for p in 0..4 {
                        // xor-1 butterfly: even and odd lanes hold the two K halves of each row.
                        partial[(slot * s + split) * n + nt * 64 + lane_row(lane, p)] = acc[lane][p] + acc[lane + 1][p];
                    }
                }
            }
        }
    }
    let mut y = vec![0f32; slots * n];
    for slot in 0..slots {
        let e = if g.experts > 0 { ids[slot] as usize } else { 0 };
        for r in 0..n {
            let sum = (0..s).fold(0f32, |a, q| a + partial[(slot * s + q) * n + r]);
            let col = segment_col(&g.segments, r as u32) as usize;
            let gs = if g.gscale_stride > 0 { gscale[e * g.gscale_stride as usize + col] } else { gscale[col] };
            y[slot * n + r] = bf16(sum * gs);
        }
    }
    y
}

/// Worst per-element error, as `verify` computes it; a non-finite output is infinite error (a NaN
/// would otherwise vanish in `f64::max`, codex-review of #20).
fn worst(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let amax = want.iter().fold(0f64, |a, &b| a.max(b.abs()));
    got.iter().zip(want).map(|(&g, &r)| {
        let d = (g as f64 - r).abs() / (r.abs() + 1e-2 * amax);
        if d.is_finite() { d } else { f64::INFINITY }
    }).fold(0.0, f64::max)
}

#[test]
fn a_nan_output_row_fails_the_host_comparison() {
    assert_eq!(worst(&[f32::NAN, 2.0], &[1.0, 2.0]), f64::INFINITY);
    assert_eq!(worst(&[2.0, f32::INFINITY], &[1.0, 2.0]), f64::INFINITY);
    assert_eq!(worst(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
}

#[test]
fn a_one_expert_routed_down_projection_reads_the_down_scale() {
    // One stored expert and one slot is still the grouped layout: [1, 3] scales, column 2 for down.
    // No config reaches this today (a single expert parses as a dense MLP), so the linear is built
    // by hand at Flash Next's shapes.
    use sparkforge::{config::WeightFormat, plan::{plan_gemv_fp4, Linear}};
    let s = sparkforge::config::ModelSpec::from_json(&std::fs::read_to_string("configs/qwen3.8-flash-next-nvfp4.json").unwrap(), None).unwrap();
    let lin = |name: &str, n, k| Linear { name: name.into(), n, k, copies: 1, active: 1, format: WeightFormat::Nvfp4, per_token: 48 };
    let down = plan_gemv_fp4(&lin("expert_down", 2560, 640), &s, &sparkforge::target::GB10).fp4.unwrap();
    assert_eq!((down.experts, down.slots, down.gscale_stride, down.x_stride), (1, 1, 3, 640));
    assert_eq!(down.segments, vec![Segment { end: 2560, col: 2 }]);
    let gu = plan_gemv_fp4(&lin("expert_gate_up", 1280, 2560), &s, &sparkforge::target::GB10).fp4.unwrap();
    assert_eq!((gu.experts, gu.gscale_stride, gu.segments.clone()), (1, 3, vec![Segment { end: 640, col: 0 }, Segment { end: 1280, col: 1 }]));
    // Against a [1, 3] table with distinct gate, up and down scales, the emulation matches the
    // reference, and reading column 0 (the gate scale) instead would not.
    assert!(check(&down, 2560, 640, 1, &[0], 17) < 1.0 / 64.0);
    let mut gate = down.clone();
    gate.segments = vec![Segment { end: 2048, col: 0 }];
    let mut rng = Rng(19);
    let ops = Operands::random(&down, 2560, 640, 1, &mut rng);
    assert!(ops.gscale[0] != ops.gscale[2], "gate and down scales differ");
    let (w, sf) = ops.packed().unwrap();
    let x: Vec<f32> = (0..640).map(|i| bf16(((i * 37 % 64) as f32 - 32.0) / 32.0)).collect();
    let r = ops.reference(&down, &[0], &x.iter().map(|&v| v as f64).collect::<Vec<_>>()).unwrap();
    assert!(worst(&emulate(&gate, 2560, 640, &w, &sf, &ops.gscale, &[0], &x), &r) > 1.0 / 64.0);
}

#[test]
fn benchmark_expert_sets_never_reuse_an_expert_within_3x_l2() {
    use sparkforge::fp4::{bench_banks, bench_sets};
    let l2 = 24 << 20;
    // Reviewer cases: 8 experts of 1.99 MB with 2 slots (one table is 15.9 MB, under 3 x L2), and
    // 10 experts of 18.9 MB with 6 slots (a second batch would wrap into the first). Then Flash
    // Next, Qwen3-30B-A3B, and a table read whole by every launch.
    for (experts, slots, bytes, want) in [(8usize, 2usize, 1_990_656usize, 5usize), (10, 6, 18_874_368, 2), (512, 10, 1_843_200, 1),
        (128, 8, 1_769_472, 1), (3, 3, 4_000_000, 8)] {
        let (banks, sets) = bench_sets(experts, slots, bytes, l2, 300, &mut Rng(23));
        assert_eq!(banks, bench_banks(experts, slots, bytes, l2));
        assert_eq!(banks, want, "{experts} experts, {slots} slots");
        let mut last: std::collections::HashMap<(usize, u32), usize> = Default::default();
        for (i, (b, ids)) in sets.iter().enumerate() {
            let mut d = ids.clone();
            d.sort();
            d.dedup();
            assert_eq!(d.len(), slots, "launch {i}: distinct ids");
            assert!(ids.iter().all(|&e| (e as usize) < experts));
            for &e in ids {
                if let Some(j) = last.insert((*b, e), i) {
                    // Bytes read by the launches in between must cover 3 x L2.
                    assert!((i - j - 1) * slots * bytes >= 3 * l2, "{experts} experts: ({b}, {e}) reused after {} launches", i - j);
                }
            }
        }
    }
}

fn check(g: &Fp4Gemv, n: usize, k: usize, experts: usize, ids: &[u32], seed: u64) -> f64 {
    let mut rng = Rng(seed);
    let ops = Operands::random(g, n, k, experts, &mut rng);
    assert_eq!(&ops.scales[..2], &EDGE_SCALES[..2], "edge scales are present");
    let (w, sf) = ops.packed().unwrap();
    let xs = (g.slots as usize - 1) * g.x_stride as usize + k;
    let x: Vec<f32> = (0..xs).map(|_| bf16((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)).collect();
    let r = ops.reference(g, ids, &x.iter().map(|&v| v as f64).collect::<Vec<_>>()).unwrap();
    worst(&emulate(g, n, k, &w, &sf, &ops.gscale, ids, &x), &r)
}

#[test]
fn the_emulated_dense_gemv_matches_the_reference_with_and_without_split_k() {
    // K 80 is 5 K tiles: S 3 runs splits of 2, 2 and 1 tiles (a short last split).
    let segs = vec![Segment { end: 64, col: 0 }, Segment { end: 128, col: 1 }];
    for s in [1, 3] {
        let err = check(&gemv(s, 5, 1, 0, segs.clone(), 0, 0), 128, 80, 1, &[], 7);
        assert!(err < 1.0 / 64.0, "S {s}: worst error {err}");
    }
}

#[test]
fn the_emulated_grouped_gemv_reads_each_slots_expert_and_scales() {
    let ids = [4, 0, 2];
    // gate/up: one shared activation vector, gate and up global scales from columns 0 and 1.
    let gu = gemv(2, 4, 3, 5, vec![Segment { end: 64, col: 0 }, Segment { end: 128, col: 1 }], 3, 0);
    let err = check(&gu, 128, 64, 5, &ids, 11);
    assert!(err < 1.0 / 64.0, "gate/up worst error {err}");
    // down: one activation vector per slot, global scale column 2.
    let down = gemv(1, 4, 3, 5, vec![Segment { end: 128, col: 2 }], 3, 64);
    let err = check(&down, 128, 64, 5, &ids, 13);
    assert!(err < 1.0 / 64.0, "down worst error {err}");
}

#[test]
fn the_emulation_detects_a_wrong_segment_scale() {
    // Guards the check itself: swapping the gate and up columns must fail, since they differ 2x.
    let g = gemv(1, 4, 1, 0, vec![Segment { end: 64, col: 0 }, Segment { end: 128, col: 1 }], 0, 0);
    let mut swapped = g.clone();
    swapped.segments = vec![Segment { end: 64, col: 1 }, Segment { end: 128, col: 0 }];
    let mut rng = Rng(5);
    let ops = Operands::random(&g, 128, 64, 1, &mut rng);
    let (w, sf) = ops.packed().unwrap();
    let x: Vec<f32> = (0..64).map(|i| bf16(((i * 37 % 64) as f32 - 32.0) / 32.0)).collect();
    let r = ops.reference(&g, &[], &x.iter().map(|&v| v as f64).collect::<Vec<_>>()).unwrap();
    assert!(worst(&emulate(&swapped, 128, 64, &w, &sf, &ops.gscale, &[], &x), &r) > 1.0 / 64.0);
}

#[test]
fn reference_decoders_cover_the_edge_scales() {
    assert_eq!(sparkforge::fp4::e4m3(0x00), 0.0);
    assert_eq!(sparkforge::fp4::e4m3(0x01), 2f64.powi(-9));
    assert_eq!(sparkforge::fp4::e4m3(0x07), 7.0 * 2f64.powi(-9));
    for b in 0u8..0x7f {
        // The kernel's decode of the repacked byte equals the reference's decode of the original.
        assert_eq!(s0e5m3(e4m3_to_s0e5m3(b).unwrap()) as f64, sparkforge::fp4::e4m3(b), "{b:#04x}");
    }
    assert_eq!((0..16).map(e2m1).collect::<Vec<_>>(), [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]);
}

#[test]
fn a_bf16_max_activation_times_a_small_scale_stays_finite() {
    use sparkforge::fp4::{OVERFLOW_ROW, BF16_MAX};
    // Dense (S 2) and grouped down (2 slots, one x per slot): row 1 is code +6 x 2^-9 x 3.39e38.
    let segs = vec![Segment { end: 128, col: 0 }];
    let dense = gemv(2, 4, 1, 0, segs.clone(), 0, 0);
    let grouped = gemv(1, 4, 2, 3, vec![Segment { end: 128, col: 2 }], 3, 64);
    for (g, ids) in [(&dense, vec![]), (&grouped, vec![2u32, 0])] {
        let ops = Operands::random(g, 128, 64, g.experts.max(1) as usize, &mut Rng(31));
        let (big, x) = ops.overflow_case(g, g.slots as usize);
        let (w, sf) = big.packed().unwrap();
        let xf: Vec<f32> = x.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        assert_eq!(xf[0], f32::from_bits((BF16_MAX as u32) << 16));
        assert!(6.0 * xf[0] == f32::INFINITY, "the unscaled product overflows f32");
        let r = big.reference(g, &ids, &xf.iter().map(|&v| v as f64).collect::<Vec<_>>()).unwrap();
        let y = emulate(g, 128, 64, &w, &sf, &big.gscale, &ids, &xf);
        for slot in 0..g.slots as usize {
            let (got, want) = (y[slot * 128 + OVERFLOW_ROW] as f64, r[slot * 128 + OVERFLOW_ROW]);
            assert!(got.is_finite() && (got - want).abs() <= want.abs() / 64.0, "slot {slot}: {got} vs {want}");
        }
    }
}

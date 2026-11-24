//! NVFP4 repacking for the decode GEMV, checked against FlashInfer's own prep functions.
//!
//! `tests/golden/fp4_pack_flashinfer_db23734.json` holds bytes produced by
//! `_cute_dsl_pack_fp4_weight` and `_e4m3_to_s0e5m3` (flashinfer@db2373490172, torch 2.13 on the
//! CPU, run by `tools/golden/fp4_pack_flashinfer.py` in the `vllm/vllm-openai:v0.32.0` image on
//! spark-5) for random `[N, K/2]` weights and positive E4M3 scales, including zero and subnormals.
use sparkforge::repack::{e4m3_to_s0e5m3, pack_fp4, scales_s0e5m3, tile_byte, unpack_fp4, TILE_BYTES};

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

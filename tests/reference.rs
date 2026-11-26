//! The CPU reference decode step (`sparkforge::reference`) on a tiny Llama-shaped model: GQA head
//! mapping, KV-cache slots, RoPE at position 0, the dense-only guard and determinism. The GPU side
//! is compared against it by `verify` on a GB10.
use sparkforge::config::ModelSpec;
use sparkforge::reference::{self, bf2f, f2bf, f64_to_bf16, DenseWeights, KvCache, Linear, Norm, Rng, Weights};

/// Llama 3.1 8B's config shrunk to 2 layers of width 64: 4 query heads sharing 2 KV heads of 16.
fn tiny() -> ModelSpec {
    let mut m = ModelSpec::from_json(&std::fs::read_to_string("configs/llama-3.1-8b.json").unwrap(), None).unwrap();
    (m.hidden, m.heads, m.kv_heads, m.head_dim, m.intermediate, m.vocab, m.layers) = (64, 4, 2, 16, 96, 40, 2);
    m
}

#[test]
fn each_query_head_reads_its_own_kv_group() {
    let m = tiny();
    let (hd, kv) = (16, 32);
    let q: Vec<u16> = Rng(3).bf16(m.q_dim() as usize);
    // One cached position: softmax over a single score is 1, so each head returns that V row.
    let k = Rng(5).bf16(kv);
    let v: Vec<u16> = (0..kv).map(|i| f2bf(i as f32 * 0.25 - 3.0)).collect();
    let out = reference::attention(&m, &q, &k, &v, 1);
    for h in 0..4 {
        let g = h / 2;
        assert_eq!(&out[h * hd..(h + 1) * hd], &v[g * hd..(g + 1) * hd], "head {h} must read kv head {g}");
    }
}

#[test]
fn attention_weights_the_matching_key() {
    let m = tiny();
    let hd = 16;
    let q = vec![f2bf(1.0); m.q_dim() as usize];
    // Position 1's keys align with every query; position 0's oppose them, so position 1 dominates.
    let mut k = vec![f2bf(-4.0); 2 * 32];
    k[32..].fill(f2bf(4.0));
    let mut v = vec![f2bf(-1.0); 2 * 32];
    v[32..].fill(f2bf(2.0));
    let out = reference::attention(&m, &q, &k, &v, 2);
    for (i, &o) in out.iter().enumerate() {
        assert!((bf2f(o) - 2.0).abs() < 1e-3, "element {i} of {} is {}", hd * 4, bf2f(o));
    }
}

#[test]
fn rope_at_position_zero_leaves_q_and_k_unchanged() {
    let m = tiny();
    let before = Rng(9).bf16(m.qkv_dim() as usize);
    let mut after = before.clone();
    reference::rope(&m, &mut after, 0);
    assert_eq!(before, after);
}

#[test]
fn rmsnorm_rounds_the_normalized_input_before_the_weight() {
    // Rounding x * r to bf16 first, then multiplying by the weight, differs from rounding once at
    // the end for some of these elements; the test checks that it does.
    let n = 4096;
    let mut rng = Rng(17);
    let x = rng.bf16(n);
    let w: Vec<u16> = (0..n).map(|_| f2bf(1.0 + 0.1 * rng.unif())).collect();
    let y = reference::rmsnorm(&x, &w, 1e-5);
    let ss: f64 = x.iter().map(|&v| (bf2f(v) as f64).powi(2)).sum();
    let inv = 1.0 / (ss / n as f64 + 1e-5).sqrt();
    let mut differs = 0;
    for i in 0..n {
        let want = f2bf(bf2f(w[i]) * bf2f(f64_to_bf16(bf2f(x[i]) as f64 * inv)));
        assert_eq!(y[i], want, "element {i}");
        differs += (y[i] != f64_to_bf16(bf2f(w[i]) as f64 * bf2f(x[i]) as f64 * inv)) as usize;
    }
    assert!(differs > 0, "the inputs must tell one rounding from two");
}

#[test]
fn each_step_appends_its_kv_at_the_position() {
    let m = tiny();
    let mut rng = Rng(11);
    let w = DenseWeights::random(&m, 2, 2, &mut rng);
    let mut cache = KvCache::new(&m, 2, 3);
    let kv = m.kv_dim() as usize;
    let slot = |c: &Vec<u16>, i: usize| c[i * kv..(i + 1) * kv].to_vec();
    reference::decode_step(&m, &w, &mut cache, 0, 0).unwrap();
    let first: Vec<(Vec<u16>, Vec<u16>)> = (0..2).map(|l| (slot(&cache.k[l], 0), slot(&cache.v[l], 0))).collect();
    reference::decode_step(&m, &w, &mut cache, 1, 1).unwrap();
    for l in 0..2 {
        for (name, c, before) in [("K", &cache.k[l], &first[l].0), ("V", &cache.v[l], &first[l].1)] {
            assert!(before.iter().any(|&x| x != 0), "layer {l} {name}: step 0 writes slot 0");
            assert_eq!(&slot(c, 0), before, "layer {l} {name}: step 1 leaves slot 0 as step 0 wrote it");
            assert!(slot(c, 1).iter().any(|&x| x != 0), "layer {l} {name}: step 1 writes slot 1");
            assert_ne!(slot(c, 1), slot(c, 0), "layer {l} {name}: slot 1 holds a different token");
            assert!(slot(c, 2).iter().all(|&x| x == 0), "layer {l} {name}: slot 2 is untouched");
        }
    }
}

#[test]
fn f64_results_are_rounded_to_bf16_once() {
    // 1 + 2^-8 + 2^-25 is just above the midpoint between 1 and 1 + 2^-7, but f32 rounds it onto
    // the midpoint, which then ties down to 1. Rounded once it is 1 + 2^-7.
    let x = 1.0 + 2f64.powi(-8) + 2f64.powi(-25);
    assert_eq!(f2bf(x as f32), f2bf(1.0), "through f32 it double-rounds");
    assert_eq!(bf2f(f64_to_bf16(x)), 1.0078125);
    // A GEMV row with exactly those products gives the same.
    let w = [f2bf(1.0), f2bf(2f32.powi(-8)), f2bf(2f32.powi(-25))];
    let r = reference::gemv(&w, 1, &[f2bf(1.0); 3]);
    assert_eq!(bf2f(f64_to_bf16(r[0])), 1.0078125);
    // Exact midpoints tie to even, in both signs.
    assert_eq!(bf2f(f64_to_bf16(1.0 + 2f64.powi(-8))), 1.0);
    assert_eq!(bf2f(f64_to_bf16(1.0 + 3.0 * 2f64.powi(-8))), 1.015625);
    assert_eq!(bf2f(f64_to_bf16(-(1.0 + 3.0 * 2f64.powi(-8)))), -1.015625);
    // Values representable in bf16 are unchanged, and overflow goes to infinity.
    for v in [0.0, -0.0, 1.0, -3.5, 2f64.powi(-100), 3.0e38] {
        assert_eq!(f64_to_bf16(v), f2bf(v as f32), "{v}");
    }
    assert!(bf2f(f64_to_bf16(1e39)).is_infinite());
}

#[test]
fn malformed_weights_and_caches_are_refused_before_upload() {
    let m = tiny();
    let mut rng = Rng(19);
    let w = DenseWeights::random(&m, 2, 2, &mut rng);
    assert!(reference::check_shapes(&m, &w, &KvCache::new(&m, 2, 2)).is_ok());
    let kv = m.kv_dim() as usize;
    // Layer 0's K cache holds one token while the cache claims two.
    let mut short = KvCache::new(&m, 2, 2);
    short.k[0].truncate(kv);
    let e = reference::check_shapes(&m, &w, &short).unwrap_err().to_string();
    assert!(e.contains("layer 0 K cache"), "{e}");
    // The second V buffer is missing.
    let mut missing = KvCache::new(&m, 2, 2);
    missing.v.pop();
    let e = reference::check_shapes(&m, &w, &missing).unwrap_err().to_string();
    assert!(e.contains("1 V layers"), "{e}");
    // A weight of the wrong shape.
    let mut bad = DenseWeights::random(&m, 2, 2, &mut rng);
    bad.layers[1].down.pop();
    let e = reference::check_shapes(&m, &bad, &KvCache::new(&m, 2, 2)).unwrap_err().to_string();
    assert!(e.contains("layer 1 down"), "{e}");
}

#[test]
fn a_step_is_deterministic_and_finite() {
    let m = tiny();
    let run = || {
        let mut rng = Rng(13);
        let w = DenseWeights::random(&m, 2, 3, &mut rng);
        let mut cache = KvCache::new(&m, 2, 8);
        cache.fill_random(&m, 5, &mut rng);
        (0..3).map(|i| reference::decode_step(&m, &w, &mut cache, i, 5 + i).unwrap()).collect::<Vec<_>>()
    };
    let (a, b) = (run(), run());
    for (s, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.after_attn, y.after_attn, "step {s}");
        assert_eq!(x.after_mlp, y.after_mlp, "step {s}");
        assert_eq!(x.logits, y.logits, "step {s}");
        assert_eq!(x.after_attn.len(), 2);
        assert_eq!(x.logits.len(), 40);
        assert!(x.logits.iter().all(|v| v.is_finite()), "step {s} logits are finite");
        assert!(x.after_mlp[1].iter().all(|&v| bf2f(v).is_finite()), "step {s} hidden state is finite");
    }
}

#[test]
fn only_dense_bf16_models_are_covered() {
    assert!(reference::require_dense(&tiny()).is_ok());
    let moe = ModelSpec::from_json(&std::fs::read_to_string("configs/qwen3-30b-a3b.json").unwrap(), None).unwrap();
    let e = reference::require_dense(&moe).unwrap_err().to_string();
    assert!(e.contains("mixture-of-experts") || e.contains("q/k norm"), "{e}");
}

#[test]
fn subnormal_results_are_rounded_once() {
    let half = 2f64.powi(-134); // half the smallest bf16 subnormal, 2^-133
    for sign in [1.0, -1.0] {
        let neg = if sign < 0.0 { 0x8000 } else { 0 };
        // Just above the midpoint between 0 and 2^-133: f32 rounds it onto the midpoint, which then
        // ties to 0; rounded once it is the smallest subnormal.
        let x = sign * (half + 2f64.powi(-151));
        assert_eq!(f2bf(x as f32), neg, "through f32 it double-rounds");
        assert_eq!(f64_to_bf16(x), neg | 0x0001);
        assert_eq!(f64_to_bf16(sign * (half - 2f64.powi(-151))), neg, "just below the midpoint");
        assert_eq!(f64_to_bf16(sign * half), neg, "the midpoint ties to even (0)");
        assert_eq!(f64_to_bf16(sign * 3.0 * half), neg | 0x0002, "1.5 subnormals tie to even (2)");
        // The largest subnormal rounds up into the smallest normal, 0x0080.
        assert_eq!(f64_to_bf16(sign * (f32::MIN_POSITIVE as f64 - 2f64.powi(-140))), neg | 0x0080);
    }
}

#[test]
fn a_refused_step_leaves_the_cache_unchanged() {
    let m = tiny();
    let mut rng = Rng(23);
    let w = DenseWeights::random(&m, 2, 2, &mut rng);
    let mut cache = KvCache::new(&m, 2, 2);
    cache.v[1].clear();
    let before = (cache.k.clone(), cache.v.clone());
    let e = reference::decode_step(&m, &w, &mut cache, 0, 0).err().expect("a truncated V cache is refused").to_string();
    assert!(e.contains("layer 1 V cache"), "{e}");
    assert_eq!((cache.k.clone(), cache.v.clone()), before, "nothing was written");
    let mut ok = KvCache::new(&m, 2, 2);
    assert!(reference::decode_step(&m, &w, &mut ok, 2, 0).is_err(), "row 2 is past the 2 embedding rows");
    assert!(reference::decode_step(&m, &w, &mut ok, 0, 2).is_err(), "position 2 is past the capacity");
    assert!(ok.k.iter().chain(&ok.v).all(|c| c.iter().all(|&x| x == 0)));
}

/// `DenseWeights` whose GEMV for one linear fails, as a checkpoint read can partway through a step.
struct FailAt<'a>(&'a DenseWeights, &'static str);
impl Weights for FailAt<'_> {
    fn layers(&self) -> usize { self.0.layers() }
    fn embed_rows(&self) -> usize { self.0.embed_rows() }
    fn check(&self, m: &ModelSpec) -> anyhow::Result<()> { self.0.check(m) }
    fn embed_row(&self, row: usize) -> anyhow::Result<Vec<u16>> { self.0.embed_row(row) }
    fn norm(&self, n: Norm) -> anyhow::Result<Vec<u16>> { self.0.norm(n) }
    fn gemv(&self, l: Linear, x: &[u16]) -> anyhow::Result<Vec<f64>> {
        anyhow::ensure!(format!("{l:?}") != self.1, "injected failure at {l:?}");
        self.0.gemv(l, x)
    }
}

#[test]
fn a_step_that_fails_partway_restores_every_slot_it_wrote() {
    let m = tiny();
    let mut rng = Rng(31);
    let w = DenseWeights::random(&m, 2, 2, &mut rng);
    // Layer 1's o_proj runs after layers 0 and 1 have appended K and V; lm_head after every layer.
    for at in ["O(1)", "LmHead"] {
        let mut cache = KvCache::new(&m, 2, 3);
        cache.fill_random(&m, 3, &mut rng);
        let before = (cache.k.clone(), cache.v.clone());
        let e = reference::decode_step(&m, &FailAt(&w, at), &mut cache, 1, 2).err().expect("the injected failure").to_string();
        assert!(e.contains(at), "{e}");
        assert_eq!((cache.k.clone(), cache.v.clone()), before, "failure at {at}: every slot is restored");
    }
}

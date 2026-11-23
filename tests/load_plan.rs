//! The GPU load plan of a hybrid checkpoint, driven against host memory: which buffers exist, the
//! bytes of each, the expert repack checked against the checkpoint independently of the loader,
//! the read-back check, and PLE rows read from disk.
use anyhow::Result;
use sparkforge::config::ModelSpec;
use sparkforge::load_plan::{load, plan, samples_with, verify, verify_samples, Device, LoadPlan, PleMode};
use sparkforge::repack::{e4m3_to_s0e5m3, unpack_fp4};
use sparkforge::weights::{resolve_checkpoint_dir, spec_from_checkpoint, Checkpoint};

mod common;

/// GPU memory as host vectors.
struct Host;

impl Device for Host {
    type Buf = Vec<u8>;
    fn alloc(&mut self, bytes: usize) -> Result<Vec<u8>> {
        Ok(vec![0xee; bytes])
    }
    fn write(&mut self, buf: &mut Vec<u8>, off: usize, src: &[u8]) -> Result<()> {
        buf[off..off + src.len()].copy_from_slice(src);
        Ok(())
    }
    fn read(&mut self, buf: &Vec<u8>, off: usize, dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(&buf[off..off + dst.len()]);
        Ok(())
    }
}

fn tiny() -> (ModelSpec, Checkpoint) {
    let s = ModelSpec::from_json(&common::tiny_config().to_string(), None).unwrap();
    let ts = common::hybrid_tensors(&s, "layers.1.ple", common::TINY_TABLE_ROWS);
    let d = common::write_hybrid("load-plan", &common::tiny_config(), &ts, &common::TINY_VOCAB_SIZES);
    let dir = resolve_checkpoint_dir(&d).unwrap();
    (spec_from_checkpoint(&dir, None, None).unwrap(), Checkpoint::open_dir(&dir).unwrap())
}

fn buffer<'a>(p: &'a LoadPlan, name: &str) -> &'a sparkforge::load_plan::Buffer {
    p.buffers.iter().find(|b| b.name == name).unwrap_or_else(|| panic!("no buffer {name}"))
}

fn tensor(ck: &Checkpoint, name: &str) -> Vec<u8> {
    let mut v = vec![];
    ck.read_into(&ck.tensors[name], &mut v).unwrap();
    v
}

const L0: &str = "model.language_model.layers.0.mlp.experts";

#[test]
fn experts_are_grouped_per_layer_and_every_checked_byte_is_placed() {
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    // Resident: every checked byte is copied.
    assert_eq!((p.bytes, p.on_disk_bytes), (p.check.bytes, 0));
    // 4 experts of gate|up (64 x 64 FP4) and down (64 x 32 FP4), their scales, and the scalars.
    for (name, bytes) in [("gate_up.weight", 4 * 64 * 32), ("gate_up.scale", 4 * 64 * 4), ("down.weight", 4 * 64 * 16), ("down.scale", 4 * 64 * 2),
        ("weight_scale_2", 4 * 3 * 4), ("input_scale", 4 * 3 * 4)] {
        assert_eq!(buffer(&p, &format!("layers.0.mlp.experts.{name}")).bytes, bytes, "{name}");
    }
    // The MTP experts are weight-only NVFP4: no activation scales.
    assert!(p.buffers.iter().any(|b| b.name == "mtp.layers.0.mlp.experts.weight_scale_2"));
    assert!(!p.buffers.iter().any(|b| b.name == "mtp.layers.0.mlp.experts.input_scale"));
    // No per-expert tensor keeps a buffer of its own; other tensors do, by canonical name.
    assert!(!p.buffers.iter().any(|b| b.name.contains(".mlp.experts.0.")));
    assert_eq!(buffer(&p, "layers.0.mlp.gate.weight").bytes, 4 * 64 * 2);
    // 4 layers + 1 MTP layer, 6 or 5 expert buffers each.
    assert_eq!(p.buffers.iter().filter(|b| b.name.contains(".mlp.experts.")).count(), 4 * 6 + 5);
}

#[test]
fn disk_mode_leaves_only_the_table_shards_on_disk() {
    let (s, ck) = tiny();
    let (r, d) = (plan(&ck, &s, PleMode::Resident).unwrap(), plan(&ck, &s, PleMode::Disk).unwrap());
    // 4 shards x 15 rows x (8 value bytes + 1 scale byte).
    assert_eq!(d.on_disk_bytes, 4 * 15 * 9);
    assert_eq!(d.bytes + d.on_disk_bytes, r.bytes);
    assert!(!d.buffers.iter().any(|b| b.name.contains("ngram_embedding.shard_")));
    assert_eq!(r.buffers.iter().filter(|b| b.name.contains("ngram_embedding.shard_")).count(), 8);
    // The table's global scale and the size tensors stay resident.
    for name in ["layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2", "layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"] {
        buffer(&d, name);
    }
}

#[test]
fn a_loaded_expert_unpacks_to_the_checkpoints_gate_and_up_rows() {
    // Checked against the checkpoint directly, not through the loader's own fill code.
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    let bufs = load(&mut Host, &ck, &p, 1 << 20).unwrap();
    let at = |name: &str| &bufs[p.buffers.iter().position(|b| b.name == name).unwrap()];
    let x = 2;
    let (gate, up) = (tensor(&ck, &format!("{L0}.{x}.gate_proj.weight")), tensor(&ck, &format!("{L0}.{x}.up_proj.weight")));
    let block = &at("layers.0.mlp.experts.gate_up.weight")[x * 2048..(x + 1) * 2048];
    let mut rows = vec![0u8; 2048];
    unpack_fp4(block, 64, 64, &mut rows).unwrap();
    assert_eq!(&rows[..1024], &gate[..]);
    assert_eq!(&rows[1024..], &up[..]);
    // Scales: (64/16 = 4 blocks, 64 rows) per expert; row r < 32 is gate row r, else up row r - 32.
    let (gs, us) = (tensor(&ck, &format!("{L0}.{x}.gate_proj.weight_scale")), tensor(&ck, &format!("{L0}.{x}.up_proj.weight_scale")));
    let sc = &at("layers.0.mlp.experts.gate_up.scale")[x * 256..(x + 1) * 256];
    for c in 0..4 {
        for r in 0..64 {
            let src = if r < 32 { gs[r * 4 + c] } else { us[(r - 32) * 4 + c] };
            assert_eq!(sc[c * 64 + r], e4m3_to_s0e5m3(src).unwrap(), "block {c} row {r}");
        }
    }
    // Global scales: [expert, (gate, up, down)] as stored.
    let ws2 = at("layers.0.mlp.experts.weight_scale_2");
    for (j, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
        assert_eq!(&ws2[(x * 3 + j) * 4..][..4], &tensor(&ck, &format!("{L0}.{x}.{proj}.weight_scale_2"))[..]);
    }
    // A tensor copied as stored.
    assert_eq!(at("layers.0.mlp.gate.weight"), &tensor(&ck, "model.language_model.layers.0.mlp.gate.weight"));
}

#[test]
fn a_small_staging_buffer_copies_the_same_bytes() {
    // 4 KiB staging: the 16 KiB embedding goes in parts, the 2 KiB expert blocks one batch at a time.
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    assert_eq!(load(&mut Host, &ck, &p, 4096).unwrap(), load(&mut Host, &ck, &p, 1 << 20).unwrap());
    // A packed piece larger than the staging buffer is an error, not a partial copy.
    let err = format!("{:#}", load(&mut Host, &ck, &p, 1024).unwrap_err());
    assert!(err.contains("does not fit the 1024-byte staging buffer"), "{err}");
}

#[test]
fn read_back_verifies_every_buffer_and_names_a_mismatch() {
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    let mut bufs = load(&mut Host, &ck, &p, 1 << 20).unwrap();
    // Every buffer is under 64 MiB, so everything is read back.
    let v = verify(&mut Host, &ck, &p, &bufs).unwrap();
    assert_eq!((v.buffers, v.bytes), (p.buffers.len(), p.bytes));
    // One flipped byte in the last expert's down weights.
    let bi = p.buffers.iter().position(|b| b.name == "layers.3.mlp.experts.down.weight").unwrap();
    let last = bufs[bi].len() - 1;
    bufs[bi][last] ^= 0x10;
    let err = format!("{:#}", verify(&mut Host, &ck, &p, &bufs).unwrap_err());
    assert!(err.contains("layers.3.mlp.experts.down.weight piece 3 differs from the checkpoint"), "{err}");
}

#[test]
fn large_buffers_are_sampled_at_their_first_and_last_piece_and_edges() {
    // With 1 KiB thresholds: an expert buffer (8 KiB, 4 pieces) is read at experts 0 and 3, and the
    // 16 KiB embedding by its first and last KiB.
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    let sm = samples_with(&p, &ck, 1024, 1024);
    let of = |name: &str| {
        let bi = p.buffers.iter().position(|b| b.name == name).unwrap();
        sm.iter().filter(|x| x.0 == bi).map(|x| (x.1, x.2, x.3)).collect::<Vec<_>>()
    };
    assert_eq!(of("layers.0.mlp.experts.gate_up.weight"), vec![(0, 0, 2048), (3, 0, 2048)]);
    assert_eq!(of("embed_tokens.weight"), vec![(0, 0, 1024), (0, 15 * 1024, 1024)]);
    // A corrupted middle expert is outside these samples; the last one is caught.
    let mut bufs = load(&mut Host, &ck, &p, 1 << 20).unwrap();
    let bi = p.buffers.iter().position(|b| b.name == "layers.0.mlp.experts.gate_up.weight").unwrap();
    bufs[bi][2048 + 5] ^= 1;
    verify_samples(&mut Host, &ck, &p, &bufs, &sm).unwrap();
    bufs[bi][3 * 2048 + 5] ^= 1;
    assert!(verify_samples(&mut Host, &ck, &p, &bufs, &sm).is_err());
}

#[test]
fn ple_rows_are_gathered_from_their_shards() {
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Disk).unwrap();
    let t = &p.ple_table;
    assert_eq!((t.rows_per_shard, t.shards.len(), t.row_bytes.clone(), t.rows()), (15, 4, vec![8, 1], 60));
    let rows = [0u64, 14, 15, 59, 31];
    let mut out = vec![0u8; rows.len() * 9];
    t.gather(&ck, &rows, &mut out).unwrap();
    let table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    for (i, &r) in rows.iter().enumerate() {
        let (shard, row) = (r as usize / 15, r as usize % 15);
        let w = tensor(&ck, &format!("{table}.shard_{shard}.weight"));
        let sc = tensor(&ck, &format!("{table}.shard_{shard}.weight_scale"));
        assert_eq!(&out[i * 8..(i + 1) * 8], &w[row * 8..(row + 1) * 8], "row {r}");
        assert_eq!(out[rows.len() * 8 + i], sc[row], "row {r} scale");
    }
    let err = format!("{:#}", t.gather(&ck, &[60], &mut [0u8; 9]).unwrap_err());
    assert!(err.contains("PLE row 60 is outside the table's 60 rows"), "{err}");
}

#[test]
fn a_negative_expert_scale_fails_the_load_naming_the_buffer() {
    // The checkpoint's E4M3 scales must be finite and not negative to become S0E5M3.
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Disk).unwrap();
    let name = format!("{L0}.1.up_proj.weight_scale");
    let t = &ck.tensors[&name];
    let path = ck.shards[t.shard].path.clone();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[t.start as usize] = 0x80 | 0x38;
    let d = common::dir("load-plan-negative");
    std::fs::write(d.join("model.safetensors"), &bytes).unwrap();
    std::fs::copy(ck.dir.join("config.json"), d.join("config.json")).unwrap();
    let ck2 = Checkpoint::open(&d).unwrap();
    let err = format!("{:#}", load(&mut Host, &ck2, &p, 1 << 20).unwrap_err());
    assert!(err.contains("copying layers.0.mlp.experts.gate_up.scale") && err.contains("is negative"), "{err}");
}

#[test]
fn a_buffer_under_the_whole_threshold_is_read_back_whole() {
    // With a 1 MiB whole threshold and 1 KiB edges, the 16 KiB embedding is one stored tensor in a
    // small buffer: it is read whole, so a byte flipped in its middle fails verification.
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    let sm = samples_with(&p, &ck, 1 << 20, 1024);
    let bi = p.buffers.iter().position(|b| b.name == "embed_tokens.weight").unwrap();
    assert_eq!(sm.iter().filter(|x| x.0 == bi).map(|x| (x.1, x.2, x.3)).collect::<Vec<_>>(), vec![(0, 0, 16 * 1024)]);
    let mut bufs = load(&mut Host, &ck, &p, 1 << 20).unwrap();
    bufs[bi][8 * 1024] ^= 1;
    let err = format!("{:#}", verify_samples(&mut Host, &ck, &p, &bufs, &sm).unwrap_err());
    assert!(err.contains("embed_tokens.weight piece 0 differs from the checkpoint at byte 8192"), "{err}");
}

/// `Host`, counting the allocations and writes made through it.
#[derive(Default)]
struct Counting {
    allocs: usize,
    writes: usize,
}

impl Device for Counting {
    type Buf = Vec<u8>;
    fn alloc(&mut self, bytes: usize) -> Result<Vec<u8>> {
        self.allocs += 1;
        Host.alloc(bytes)
    }
    fn write(&mut self, buf: &mut Vec<u8>, off: usize, src: &[u8]) -> Result<()> {
        self.writes += 1;
        Host.write(buf, off, src)
    }
    fn read(&mut self, buf: &Vec<u8>, off: usize, dst: &mut [u8]) -> Result<()> {
        Host.read(buf, off, dst)
    }
}

#[test]
fn staging_is_sized_from_the_largest_packed_piece() {
    // An expert's packed gate|up block (64 x 64 FP4, 2 KiB) is produced whole, so a 1 KiB base
    // staging buffer grows to 2 KiB and then copies the same bytes; a base above it is kept.
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    assert_eq!(p.staging_bytes(&ck, 1024), 2048);
    assert_eq!(p.staging_bytes(&ck, 1 << 20), 1 << 20);
    assert_eq!(load(&mut Host, &ck, &p, 2048).unwrap(), load(&mut Host, &ck, &p, 1 << 20).unwrap());
}

#[test]
fn a_staging_buffer_too_small_or_empty_fails_before_any_allocation() {
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Resident).unwrap();
    for (staging, want) in [(0, "the staging buffer must hold at least one byte"), (1024, "a packed or scale piece of 2048 bytes does not fit the 1024-byte staging buffer")] {
        let mut dev = Counting::default();
        let err = format!("{:#}", load(&mut dev, &ck, &p, staging).unwrap_err());
        assert!(err.contains(want), "{err}");
        assert_eq!((dev.allocs, dev.writes), (0, 0), "staging {staging}");
    }
}

#[test]
fn timed_ple_rows_take_one_row_from_each_heads_own_range() {
    // The tiny table's heads hold 11, 13, 17 and 19 rows (rows 0..60 of a 60-row table).
    let (s, ck) = tiny();
    let p = plan(&ck, &s, PleMode::Disk).unwrap();
    let t = &p.ple_table;
    assert_eq!(t.heads, vec![(0, 11), (11, 13), (24, 17), (41, 19)]);
    for seed in [1u64, 0x5eed, 0xfeed] {
        let rows = t.sample_rows(500, seed).unwrap();
        assert_eq!(rows.len(), 500 * 4);
        for (i, r) in rows.iter().enumerate() {
            let (first, n) = t.heads[i % 4];
            assert!((first..first + n).contains(r), "seed {seed:#x}: token {} head {} drew row {r}", i / 4, i % 4);
        }
        // Over 500 tokens every row of the smallest head comes up.
        let mut seen: Vec<u64> = rows.iter().step_by(4).copied().collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, (0..11).collect::<Vec<_>>(), "seed {seed:#x}");
        let mut out = vec![0u8; t.gather_size(500).unwrap().1 as usize];
        t.gather(&ck, &rows, &mut out).unwrap();
    }
    assert_eq!(t.gather_size(3).unwrap(), (12, 12 * 9));
    let err = format!("{:#}", t.gather_size(0).unwrap_err());
    assert!(err.contains("at least one token"), "{err}");
    let err = format!("{:#}", t.gather_size(usize::MAX / 2).unwrap_err());
    assert!(err.contains("overflow"), "{err}");
}

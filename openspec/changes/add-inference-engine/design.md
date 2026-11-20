# Design

## Context

See proposal.md (Why). The numbers below come from the research recorded in the Linear plan document "Qwen3.8 Flash Next engine plan (2026-11-20)". There, [obs] marks a measurement or a source fact, and [inf] marks arithmetic from tensor sizes.

**Model** (`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, weight revision `6909a5be`, config `qwen4_exp`):
- Layers: 48 = 36 Gated DeltaNet (GDN, linear attention) + 12 sparse full attention (QSA). QSA reads at most 2,051 positions per query, chosen by a learned indexer.
- Width and attention: hidden 2560; 24 query heads and 2 KV heads at head_dim 256.
- HyperConnection (HC): a 4-stream residual mixed by 97 low-rank BF16 mixers per token.
- MoE: 512 experts, top-10, width 640, plus a shared expert.
- PLE: an n-gram embedding at layer 2, backed by a 28.8 GB NVFP4 table.
- One MTP layer; vocab 248,320.
- Precision: routed experts and the PLE table are NVFP4; attention, GDN and shared-expert projections are MXFP8; HC, routers, `lm_head` and embeddings are BF16. 106.29 GB in total [obs].

**Byte budget at batch 1** [inf]:
- 7.34 GB/token without speculation, which caps decode at 33 tok/s (245 GB/s).
- The largest shares: GDN projections 29%, routed experts 18%, HC 17.5%, BF16 `lm_head` 17%.

**The fleet engine** (vLLM fork on spark-5) [obs]:
- ~50 tok/s at c=1 with MTP k=4 and ~50% per-draft acceptance.
- 3,060 tok/s prefill; 16K time to first token 4.89 s.
- 69% of nominal bandwidth during decode.
- Already fuses: HC up + gated mean, q/k norm + mRoPE + gate, and MoE as one persistent cooperative kernel.

**sparkforge today** [obs]:
- Generated BF16 GEMV (245–251 GB/s on GB10), RMSNorm, RoPE + KV append, SiLU·mul, and split-K decode attention, each verified against an f64 reference.
- A dense BF16 weight loader that copies into device memory (PR swesty/sparkforge#4). Zero-copy ATS reads were measured 35% slower warm, so the loader copies.

## Goals / Non-Goals

**Goals:**
- One step program per model, built from the plan and the kernel manifest. Any change to the planner's kernels flows into the engine without hand wiring.
- Every milestone ends with a measured number on a GB10, against a stated ceiling.
- Speculation is exact: greedy output is unchanged by MTP, and sampling stays distribution-preserving.

**Non-Goals:**
- Dynamic shapes. Every step shape is planned, generated and captured ahead of time.
- Running beside the fleet engine at full size. Full-model runs need the node, under the stop protocol in decision 10.

## Decisions

1. **One process, one GPU thread.** The CLI (and later the HTTP front end) runs on the main thread. One dedicated thread owns the CUDA context, the graphs and all device memory, and requests are serialized at batch 1. *Alternative:* a separate engine process with IPC. Rejected: no benefit at batch 1, and it duplicates the context.

2. **Step program from plan + manifest.** `plan.rs` emits one entry per op instance class. Shapes repeat across layers of the same kind, so there are about 30 distinct kernels, not 900. The engine builds, per step shape, a dependency graph of (kernel, grid, block, smem, argument bindings) nodes that refer to named buffers in arenas. Mostly it is a chain, but independent work forks into parallel branches captured as parallel graph branches: the indexer side of sparse attention, HyperConnection side reductions, and the PLE gather (decision 13). A branch may start at the head of the step only if every input it reads exists at graph entry. For the PLE gather that is the step's first input token and the committed token history. In a speculative step, the gathers for draft and verify rows read token ids that the draft nodes produce later in the same replay, so those gathers take an explicit dependency on the draft node that writes each id and on the provisional history entry it consumes. *Alternative:* a hand-written forward pass per model. Rejected: it would drift from the planner and defeat model specialization.

3. **One CUDA Graph per step shape, with a device-side step header.**
   - Step shapes: decode M=1; one **speculative step** (k drafts at M=1, then verify at M=k+1, then accept); prefill chunk of C tokens.
   - The speculative step is a single graph. The accept kernel advances the step header on the device (committed length, next input tokens, RNG offset), so consecutive steps replay with no host round trip. The host keeps the next replay queued. Each replay writes its accepted count and output tokens into its own result slot, a ring of R slots indexed by the replay's sequence number, where R exceeds the number of replays that can be queued at once. The host copies a slot after that replay's completion event and only then releases it for reuse, so a delayed reader can never see a later replay's tokens in an earlier slot. A device-side conditional node can later choose the verify length (adaptive k) without a new graph per k.
   - Per-step scalars (position, sequence length, token ids, RNG offset) live in that device "step header". In a plain decode graph the sampling kernel advances it the same way. Only prefill and a new request write it from the host, with one memcpy node at the head of the graph, so graphs are never re-instantiated.
   - Every kernel uses programmatic dependent launch (PDL). Before `griddepcontrol.wait`, a kernel may load only data its predecessor doesn't produce, meaning weights at fixed addresses. Anything whose address or content depends on the predecessor comes after the wait: the routed expert ids and weights, activations, the step header and KV. In MoE, the grouped gate/up kernel can prefetch the shared expert's weights before the wait, but it must wait before reading the router's top-k and loading the routed experts. The PDL-versus-serialized check in tasks.md (group 3) enforces this rule.
   - *Alternative:* per-kernel launches. Rejected: about 900 launches at 2–4 µs each [obs launch-bound kernels in `verify --bench`] would cost 1–3 ms per ~30 ms step.

4. **Arenas sized by the plan; weights copied, experts repacked.**
   - Arenas: weights, KV, recurrent state (×(k+1) snapshots), activations, workspaces. All are allocated once, after the memory budget passes, and the budget reuses the weight-loading admission lock.
   - Weights are copied as in PR #4.
   - The 24,576 per-expert NVFP4 tensors are repacked at load time into one contiguous grouped buffer per layer, in the GEMV tile layout, so one launch covers all active experts.
   - *Alternative:* zero-copy from the file mapping. Rejected on PR #4's measurement.

5. **Prefill in fixed chunks with a valid-token count.** The prefill graph is captured at the planned chunk size (16K by default). The step header carries the number of real tokens in the chunk. Every kernel masks positions at or beyond that count: no KV, indexer, recurrent-state or token-history write and no output. The committed length advances by the real count. The final partial chunk of any prompt therefore runs through the same graph. *Alternative:* finish the tail with M=1 decode steps. Rejected: up to 16K serial steps.

6. **PLE table on disk first, resident later.**
   - Weights total 76.6 GB with the PLE table on disk and 105.4 GB with it resident [inf]. Disk mode gathers the needed rows into a pinned staging buffer before each graph replay.
   - MTP verify needs PLE rows for the draft tokens, which forces a host round trip between drafting and verifying, and that round trip is impossible inside the single speculative-step graph (decision 3). So the resident table is the end state, once the memory guard shows margin. Rows are then hashed on the device from the device token history (decision 8), with no host involvement.
   - Speculative decoding therefore requires the resident table. When the memory guard keeps the table on disk, the engine runs non-speculative decode for every request and reports that speculation is disabled and why, rather than failing or staging rows mid-graph.
   - Before SHA-772: check whether the PLE hash keys on raw token ids or on a remapped id table derived from the tokenizer (the reference engine's n-gram embedding uses such a map); the planner and loader need it either way.
   - *Alternative:* resident from day one. Deferred: about 10 GiB of margin on a node where running out of memory freezes it.

7. **Fusion groups.** Each one lands with its kernel change and is measured in launches and µs per step:
   - F1: HC combine + grouped RMSNorm in the HC down-GEMV prologue.
   - F2: HC scaled SiLU + up GEMV + gated mean.
   - F4: GDN input projections as one GEMV, with conv1d, gates and L2 norm in the epilogue.
   - F5: GDN recurrence + gated norm, one CTA per value head (48 heads on 48 SMs).
   - F6: QSA projections as one GEMV, with q/k norm, partial mRoPE and the FP8 KV append in the epilogue.
   - F7: indexer scoring + top-k, then sparse attention with an in-kernel merge.
   - F9: MoE in 3 launches: router + top-k; grouped gate/up + SiLU over 10 routed experts and the shared expert; grouped down + weighted combine.
   - F11: `lm_head` + sampling.
   - F13: graphs + PDL.

   Together they take a decode step from about 900 launches to 330–370 [inf].

   Rejected: attention + `o_proj` (F8). `o_proj` needs all 48 SMs streaming, and an attention CTA holds only its own heads.

   Deferred: router + top-k + experts in one kernel (F10), and a persistent megakernel (F14). F14 is estimated at ≤3–5% over F1–F13. The original cost model priced both with grid-wide barriers (about 450 per token at an unmeasured ~1–2 µs each, needing cooperative launch inside graphs). A cheaper mechanism exists: per-slot release/acquire counters, where producers do a release-add on a slot's counter and consumers spin on an acquire-load, with no grid barrier and no cooperative launch (decision 13). It requires every CTA of the grid to be co-resident, so the planner must cap such grids at 48 SMs × the kernel's occupancy, derived from `target.rs`, not taken from a datacenter grid size. That cap describes the kernel in isolation: if another context's work holds SM resources, a producer CTA may never be admitted while consumers spin, and the step stalls. Forward progress therefore requires the engine to own the GPU for as long as such a kernel runs, and a point-in-time check of other contexts cannot guarantee that, because a later context could still arrive. Counter-based kernels are therefore used only when the driver enforces exclusivity for the whole run: the GPU is in exclusive-process compute mode (set by the operator for full-model runs, decision 11), so no other context can be created while the engine holds the device. In any other mode the engine uses the non-counter path. If GB10 does not support exclusive-process mode, counter kernels are never used. Adoption requires the microbenchmark to pass at the largest planned grid in that mode, plus a check that a second context cannot be created during the run and that the engine selects the non-counter path when the mode is not set. F10 and F14 stay deferred until then.

8. **Speculative decoding.**
   - MTP drafts k tokens with the checkpoint's MTP layer, and one M=k+1 verify pass scores them.
   - Rejection sampling keeps the target distribution. Greedy is a special case, so output is identical with and without MTP.
   - Recurrent state is snapshotted after each verify position and rolled back to the last accepted one. That costs up to about 0.45 GB of state writes per step [inf], about 3% of a step.
   - **Committed KV length and verify visibility.** The step header holds the committed length C. A verify step writes the k+1 block positions at C..C+k. Its query at block offset i reads only positions below C+i+1 (its causal prefix, which includes earlier positions of the same block), in attention and in the indexer's scoring and selection. Each kernel derives that bound per row from C, so no per-row state is stored. The step header also carries the request's remaining generation budget G (requested tokens minus tokens already published). Acceptance of j drafts publishes n = min(j+1, L−C, G) tokens, cut further at the first stop token, and sets C to C+n and G to G−n (G becomes 0 after a stop token). Any replay, speculative or plain decode, that starts with C = L or G = 0 (for example one queued before its predecessor's final commit) publishes zero tokens, writes an accepted count of 0 and a "finished" flag (context full, budget exhausted or stop token) into its result slot, and leaves every committed value unchanged: C, G, KV, recurrent state and token history. The host stops queueing when it reads that flag. The limit and the budget are enforced on the device, so the host's read latency never lets output overshoot. Later steps read only below C, and they overwrite the rejected slots. Rolling back KV is one scalar write, not a copy.
   - **Room for provisional writes.** A circular buffer that verify appends to is sized to its live window plus k+1 slots: the raw-key ring, and the compressed indexer cache if it is circular. Its head advances only on commit, so a verify block never overwrites an entry that stays live after a rejection, even across the wrap point. Linear buffers (per-layer KV, MTP-layer KV, a linear indexer cache) are allocated for the context limit plus k+1 provisional slots, and the memory budget counts those slots. A verify block that starts near the limit therefore writes into reserved space, never past an allocation. Positions beyond the limit are never committed, so generation stops at the limit, and later tokens are not silently dropped.
   - **Shape-invariant target math.** Greedy exactness is guaranteed by construction rather than by tolerance. Every kernel that runs in both decode (M=1) and verify (M=k+1) computes each position with the same reduction partition and accumulation order: the same split-K slices and the same reduce order, with expert combines in a fixed expert-id order. A position's logits are then bitwise identical in both shapes, and near ties can't flip. Ties go to the lowest token id. The price is that verify kernels can't regroup the K dimension across positions for speed.
   - **One K-split per linear, chosen for M = k+1.** A single GEMV kernel can serve M = 1..k+1 rows with the same per-row MMA sequence, provided its K-split does not change with M. The planner therefore picks each linear's K-split once, as the smallest split whose staging fits GB10's 99 KB per block at M = k+1, and uses the same split at M = 1. This costs a little at M = 1 and keeps decode and verify bitwise identical. (The reference engine lets its split grow with M, which is why it promises only distributional equivalence.)
   - **Token history.** The PLE n-gram lookup reads token ids from a history buffer bounded the same way as KV. Within a verify block, query i sees the committed ids plus block ids up to i. Acceptance commits them with C, so a rolled-back draft never reaches a later lookup.
   - k is fixed first; adaptive k comes later.
   - *Alternative to snapshots:* recompute the rejected suffix. Rejected: it rereads every weight.
   - *Alternative to shape invariance:* accept tolerance-level differences and re-verify near-tie positions with an M=1 pass. Rejected: it adds a data-dependent extra pass to a captured graph, and it still needs a tie threshold to be chosen.

9. **Byte reductions are options, not defaults.**
   - MXFP8 HC weights save 0.64 GB/token; the fleet's experiment kept logits within noise.
   - A truncated-vocabulary draft head saves about 2.2 GB per MTP step.
   - Both are off by default. A target-numerics option (MXFP8 HC) must pass the precision gate on the reference set (decision 10). It must pass every correctness limit, and its mean and 99th-percentile absolute logprob difference may exceed the checkpoint-precision engine's own figures by at most 10%. Otherwise it reports FAIL and stays off. The baseline is the checkpoint-precision engine, not zero, because the reference engine itself differs from sparkforge by kernel rounding. A draft-only option (truncated head) can't change greedy output, so it reports acceptance instead.

10. **Correctness gates.**
   - **Per-layer golden diff:** activations dumped teacher-forced from a 4-layer side engine of the fleet stack, compared with the engine's per-layer output.
   - **Reference set:** checked into the repo as token ids: the 4 fleet A/B prompts plus 16 chat transcripts, 256 target tokens each.
   - **Noise floor, measured once:** two fleet-engine runs on the reference set that differ only in kernel rounding (the comparison behind the fleet's 96.5–97.1% top-1 band [obs]). It records:
     - per-layer activation error, from the side engine;
     - top-1 agreement;
     - mean and 99th-percentile absolute logprob difference of the reference tokens;
     - the difference in mean reference-token log-likelihood.

     A metric whose measured floor is zero falls back to the repo's verify tolerance (relative error 1/64).
   - **Checkpoint-precision gate:**
     - top-1 agreement of at least 96.5%, the band's lower edge. Agreement above the band passes.
     - every other metric at most 1.5 times its noise floor. The log-likelihood term catches errors that keep the argmax, such as a mis-scaled output head.

     Only a passing run's logprob figures become the baseline for the precision gate (decision 9).
   - **Llama first:** SHA-755 proves the runtime on Llama 3.1 8B BF16, which has a simple Hugging Face reference, before Flash Next.
   - *Alternative:* a full scalar Rust reference of GDN, QSA, HC and PLE. Rejected for now: costly, and part of the math is still unread in the fork's source.

11. **Shared-node protocol.** Full-model runs on spark-5 need the experimental Flash Next serving lane stopped.
   - The engine never stops another process; its memory budget refuses to start instead.
   - The run procedure is: ask the operator, stop the lane per the fleet runbook, run, then restart and health-check the lane. The operator confirms each stop.

12. **Milestones.** Each has the number it must record; tickets are in the Linear plan:

   | M | Deliverable | Recorded number |
   |---|---|---|
   | M0 | Flash Next config parsed and planned | 7.34 GB/token ±1%, ceiling 33 tok/s |
   | M1 | Mixed-format loader | load time and resident GiB under the guard |
   | M2 | Standalone decode kernels (`verify --bench`) | GB/s per kernel (≥ 225 for GEMVs ≥ 10 MB), error ≤ 1/64 |
   | M3 | Golden diff | per-layer error, final-layer top-1 agreement |
   | M4 | End-to-end decode, no MTP, graph per step | tok/s at 4K/32K/128K (≥ 28) |
   | M5 | Fusions + PDL | launches per step (≤ 370), tok/s delta |
   | M6 | MTP k=4 | tok/s on the A/B prompts vs ~50; acceptance; greedy identical |
   | M7 | Prefill (MXFP8 GEMM, W4A4 MoE, chunked GDN, QSA prefill, 16K chunks) | 16K time to first token vs 4.89 s |
   | M8 | Tokenizer + chat template + HTTP (separate `serving-api` change) | fleet smoke checks |
   | M9 | Byte-reduction options | tok/s gain with agreement and acceptance |

13. **Reference designs, and what transfers to GB10.** basetenlabs/deepseek-ultrafast (studied at commit `76766a9`) is a speculative-decoding engine for a model with the same building blocks: a 4-stream HyperConnection, sparse attention with a learned indexer, an n-gram hashed embedding, MoE with a shared expert. Its *ideas* inform decisions 2, 3, 6, 7 and 8. Its *code* is under the Elastic License 2.0 and is not copied into this Apache-2.0 repository; every borrowed idea is rebuilt from public papers or Apache-2.0 sources and re-derived by the planner.

   Its tuning does not transfer, because the hardware is different in every dimension that sets a shape:

   | | Reference target | GB10 (this engine) |
   |---|---|---|
   | GPU | datacenter Blackwell, `sm_100a`/`sm_103a` (B200, B300/GB300), 4–8 GPUs | one GB10, `sm_121a` |
   | SMs | about 148–160 per GPU [vendor, approximate] | **48** [obs, driver] |
   | Memory bandwidth | HBM3e, about 8 TB/s per GPU [vendor, approximate] | LPDDR5X shared with the CPU, 273 GB/s nominal, ~245 GB/s measured |
   | Shared memory | up to about 228 KB per SM [vendor, approximate] | **101,376 B per block** opt-in, 102,400 B per SM [obs, driver] |
   | L2 | far larger than GB10's [exact per-SKU size not verified here] | **24 MiB** (25,165,824 B) [obs, driver] |
   | Tensor cores | tcgen05 with TMEM, TMA tensor maps, clusters with DSMEM | `mma.sync` only; FP4 via `kind::mxf4nvf4` |

   Transfer rules:
   - **Grids are re-derived from 48 SMs.** A rule such as "split K until the grid reaches two waves" means about 96 CTAs here, not about 300. Co-residency caps (decision 7) use 48 × occupancy.
   - **Pipeline depth is re-derived from 99 KB.** A 4-stage weight ring sized for ~228 KB per SM generally does not fit; the planner chooses stages and tile sizes from `target.rs`.
   - **L2 reuse is not assumed.** With 24 MiB of L2, expert weights and most dense weights do not survive between uses, so cache hints and reuse arguments made for a much larger L2 are re-checked against GB10, not inherited.
   - **The compute-to-bandwidth ratio differs by roughly an order of magnitude**, so the M at which a verify GEMV stops being bandwidth-bound is measured on GB10 rather than taken from the reference.
   - **Datacenter-only features are excluded:** tcgen05/TMEM, TMA tensor maps, and cluster DSMEM reductions. GB10 reports cluster launch as supported, but DSMEM K-split is not planned until measured.
   - **No hand-tuned profiles.** Every number still comes from the config or `target.rs` (project rule).

   Probe on spark-5 (GB10, driver 580.178.04, NVRTC targeting `sm_121a`): a 1-D bulk async copy (`cp.async.bulk` with `mbarrier` completion) of 4 KB matches byte for byte, and `cvt.rn.f16x2.e2m1x2` decodes all 256 FP4 byte values correctly (low nibble to the low half). The reference's GEMV structure, a loader warp streaming pre-packed weights into a shared-memory ring while the other warps run `mma.sync`, is therefore buildable on GB10. Whether it beats the current CUDA-core GEMV on our expert shapes (640×2560, 2560×640) is measured in SHA-756, not assumed.

## Risks / Trade-offs

- **[The fleet engine already sits near its byte ceiling, so matching its algorithm gains only 10–20%]** → The plan relies on byte reductions (decision 9). A falsifying experiment is recorded in the plan: arm the route-dump hook during decode, and compare measured step time with predicted bytes ÷ 236 GB/s. If measured time is ≥ 1.4× predicted, kernel and launch work alone would pay more.
- **[Unread model math: router normalization, indexer compression and scoring, PLE apply, HC SiLU scale, MTP input combination]** → Each kernel ticket starts by reading the fork's source, and the golden diff (M3) catches any misreading before speed is measured.
- **[A resident PLE table leaves about 10 GiB of margin]** → Start with disk mode. Move to resident only after the guard and a soak run show margin.
- **[Full-model runs interrupt the experimental serving lane]** → Ask before each stop, use the runbook procedure, and health-check after restart.
- **[The checkpoint-precision engine misses a 1.5× noise-floor limit for a legitimate reason, such as W8A16 decode against the fleet's MXFP8 activations]** → First match the reference's activation quantization in the affected kernels. If the difference is intended, change the limit through a reviewed change to this spec, with the measured figures. Never widen it silently in code.
- **[Distinct experts per verify step are unknown (30–48 assumed)]** → Measure them in M6. The MTP ceiling moves by about 8% across that range.

## Open Questions

- The GB10 cost of grid barriers (cooperative launch inside CUDA graphs) and of per-slot release/acquire counters decides whether F10 and F14 are ever worth doing. A microbenchmark of both at the largest planned grid answers it after M5, under exclusive GPU use (decision 7). It does not change M0–M5.
- Whether cluster DSMEM reductions help any GB10 shape. Cluster launch is reported as supported; no plan depends on it until a probe measures it.
- Whether `minijinja` renders the checkpoint's chat template unchanged. This matters only for M8, and the `serving-api` change resolves it.

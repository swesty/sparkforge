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

2. **Step program from plan + manifest.** `plan.rs` emits one entry per op instance class. Shapes repeat across layers of the same kind, so there are about 30 distinct kernels, not 900. The engine builds, per step shape, an ordered list of (kernel, grid, block, smem, argument bindings) that refers to named buffers in arenas. *Alternative:* a hand-written forward pass per model. Rejected: it would drift from the planner and defeat model specialization.

3. **One CUDA Graph per step shape, with a device-side step header.**
   - Step shapes: decode M=1; draft M=1 (run k times); verify M=k+1; prefill chunk of C tokens.
   - Per-step scalars (position, sequence length, token ids, RNG offset) live in a device "step header", updated by one memcpy node at the head of each graph, so the graphs are never re-instantiated.
   - Every kernel uses programmatic dependent launch (PDL). Before `griddepcontrol.wait`, a kernel may load only data its predecessor doesn't produce, meaning weights at fixed addresses. Anything whose address or content depends on the predecessor comes after the wait: the routed expert ids and weights, activations, the step header and KV. In MoE, the grouped gate/up kernel can prefetch the shared expert's weights before the wait, but it must wait before reading the router's top-k and loading the routed experts. The PDL-versus-serialized check in tasks.md (group 3) enforces this rule.
   - *Alternative:* per-kernel launches. Rejected: about 900 launches at 2–4 µs each [obs launch-bound kernels in `verify --bench`] would cost 1–3 ms per ~30 ms step.

4. **Arenas sized by the plan; weights copied, experts repacked.**
   - Arenas: weights, KV, recurrent state (×(k+1) snapshots), activations, workspaces. All are allocated once, after the memory budget passes, and the budget reuses the weight-loading admission lock.
   - Weights are copied as in PR #4.
   - The 24,576 per-expert NVFP4 tensors are repacked at load time into one contiguous grouped buffer per layer, in the GEMV tile layout, so one launch covers all active experts.
   - *Alternative:* zero-copy from the file mapping. Rejected on PR #4's measurement.

5. **PLE table on disk first, resident later.**
   - Weights total 76.6 GB with the PLE table on disk and 105.4 GB with it resident [inf]. Disk mode gathers the needed rows into a pinned staging buffer before each graph replay.
   - MTP verify needs PLE rows for the draft tokens, which forces a host round trip between drafting and verifying. So the resident table is the end state, once the memory guard shows margin.
   - *Alternative:* resident from day one. Deferred: about 10 GiB of margin on a node where running out of memory freezes it.

6. **Fusion groups.** Each one lands with its kernel change and is measured in launches and µs per step:
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

   Deferred: router + top-k + experts in one kernel (F10, needs a grid barrier), and a persistent megakernel (F14). F14 is estimated at ≤3–5% over F1–F13. It costs about 450 grid barriers per token at an unmeasured ~1–2 µs each, and its behavior with cooperative launch inside graphs on GB10 is unknown.

7. **Speculative decoding.**
   - MTP drafts k tokens with the checkpoint's MTP layer, and one M=k+1 verify pass scores them.
   - Rejection sampling keeps the target distribution. Greedy is a special case, so output is identical with and without MTP.
   - Recurrent state is snapshotted after each verify position and rolled back to the last accepted one. That costs up to about 0.45 GB of state writes per step [inf], about 3% of a step.
   - **Committed KV length and verify visibility.** The step header holds the committed length C. A verify step writes the k+1 block positions at C..C+k. Its query at block offset i reads only positions below C+i+1 (its causal prefix, which includes earlier positions of the same block), in attention and in the indexer's scoring and selection. Each kernel derives that bound per row from C, so no per-row state is stored. Acceptance of j drafts sets C to C+j+1. Later steps read only below C, and they overwrite the rejected slots. Rolling back KV is one scalar write, not a copy.
   - **Room for provisional writes.** A circular buffer that verify appends to is sized to its live window plus k+1 slots: the raw-key ring, and the compressed indexer cache if it is circular. Its head advances only on commit, so a verify block never overwrites an entry that stays live after a rejection, even across the wrap point. Linear per-layer KV has no wrap, so the constraint doesn't apply to it.
   - **Shape-invariant target math.** Greedy exactness is guaranteed by construction rather than by tolerance. Every kernel that runs in both decode (M=1) and verify (M=k+1) computes each position with the same reduction partition and accumulation order: the same split-K slices and the same reduce order, with expert combines in a fixed expert-id order. A position's logits are then bitwise identical in both shapes, and near ties can't flip. Ties go to the lowest token id. The price is that verify kernels can't regroup the K dimension across positions for speed.
   - k is fixed first; adaptive k comes later.
   - *Alternative to snapshots:* recompute the rejected suffix. Rejected: it rereads every weight.
   - *Alternative to shape invariance:* accept tolerance-level differences and re-verify near-tie positions with an M=1 pass. Rejected: it adds a data-dependent extra pass to a captured graph, and it still needs a tie threshold to be chosen.

8. **Byte reductions are options, not defaults.**
   - MXFP8 HC weights save 0.64 GB/token; the fleet's experiment kept logits within noise.
   - A truncated-vocabulary draft head saves about 2.2 GB per MTP step.
   - Both are off by default. A target-numerics option (MXFP8 HC) must pass the precision gate on the reference set (decision 9). It must pass every correctness limit, and its mean and 99th-percentile absolute logprob difference may exceed the checkpoint-precision engine's own figures by at most 10%. Otherwise it reports FAIL and stays off. The baseline is the checkpoint-precision engine, not zero, because the reference engine itself differs from sparkforge by kernel rounding. A draft-only option (truncated head) can't change greedy output, so it reports acceptance instead.

9. **Correctness gates.**
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

     Only a passing run's logprob figures become the baseline for the precision gate (decision 8).
   - **Llama first:** SHA-755 proves the runtime on Llama 3.1 8B BF16, which has a simple Hugging Face reference, before Flash Next.
   - *Alternative:* a full scalar Rust reference of GDN, QSA, HC and PLE. Rejected for now: costly, and part of the math is still unread in the fork's source.

10. **Shared-node protocol.** Full-model runs on spark-5 need the experimental Flash Next serving lane stopped.
   - The engine never stops another process; its memory budget refuses to start instead.
   - The run procedure is: ask the operator, stop the lane per the fleet runbook, run, then restart and health-check the lane. The operator confirms each stop.

11. **Milestones.** Each has the number it must record; tickets are in the Linear plan:

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

## Risks / Trade-offs

- **[The fleet engine already sits near its byte ceiling, so matching its algorithm gains only 10–20%]** → The plan relies on byte reductions (decision 8). A falsifying experiment is recorded in the plan: arm the route-dump hook during decode, and compare measured step time with predicted bytes ÷ 236 GB/s. If measured time is ≥ 1.4× predicted, kernel and launch work alone would pay more.
- **[Unread model math: router normalization, indexer compression and scoring, PLE apply, HC SiLU scale, MTP input combination]** → Each kernel ticket starts by reading the fork's source, and the golden diff (M3) catches any misreading before speed is measured.
- **[A resident PLE table leaves about 10 GiB of margin]** → Start with disk mode. Move to resident only after the guard and a soak run show margin.
- **[Full-model runs interrupt the experimental serving lane]** → Ask before each stop, use the runbook procedure, and health-check after restart.
- **[The checkpoint-precision engine misses a 1.5× noise-floor limit for a legitimate reason, such as W8A16 decode against the fleet's MXFP8 activations]** → First match the reference's activation quantization in the affected kernels. If the difference is intended, change the limit through a reviewed change to this spec, with the measured figures. Never widen it silently in code.
- **[Distinct experts per verify step are unknown (30–48 assumed)]** → Measure them in M6. The MTP ceiling moves by about 8% across that range.

## Open Questions

- The GB10 grid-barrier latency, and cooperative launch inside CUDA graphs, decide whether F10 and F14 are ever worth doing. A 48-CTA microbenchmark answers it after M5. It does not change M0–M5.
- Whether `minijinja` renders the checkpoint's chat template unchanged. This matters only for M8, and the `serving-api` change resolves it.

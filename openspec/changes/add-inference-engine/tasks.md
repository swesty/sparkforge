# Tasks

This change covers the engine itself. Linear plan: "Qwen3.8 Flash Next engine plan (2026-11-20)", with milestones M0–M9. These dependencies land as their own changes, at the milestone in design.md decision 12:
- the Flash Next config (M0, SHA-766);
- the mixed-format loader (M1, SHA-767, after the loader hardening in SHA-765);
- the decode kernel families (M2): SHA-756, SHA-768, SHA-769, SHA-770, SHA-771, SHA-772, SHA-758, SHA-747, SHA-749;
- the prefill kernels (M7): SHA-776, SHA-751, SHA-759, SHA-777, SHA-778, SHA-757;
- chat template, reasoning and tool parsers, and the serving API (M8): SHA-779, SHA-780.

Plain-text tokenization lives in this change (task 1.4), because `generate` needs it from the first milestone. Each new or changed kernel in those changes carries its own `sparkforge verify --bench` task on a GB10.

## 1. Runtime proving ground on Llama 3.1 8B (SHA-755)

- [ ] 1.1 Step program built from plan + manifest for a dense model, with arena allocation behind the memory budget. Verify: `cargo test` builds the program for both example configs, and its kernel list matches the manifest.
- [ ] 1.2 One CUDA Graph per decode step, with a device step header that the sampling kernel advances on the device (a host memcpy node only for prefill and a new request). Verify: on a GB10, 256 decode steps of Llama 3.1 8B issue one graph launch each, and the text matches non-graph execution.
- [ ] 1.3 Greedy and temperature sampling (Philox) in the `lm_head` epilogue, with ties broken toward the lowest token id. Verify: greedy output matches an argmax over full logits for 256 steps, including a constructed exact tie; `verify --bench` on the fused kernel on a GB10.
- [ ] 1.4 Plain-text prompt encoding and output decoding from the checkpoint's `tokenizer.json` (HF `tokenizers` crate), with no chat template. Verify: `cargo test` round-trips text and matches the Hugging Face tokenizer's ids for 20 fixed strings (ids checked in as fixtures).
- [ ] 1.5 `sparkforge generate --prompt "<text>"` prints decoded text, decode tok/s, time to first token and the ceiling. Verify: Llama 3.1 8B on a GB10 with a plain-text prompt, with the tok/s recorded in the PR against the planner ceiling (12.2 tok/s at 32K, 15.1 at 4K).

## 2. Flash Next engine, batch 1, no speculation (M3–M4: SHA-773, SHA-774)

- [ ] 2.1 Step program for the Flash Next layer schedule (36 GDN + 12 QSA, HC, MoE, PLE at layer 2). Verify: `cargo test` checks the step program's per-step bytes equal the planner's 7.34 GB ±1%.
- [ ] 2.2 Engine memory budget covering weights, KV, GDN state, snapshots, activations and workspaces. Verify: `cargo test` covers the refusal arithmetic; on spark-5 with the fleet lane running, the engine refuses with the budget message and leaves the lane untouched.
- [ ] 2.3 Measure the reference noise floor (design decision 10): two fleet-engine runs on the reference set that differ only in kernel rounding. Verify: per-layer error, top-1 agreement, mean and p99 absolute logprob difference, and the mean log-likelihood difference recorded in the PR and checked in as the gate's limits.
- [ ] 2.4 Per-layer golden diff and teacher-forced check against the fleet engine, with PASS/FAIL against those limits. Verify: top-1 agreement of at least 96.5%, and every other metric at most 1.5 times its floor. `cargo test` covers the gate arithmetic: it passes 97.5% agreement, fails 96.4%, and fails a run whose logits are doubled with top-1 unchanged.
- [ ] 2.5 End-to-end greedy decode with a graph per step. Verify: tok/s at 4K, 32K and 128K context on a GB10 (spark-5 lane stopped with operator approval), at least 28 tok/s against the 33 tok/s ceiling.

## 3. Fusions, PDL and launch count (M5: SHA-750)

- [ ] 3.1 `generate --stats` reports, per step: graph submissions, individual host kernel launches, and executed kernel nodes (from the graph's node count), next to the step program's kernel count. Verify: for a decode graph with N kernel nodes, one submission, zero individual launches, and N executed nodes equal to the step program, on a GB10.
- [ ] 3.2 PDL in every kernel, with pre-wait loads limited to producer-independent weights (design decision 3). Verify: a decode sequence whose expert routes change every token gives bitwise-identical logits with PDL on and with serialized launches, on a GB10.
- [ ] 3.3 Adopt the fused kernels (F1, F2, F4–F7, F9, F11) in the step program as each kernel change lands. Verify: launches per step at most 370; tok/s before and after recorded on a GB10; `verify --bench` for each fused kernel.

## 4. MTP speculative decoding (M6: SHA-753, SHA-775)

- [ ] 4.1 One speculative-step graph (k drafts at M=1, verify at M=k+1, accept), with rejection sampling, and the accept kernel advancing the step header on the device. Verify: `cargo test` checks the acceptance logic against a reference implementation on fixed probabilities; on a GB10, 256 steps replay with no host write to the step header.
- [ ] 4.2 Shape-invariant target computation: every kernel used by verify computes each position with the same reduction partition and order as the M=1 decode kernel (design decision 8). Verify: over 512 positions, verify-step logits are bitwise identical to decode-step logits, including a constructed near-tie position, on a GB10.
- [ ] 4.3 Snapshot recurrent and convolution state per verify position, and roll back to the last accepted one. Verify: after partial acceptance, state matches a non-speculative run over the same tokens within kernel tolerance, on a GB10.
- [ ] 4.4 Committed KV length in the step header, and per-query causal bounds in verify (query i reads below C+i+1, in attention and the indexer). Verify on a GB10:
  - a verify block where the indexer selects an earlier in-block key gives logits bitwise identical to sequential decode;
  - later keys in the block are never selected;
  - after forced 2-of-4 acceptance, the next step's selected positions and logits equal non-speculative decoding.
- [ ] 4.5 Circular buffers (the raw-key ring, and the compressed indexer cache if circular) sized to the live window plus k+1, with the head advanced only on commit. Verify: a full ring, a 5-position verify block across the wrap, and 2 accepted positions give next-step keys, selected positions and logits equal to sequential decode, on a GB10.
- [ ] 4.6 Context-limit headroom: every buffer verify writes (linear KV, MTP-layer KV, indexer caches, token history) holds k+1 provisional slots beyond the limit, counted in the memory budget. Verify: k=4 with 1, 2, 3 and 4 positions left before the limit keeps every write in bounds (checked by a bounds-checking debug build), commits nothing past the limit, and matches sequential decode, on a GB10.
- [ ] 4.7 PLE token history follows the committed prefix and the causal part of the block. Verify: forced 2-of-4 acceptance where the accepted and rejected histories map to different n-gram rows gives the accepted row and logits equal to sequential decode, on a GB10.
- [ ] 4.8 Greedy output identical with MTP off and with MTP k=4. Verify: the 4 A/B prompts at 256 tokens produce identical token ids.
- [ ] 4.9 Measure MTP decode. Verify: tok/s and per-draft acceptance on the A/B prompts against the fleet's ~50 tok/s on the same GB10, plus the distinct experts per verify step.

## 5. Prefill chunk graph (M7)

- [ ] 5.1 Prefill chunk step graphs (chunk size from the plan, 16K by default) built from the M7 prefill kernels, with a valid-token count in the step header that masks padded positions, and the KV, indexer, recurrent and token-history state handed to decode. Verify on a GB10 for prompts of 1, 16,384 and 16,385 tokens: committed length equals the prompt length, logits at the last prompt position equal token-by-token ingestion within kernel tolerance, and the first generated tokens match.
- [ ] 5.2 Measure prefill. Verify: `generate` on a 16K-token plain-text prompt reports time to first token below 4.89 s and prefill tok/s against the fleet's 3,060, on a GB10.

## 6. Opt-in byte reductions (M9: SHA-781)

- [ ] 6.1 Precision gate: a run on the reference set (design decision 10) passes only if it meets every limit of task 2.4 and its mean and 99th-percentile absolute logprob difference exceed the checkpoint-precision baseline from task 2.4 by at most 10%. Otherwise the option reports FAIL and stays off. Verify: `cargo test` rejects a synthetic case with unchanged top-1 and out-of-tolerance logprobs.
- [ ] 6.2 MXFP8 HyperConnection weight option, off by default and subject to the gate. Verify: tok/s and the gate result with and without it, on a GB10.
- [ ] 6.3 Truncated-vocabulary draft head option, off by default. Verify: tok/s and acceptance with and without it; greedy output still identical with MTP on and off.

## Workflow follow-up

- Archive this change with `openspec archive add-inference-engine` only when every task above is done, including the prefill chunk graph (group 5) and the 16K time-to-first-token check.
- The `serving-api` change (chat template, reasoning and tool parsers, OpenAI endpoints) is proposed separately at M8.

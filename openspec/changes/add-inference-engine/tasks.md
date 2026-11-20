# Tasks

This change covers the engine itself. Linear plan: "Qwen3.8 Flash Next engine plan (2026-11-20)", with milestones M0–M9. These dependencies land as their own changes, at the milestone in design.md decision 11:
- the Flash Next config (M0, SHA-766);
- the mixed-format loader (M1, SHA-767, after the loader hardening in SHA-765);
- the decode kernel families (M2): SHA-756, SHA-768, SHA-769, SHA-770, SHA-771, SHA-772, SHA-758, SHA-747, SHA-749;
- the prefill kernels (M7): SHA-776, SHA-751, SHA-759, SHA-777, SHA-778, SHA-757;
- tokenizer, template and serving API (M8): SHA-779, SHA-780.

Each new or changed kernel in those changes carries its own `sparkforge verify --bench` task on a GB10.

## 1. Runtime proving ground on Llama 3.1 8B (SHA-755)

- [ ] 1.1 Step program built from plan + manifest for a dense model, with arena allocation behind the memory budget. Verify: `cargo test` builds the program for both example configs, and its kernel list matches the manifest.
- [ ] 1.2 One CUDA Graph per decode step, with a device step header updated by a memcpy node. Verify: on a GB10, 256 decode steps of Llama 3.1 8B issue one graph launch each, and the text matches non-graph execution.
- [ ] 1.3 Greedy and temperature sampling (Philox) in the `lm_head` epilogue. Verify: greedy output matches an argmax over full logits for 256 steps; `verify --bench` on the fused kernel on a GB10.
- [ ] 1.4 `sparkforge generate` prints text, decode tok/s, time to first token and the ceiling. Verify: Llama 3.1 8B on a GB10, with the tok/s recorded in the PR against the planner ceiling (12.2 tok/s at 32K, 15.1 at 4K).

## 2. Flash Next engine, batch 1, no speculation (M3–M4: SHA-773, SHA-774)

- [ ] 2.1 Step program for the Flash Next layer schedule (36 GDN + 12 QSA, HC, MoE, PLE at layer 2). Verify: `cargo test` checks the step program's per-step bytes equal the planner's 7.34 GB ±1%.
- [ ] 2.2 Engine memory budget covering weights, KV, GDN state, snapshots, activations and workspaces. Verify: `cargo test` covers the refusal arithmetic; on spark-5 with the fleet lane running, the engine refuses with the budget message and leaves the lane untouched.
- [ ] 2.3 Per-layer golden diff against activations dumped teacher-forced from the fleet stack. Verify: per-layer max and mean error, and final-layer top-1 agreement, recorded in the PR.
- [ ] 2.4 Teacher-forced top-1 agreement check against the fleet engine. Verify: agreement within 96.5–97.1% on the reference prompts.
- [ ] 2.5 End-to-end greedy decode with a graph per step. Verify: tok/s at 4K, 32K and 128K context on a GB10 (spark-5 lane stopped with operator approval), at least 28 tok/s against the 33 tok/s ceiling.

## 3. Fusions and launch count (M5: SHA-750)

- [ ] 3.1 Report kernels per step from the step program, and launches per step from the runtime. Verify: `generate --stats` prints both, and they match.
- [ ] 3.2 Adopt the fused kernels (F1, F2, F4–F7, F9, F11) and PDL in the step program as each kernel change lands. Verify: launches per step at most 370; tok/s before and after recorded on a GB10; `verify --bench` for each fused kernel.

## 4. MTP speculative decoding (M6: SHA-753, SHA-775)

- [ ] 4.1 Draft (M=1, k times) and verify (M=k+1) step graphs, with rejection sampling. Verify: `cargo test` checks the acceptance logic against a reference implementation on fixed probabilities.
- [ ] 4.2 Snapshot recurrent and convolution state per verify position, and roll back to the last accepted one. Verify: after partial acceptance, state matches a non-speculative run over the same tokens within kernel tolerance, on a GB10.
- [ ] 4.3 Greedy output identical with MTP off and with MTP k=4. Verify: the 4 A/B prompts at 256 tokens produce identical token ids.
- [ ] 4.4 Measure MTP decode. Verify: tok/s and per-draft acceptance on the A/B prompts against the fleet's ~50 tok/s on the same GB10, plus the distinct experts per verify step.

## 5. Opt-in byte reductions (M9: SHA-781)

- [ ] 5.1 MXFP8 HyperConnection weight option, off by default. Verify: tok/s and top-1 agreement with and without it, on a GB10.
- [ ] 5.2 Truncated-vocabulary draft head option, off by default. Verify: tok/s and acceptance with and without it; greedy output still identical with MTP on and off.

## Workflow follow-up

- Archive this change with `openspec archive add-inference-engine` once tasks 1–4 are done. Task 5 and later milestones can move to a follow-up change.
- The `serving-api` change (tokenizer, chat template, OpenAI endpoints) is proposed separately at M8.

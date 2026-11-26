# Tasks

- [x] 1.1 Planner: `prefill_attn` tile search (shared memory with Q reuse, threads, estimated registers; resident warps, then K/V reuse, then stages)
- [x] 1.2 Codegen: `attn_prefill` (Q in registers, `cp.async` K/V pipeline, `mma.sync` Q·Kᵀ and P·V, exp2 online softmax, causal mask, start position); row-wise `rope_kv_append`, `residual_add`, `silu_mul`, `embed_gather`
- [x] 1.3 Runtime: `enable_prefill`, `prefill` in chunks with the prefill GEMMs and `attn_prefill`, then `lm_head` and `argmax_advance` for the graph handoff; `prefill_bytes` in admission
- [x] 1.4 `generate`: prefill the prompt, time to first token, `--bench-prefill`, `--check` from the prefill
- [x] 1.5 `verify`: attention vs f64 at 1/128/planned/2048/8192 and from position 1000, bitwise causality, prefill vs decode (KV cache, logits, argmax, graph handoff, chunking), TFLOP/s; attention in the prefill estimate
- [x] 1.6 CI tests: prefill attention tiles within GB10 limits and generated, none for hybrid plans, the estimate's attention row; goldens regenerated
- [x] 1.7 GB10 (spark-5): both models' `verify --bench` pass; tile sweep; mutation checks (a future key attended, a skipped tile, no O rescale, RoPE without the row) each fail `verify`; Llama 3.1 8B `generate` with `--check 4` and `--bench-prefill 128,2048,8192`
- [x] 1.8 `openspec validate add-dense-prefill-attention --strict`, archive, `verify.sh`

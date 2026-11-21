# Tasks

## 1. Blob verification (cycle 2 follow-ups 1 and 2)
- [x] 1.1 `verify_blob` (SHA256 or Git SHA1) for shards in parallel, carrying the verified stamp; metadata over the bytes returned
- [x] 1.2 Remove the pair check, `ResolveStep` and `resolve_checkpoint_dir_between`
- [x] 1.3 `sha2` with `asm`; measured on spark-5
- [x] 1.4 Tests: a mismatched blob is refused and the right bytes open; a Git-object `config.json` is verified; existing snapshot tests use real hashes

## 2. Repair (follow-ups 3 and 4)
- [x] 2.1 Closed-lock refusal message for the node lock; ACL error and runbook give close, drain, replace
- [x] 2.2 Test: a node lock the account can't open, with no group read, says a repair is under way

## 3. Test timing (follow-up 5)
- [x] 3.1 `admit_observed`; the replaced-lock test waits for B's signal; fails 3 of 3 with the re-check disabled, passes 3 of 3 with it

## 4. Verification
- [x] 4.1 `verify.sh` on macOS; tests, wasm check, CUDA build on spark-5
- [x] 4.2 `verify-weights` on Llama 3.1 8B from plain files and from a snapshot with real SHA256 blob names, timed

# Tasks

- [x] 1.1 The deadline test starts its clock at the first pause; 5 of 5 passes with the late-acquisition check, 5 of 5 failures with it disabled; with a 30 ms delay injected before `admit_with`, the old test fails and the new one passes
- [x] 1.2 `admission_lock_in` docs and spec: every sparkforge process on a node must see the same `~/.local/state/sparkforge`
- [x] 1.3 Spec: the upgrade pauses dispatch, drains or cancels pending and running old-binary invocations, switches launchers to the new binary, and leaves no old binary runnable before loads resume
- [x] 1.4 `verify.sh` on macOS

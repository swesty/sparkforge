# Tasks

- [x] 1.1 `admit` docs and spec: deleting the lock file or state directory while admitted is unsupported
- [x] 1.2 Spec: upgrades from any earlier version drain running loads first
- [x] 1.3 `AdmitHooks { on_wait, sleep }` and `admit_with`; the deadline test releases inside its own pause; 5 of 5 failures with the check disabled, 5 of 5 passes with it
- [x] 1.4 `verify.sh` on macOS; tests on spark-5

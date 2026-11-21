# Tasks

- [x] 1.1 Re-check the lock's path after `flock`; re-lock the new file within the deadline. Test: a waiter on a replaced file waits on the new file's holder (with `admit_observed`); it fails 3 of 3 runs with the re-check disabled
- [x] 1.2 Time out an acquisition at or after the deadline once the lock was found held. Test: a holder released at 14 ms with a 10 ms wait gives a timeout; the previous code admitted it in 3 of 3 runs
- [x] 1.3 Spec: drain rule for upgrades from the loader before #13, and the two scenarios above
- [x] 1.4 `verify.sh` on macOS; tests on spark-5

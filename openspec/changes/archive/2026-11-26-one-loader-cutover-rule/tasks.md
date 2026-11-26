# Tasks

- [x] 1.1 New requirement "Loader version switches follow one cutover rule" for upgrades and rollbacks: disable every way to start the outgoing version, then make the incoming one runnable only after every outgoing invocation has exited; "Upgrade from a loader before SHA-765" now points at it instead of listing steps
- [x] 1.2 Scenario "Disable before waiting": disable first (launchers, pending runs, installed binaries and build outputs), then wait for running invocations, with the incoming version unrunnable and no source-checkout runs until the wait ends
- [x] 1.3 Scenario "Start during the cutover window" (`fp:027cee08a8f2`) and scenario "Rollback" (`fp:b4ce68d7540a`)
- [x] 1.4 `openspec validate one-loader-cutover-rule --strict`, archive with `openspec archive`, and `verify.sh` on macOS
- [x] 1.5 Review cycle 1: launchers are pointed at the incoming binary before resuming (`fp:aabb6b076544`), and a start still in progress counts as an invocation (`fp:c0612278fc78`)

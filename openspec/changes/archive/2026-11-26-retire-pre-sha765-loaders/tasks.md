# Tasks

- [x] 1.1 Requirement "Loaders older than SHA-765 are retired", with the scenarios "Kept checkout of a retired loader" (`fp:10a6fa12ef96`), "Overlapping switches" (`fp:1b92534be296`) and "Telling a retired loader apart"
- [x] 1.2 Fleet: survey spark-1 to spark-5 for sparkforge trees and binaries; remove spark-5 `~/experiments/sparkforge-probe` (the only pre-SHA-765 copy), keeping its untracked `examples/probe_sm121a.rs` as `~/experiments/probe_sm121a.rs`
- [x] 1.3 `openspec validate retire-pre-sha765-loaders --strict`, archive with `openspec archive`, and `verify.sh` on macOS

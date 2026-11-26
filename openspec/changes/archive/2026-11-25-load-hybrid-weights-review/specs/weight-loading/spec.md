## MODIFIED Requirements

### Requirement: Hybrid checkpoints load into GPU memory

`sparkforge load-weights <dir> [--ple disk|resident]` (`--features cuda`) SHALL check every tensor of a hybrid checkpoint first, then copy the checked tensors into GPU memory, one allocation per buffer of the load plan. The copied bytes plus the PLE bytes left on disk MUST equal the checked bytes. The command SHALL report the bytes copied, the time, and `MemAvailable` with the weights resident and before the copy, as admission checked it right before allocating.

#### Scenario: Flash Next on GB10, PLE table on disk
- **WHEN** `load-weights --ple disk` loads the reference checkpoint (revision 6909a5bed089) on spark-5 with nothing else on the GPU
- **THEN** 76.60 GB (71.34 GiB) in 1,777 buffers are copied and read back matching the checkpoint, and 28.80 GB of table shards stay on disk

#### Scenario: Flash Next on GB10, PLE table resident
- **WHEN** `load-weights --ple resident` loads the same checkpoint
- **THEN** 105.40 GB (98.16 GiB) in 2,033 buffers are copied and read back matching the checkpoint

#### Scenario: Memory freed while the load waits
- **WHEN** another process frees memory between admission and the allocation
- **THEN** the reported "before" figure includes that memory, so the before-and-after difference counts only this load

### Requirement: Hybrid loads are admitted before CUDA starts

A hybrid load's admission (lock and memory guard) MUST cover the copied bytes plus the largest host peak of the run's phases, which run one after another: the copy (staging and fill), the read-back, and in disk mode the timed gather (its rows pinned on the host and in a device buffer, plus one 8-byte row ID per token and head). It MUST be taken before the CUDA context is created, and MUST be held until every allocation of the run is made.

#### Scenario: Not enough memory
- **WHEN** the load would leave less than the 4 GiB headroom, as with the spark-5 vLLM lane running (25.45 GiB available)
- **THEN** it is refused before any CUDA call, stating the GiB to copy and `MemAvailable`

#### Scenario: Gather row IDs larger than the load scratch
- **WHEN** a disk-mode gather's row IDs outgrow the load's scratch, as at the maximum 1,048,576 tokens on the reference checkpoint (128 MiB of row IDs, 120 MiB of scratch), and `MemAvailable` covers the gather buffers plus the scratch but not the gather buffers plus the row IDs
- **THEN** admission is refused before any CUDA call

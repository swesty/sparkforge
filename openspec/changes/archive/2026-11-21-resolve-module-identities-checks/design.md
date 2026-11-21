# Design

## Decisions

### 1. Reject what the planner can't represent

The planner prices a layer's routed experts, the PLE table and the MTP experts each as one format. Rather than collapse mixed formats to the largest, which overstates memory, resolution refuses a group with several formats. That is the honest answer until a checkpoint needs mixed groups. The same reasoning applies to MTP: the planner models the configured draft layers, so live indices beyond them are an error, not something to fold in.

### 2. Usage tracking in both modes

Coverage says which modules an entry sets. Whether the plan ever prices those modules is a separate question, so `Formats` records the entries behind every resolved lookup, as it already did for path matching. An entry covering only the vision tower stays unused.

### 3. The embedding default

`ModelSpec::weights` is the linears' format (from the config or `--format`). Embedding tables are BF16 in every supported checkpoint unless the map says otherwise, so `embed()` defaults to BF16, as the dense planner always assumed.

### 4. Out of scope: the digest oracle

The finding needs someone else to plant links in the cache and read the loader's errors. One account owns the cache and runs the loader. The mismatch error simply stops printing the computed digest, so it reveals nothing about the file it read.

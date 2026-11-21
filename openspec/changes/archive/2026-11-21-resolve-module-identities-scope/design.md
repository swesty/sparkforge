# Design

## Decisions

### 1. One confined hop instead of following links

spark-5's cache needs exactly one extra hop, from `<repo>/blobs/<sha256>` to `hub/blobs/<xx>/<id>`. Following arbitrary links would read files outside the cache before the hash check rejected them. So the loader accepts exactly that form, opens the shared store from the pinned repository descriptor at resolution, and walks `<xx>/<name>` inside it with `O_NOFOLLOW`.

### 2. Exact module first, subtree for groups

A projection is a module (`...q_proj`), and tensors kept under it (an observer's) are separate modules that must not change its price. A group, such as a layer's experts or a sharded table, has no module at its own path, so its price aggregates the subtree. A table's scalar global scale sits at the table's own path, so the single-format checks always look at the whole subtree.

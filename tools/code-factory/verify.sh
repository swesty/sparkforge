#!/usr/bin/env bash
# Generated Code Factory verification entry point.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

cargo test --locked
cargo check --locked --features cuda
git diff --check

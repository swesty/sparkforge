//! Module identities of a checkpoint, built from its tensor names, and a ModelOpt
//! `quantized_layers` map resolved onto them.
//!
//! Matching map paths against module-name patterns let stale or alias entries price real modules
//! (codex-review of swesty/sparkforge#9 and #10). Here each map entry applies only to modules that
//! exist in the checkpoint: an entry covers the modules at its path or under it, two entries that
//! cover one module must agree, and an entry that covers nothing is reported as unused.
use crate::config::{strip_text_prefix, FormatMap, WeightFormat};
use anyhow::{bail, Result};
use std::collections::{BTreeMap, BTreeSet};

/// The canonical identity of a module path: any language-model prefix removed
/// (`model.language_model.layers.0.mlp.experts` -> `layers.0.mlp.experts`). Other towers keep
/// their own names (`mtp.layers.0...`, `visual.blocks.0...`, `lm_head`).
pub fn canonical(path: &str) -> &str {
    strip_text_prefix(path)
}

/// Every module a checkpoint holds, by canonical path. A module is a tensor name without its last
/// component (`...q_proj.weight` and `...q_proj.weight_scale` both belong to `...q_proj`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleTable {
    modules: BTreeSet<String>,
}

impl ModuleTable {
    pub fn from_tensor_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let modules = names
            .into_iter()
            .filter_map(|n| n.rsplit_once('.').map(|(m, _)| canonical(m).to_owned()))
            .collect();
        ModuleTable { modules }
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Modules at `path` or under it (`path` itself, or `path.`-prefixed).
    pub fn covered<'s>(&'s self, path: &str) -> impl Iterator<Item = &'s str> + 's {
        let exact = self.modules.get(path).map(String::as_str);
        let under = format!("{path}.");
        // Names starting with `path.` sort together, right after `path.` itself.
        let below = self.modules.range(under.clone()..).take_while(move |m| m.starts_with(&under)).map(String::as_str);
        exact.into_iter().chain(below)
    }

    /// The distinct next path components under `path` (`children("mtp.layers")` gives the MTP
    /// layer indices the checkpoint holds).
    pub fn children(&self, path: &str) -> BTreeSet<&str> {
        let under = format!("{path}.");
        self.covered(path).filter_map(|m| m.strip_prefix(under.as_str())).map(|r| r.split('.').next().unwrap_or(r)).collect()
    }

    /// Whether any module exists at `path` or under it.
    pub fn has(&self, path: &str) -> bool {
        self.covered(path).next().is_some()
    }
}

/// A `quantized_layers` map resolved onto a checkpoint's modules.
#[derive(Debug, Clone, Default)]
pub struct ResolvedFormats {
    pub table: ModuleTable,
    /// Format of every module some entry covers, with every (agreeing) entry that covers it.
    by_module: BTreeMap<String, (WeightFormat, Vec<String>)>,
    /// Map entries that cover no module of the checkpoint (stale indices, paths of another layout).
    pub unused: Vec<String>,
}

impl ResolvedFormats {
    /// The format of one module, `default` when no entry covers it.
    pub fn format_of(&self, module: &str, default: WeightFormat) -> WeightFormat {
        self.by_module.get(module).map_or(default, |(f, _)| *f)
    }

    /// The modules a lookup of `path` prices: the module at exactly `path` when the checkpoint
    /// has one (a projection: tensors kept under it, such as an observer's, don't reprice it),
    /// else every module under it (a group such as a layer's routed experts or a sharded table).
    pub fn priced<'s>(&'s self, path: &'s str) -> Vec<&'s str> {
        if self.table.covered(path).next() == Some(path) {
            vec![path]
        } else {
            self.table.covered(path).collect()
        }
    }

    /// The formats of the modules a lookup of `path` prices (see `priced`): `(format, count)` per
    /// distinct format, with `default` for modules no entry covers. Empty when no module exists.
    pub fn formats_at(&self, path: &str, default: WeightFormat) -> Vec<(WeightFormat, usize)> {
        let mut out: Vec<(WeightFormat, usize)> = vec![];
        for m in self.priced(path) {
            let f = self.format_of(m, default);
            match out.iter_mut().find(|(g, _)| *g == f) {
                Some(g) => g.1 += 1,
                None => out.push((f, 1)),
            }
        }
        out
    }

    /// The formats of every module at or under `path`, for checking that a group the planner
    /// prices as one format really has one (`formats_at` would stop at a module at `path` itself,
    /// such as a table's scalar global scale beside its shards).
    pub fn formats_under(&self, path: &str, default: WeightFormat) -> Vec<(WeightFormat, usize)> {
        let mut out: Vec<(WeightFormat, usize)> = vec![];
        for m in self.table.covered(path) {
            let f = self.format_of(m, default);
            match out.iter_mut().find(|(g, _)| *g == f) {
                Some(g) => g.1 += 1,
                None => out.push((f, 1)),
            }
        }
        out
    }

    /// Every map entry that covers a module a lookup of `path` prices.
    pub fn entries_at(&self, path: &str) -> BTreeSet<&str> {
        self.priced(path).into_iter().filter_map(|m| self.by_module.get(m)).flat_map(|(_, es)| es.iter().map(String::as_str)).collect()
    }

    /// One format for the modules a lookup of `path` prices: the largest when they differ (the
    /// budget then never undercounts), `None` when no module exists there.
    pub fn format_at(&self, path: &str, default: WeightFormat) -> Option<WeightFormat> {
        self.formats_at(path, default)
            .into_iter()
            .map(|(f, _)| f)
            .reduce(|a, b| if b.bytes_per_elem() > a.bytes_per_elem() { b } else { a })
    }
}

/// Applies each entry of `map` to the modules of `table` at or under its canonical path.
pub fn resolve(map: &FormatMap, table: ModuleTable) -> Result<ResolvedFormats> {
    let mut by_module: BTreeMap<String, (WeightFormat, Vec<String>)> = BTreeMap::new();
    let mut unused = vec![];
    for (entry, &f) in &map.modules {
        let covered: Vec<&str> = table.covered(canonical(entry)).collect();
        if covered.is_empty() {
            unused.push(entry.clone());
            continue;
        }
        for m in covered {
            match by_module.get_mut(m) {
                Some((g, others)) if *g != f => bail!(
                    "quantized_layers lists {} as {g:?} and {entry} as {f:?}; both cover the checkpoint module {m}: conflicting formats for one module",
                    others[0]
                ),
                Some((_, others)) => others.push(entry.clone()),
                None => {
                    by_module.insert(m.to_owned(), (f, vec![entry.clone()]));
                }
            }
        }
    }
    Ok(ResolvedFormats { table, by_module, unused })
}

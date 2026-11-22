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

    /// Whether a module exists at exactly `path`.
    pub fn contains(&self, path: &str) -> bool {
        self.modules.contains(path)
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

    /// The format of a single module (a projection): the module at exactly `path`, `None` when the
    /// checkpoint has none there. Tensors kept under it, such as an observer's, never stand in for
    /// it, so a projection missing from the checkpoint is not priced from its subtree.
    pub fn format_at(&self, path: &str, default: WeightFormat) -> Option<WeightFormat> {
        self.table.contains(path).then(|| self.format_of(path, default))
    }

    /// Every map entry that covers the module at exactly `path`.
    pub fn entries_at(&self, path: &str) -> BTreeSet<&str> {
        self.by_module.get(path).into_iter().flat_map(|(_, es)| es.iter().map(String::as_str)).collect()
    }

    /// The members of a group (a layer's routed experts, a sharded table): the modules strictly
    /// under `path` when there are any, else the module at `path`. A module at the group's own path
    /// beside members holds only shared scalars (a table's global scale), so it has no format of
    /// its own to price or to check.
    pub fn group_members<'s>(&'s self, path: &str) -> Vec<&'s str> {
        let all: Vec<&str> = self.table.covered(path).collect();
        if all.len() > 1 && all[0] == path {
            all[1..].to_vec()
        } else {
            all
        }
    }

    /// The formats of a group's members (see `group_members`): `(format, count)` per distinct
    /// format, with `default` for members no entry covers. Empty when the group is absent.
    pub fn group_formats(&self, path: &str, default: WeightFormat) -> Vec<(WeightFormat, usize)> {
        let mut out: Vec<(WeightFormat, usize)> = vec![];
        for m in self.group_members(path) {
            let f = self.format_of(m, default);
            match out.iter_mut().find(|(g, _)| *g == f) {
                Some(g) => g.1 += 1,
                None => out.push((f, 1)),
            }
        }
        out
    }

    /// One format for a group: the largest when its members differ (the budget then never
    /// undercounts), `None` when the group is absent.
    pub fn group_format(&self, path: &str, default: WeightFormat) -> Option<WeightFormat> {
        self.group_formats(path, default)
            .into_iter()
            .map(|(f, _)| f)
            .reduce(|a, b| if b.bytes_per_elem() > a.bytes_per_elem() { b } else { a })
    }

    /// Every map entry that covers a module at or under `path`: the group's members and a module at
    /// its own path both belong to it, so per-shard entries count as used beside a table scalar.
    pub fn group_entries(&self, path: &str) -> BTreeSet<&str> {
        self.table.covered(path).filter_map(|m| self.by_module.get(m)).flat_map(|(_, es)| es.iter().map(String::as_str)).collect()
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

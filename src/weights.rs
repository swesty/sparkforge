//! Hugging Face safetensors checkpoints: open, validate against a plan, and read tensor bytes.
//! Host-only (no CUDA); GPU binding lives in `device_weights` behind `--features cuda`.
//!
//! safetensors layout: an 8-byte little-endian header length N, then N bytes of JSON mapping each
//! tensor name to `{dtype, shape, data_offsets: [begin, end]}` (offsets relative to byte 8 + N),
//! then the tensor data, which the tensors must cover exactly (no overlaps, no holes).
//!
//! Every read is a fallible positioned read. Nothing is memory-mapped: a mapped file that another
//! process truncates turns a read into SIGBUS, while `pread` returns an error we can report.
//!
//! Untrusted-input rules (codex-review cycles 1 and 2): shard names from the index must stay inside
//! the checkpoint directory; repeated JSON keys are errors, not last-one-wins; shards are opened
//! without blocking and must be regular files; and a shard modified after it was opened fails the
//! next read instead of mixing two versions of the checkpoint.
use crate::config::{ModelSpec, WeightFormat};
use anyhow::{bail, ensure, Context, Result};
use serde::de::{Deserializer, IgnoredAny, MapAccess, Visitor};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::path::{Component, Path, PathBuf};

/// Largest header we accept (the reference safetensors implementation uses the same 100 MB cap).
const MAX_HEADER_BYTES: u64 = 100 << 20;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shard: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte range inside the shard file.
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

pub struct Shard {
    pub path: PathBuf,
    /// File length when opened.
    pub len: u64,
    /// Identity and modification stamp when opened; every read re-checks it afterwards.
    stamp: FileStamp,
    file: File,
}

/// What identifies one version of an open file. A write after open changes `ctime` (which no
/// unprivileged call can set back) and usually `mtime`; truncation changes `len`.
///
/// Limitation: timestamps have the filesystem's granularity (on Linux, the kernel's coarse clock,
/// a few ms), so a write landing in the same tick as the open can go unseen. Detection is
/// best-effort against concurrent writers, not a cryptographic snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    #[cfg(unix)]
    id: (u64, u64, i64, i64, i64, i64),
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

impl FileStamp {
    fn of(file: &File) -> std::io::Result<Self> {
        let m = file.metadata()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(FileStamp { len: m.len(), id: (m.dev(), m.ino(), m.mtime(), m.mtime_nsec(), m.ctime(), m.ctime_nsec()) })
        }
        #[cfg(not(unix))]
        {
            Ok(FileStamp { len: m.len(), modified: m.modified().ok() })
        }
    }
}

/// Largest metadata file (config or index) read into memory; Flash Next's index is about 33 MB.
const MAX_METADATA_BYTES: u64 = 256 << 20;

/// A checkpoint directory opened once. The config, the index and every shard are opened relative
/// to this descriptor (`openat`), so replacing the directory, or a symlink on its path, while a
/// load runs cannot mix two generations (codex-review of SHA-765).
pub struct CheckpointDir {
    /// Canonical path at resolution time; used for messages only.
    pub path: PathBuf,
    #[cfg(unix)]
    fd: File,
}

impl CheckpointDir {
    fn entry(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Opens `name` inside the directory without blocking (a FIFO with no writer would otherwise
    /// hang `open`) and requires the opened descriptor to be a regular file.
    ///
    /// No symlink on the way is followed, except one at the entry itself: a Hugging Face cache
    /// snapshot links each shard to `../../blobs/<hash>`. That link's target is walked the same way,
    /// one component at a time with `O_NOFOLLOW`, so a target that passes through another symlink
    /// (for example a `current` alias that a publisher switches) is refused rather than followed
    /// into whatever generation the alias names now (codex-review of SHA-765).
    fn open(&self, name: &str) -> Result<File> {
        let path = self.entry(name);
        #[cfg(unix)]
        let opened = {
            use std::io::ErrorKind;
            let (parent, last) = nofollow::walk(&self.fd, Path::new(name))
                .map_err(|e| nofollow::explain(e, &path, "passes through a symlink or a missing directory"))?;
            match nofollow::open_file(&parent, &last) {
                Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
                    let target = nofollow::readlink(&parent, &last).with_context(|| format!("reading the link {}", path.display()))?;
                    let hop = nofollow::walk(&parent, &target).and_then(|(p, l)| nofollow::open_file(&p, &l));
                    match hop {
                        Err(e) if e.kind() == ErrorKind::NotFound => Err(e),
                        Err(e) if matches!(e.raw_os_error(), Some(libc::ELOOP) | Some(libc::ENOTDIR)) => bail!(
                            "{} links to {}, which passes through another symlink, so the generation it names can change mid-load; link straight to the file",
                            path.display(), target.display()
                        ),
                        r => r,
                    }
                }
                r => r,
            }
        };
        #[cfg(not(unix))]
        let opened = File::open(&path);
        let file = match opened {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!("{} is missing", path.display()),
            Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
        };
        let m = file.metadata().with_context(|| format!("inspecting {}", path.display()))?;
        ensure!(m.file_type().is_file(), "{} is not a regular file", path.display());
        Ok(file)
    }

    /// Whether the directory entry `name` exists; a dangling symlink counts as present.
    fn has_entry(&self, name: &str) -> Result<bool> {
        let path = self.entry(name);
        #[cfg(unix)]
        let r = {
            use std::os::fd::AsRawFd;
            let c = std::ffi::CString::new(name).with_context(|| format!("{name:?} contains a NUL byte"))?;
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            let rc = unsafe { libc::fstatat(self.fd.as_raw_fd(), c.as_ptr(), &mut st, libc::AT_SYMLINK_NOFOLLOW) };
            if rc == 0 { Ok(()) } else { Err(std::io::Error::last_os_error()) }
        };
        #[cfg(not(unix))]
        let r = std::fs::symlink_metadata(&path).map(|_| ());
        match r {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e).with_context(|| format!("inspecting {}", path.display())),
        }
    }

    /// Reads a metadata file (config or index) with the shard open rules plus a size cap.
    pub fn read_metadata(&self, name: &str) -> Result<String> {
        let path = self.entry(name);
        let file = self.open(name)?;
        let len = file.metadata()?.len();
        ensure!(len <= MAX_METADATA_BYTES, "{} is {len} bytes, above the {MAX_METADATA_BYTES}-byte metadata limit", path.display());
        read_capped(file, MAX_METADATA_BYTES, &path.display().to_string())
    }
}

/// Path walking that never follows a symlink (`O_NOFOLLOW` on every component).
#[cfg(unix)]
mod nofollow {
    use std::ffi::CString;
    use std::fs::File;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Component, Path, PathBuf};

    fn openat(dir: &File, name: &std::ffi::CStr, flags: i32) -> io::Result<File> {
        let fd = unsafe { libc::openat(dir.as_raw_fd(), name.as_ptr(), flags | libc::O_NOFOLLOW | libc::O_CLOEXEC) };
        if fd < 0 { Err(io::Error::last_os_error()) } else { Ok(unsafe { File::from_raw_fd(fd) }) }
    }

    fn cstr(s: &std::ffi::OsStr) -> io::Result<CString> {
        CString::new(s.as_bytes()).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))
    }

    /// Opens every directory of `path` from `start` (or from `/` for an absolute path) without
    /// following symlinks, and returns the last directory with the final component's name.
    pub fn walk(start: &File, path: &Path) -> io::Result<(File, CString)> {
        let comps: Vec<Component> = path.components().filter(|c| !matches!(c, Component::CurDir)).collect();
        let Some((last, dirs)) = comps.split_last() else { return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty path")) };
        let Component::Normal(last) = last else { return Err(io::Error::new(io::ErrorKind::InvalidInput, "path does not end in a file name")) };
        let mut cur = start.try_clone()?;
        for c in dirs {
            let name = match c {
                Component::RootDir => {
                    cur = File::open("/")?;
                    continue;
                }
                Component::ParentDir => cstr("..".as_ref())?,
                Component::Normal(n) => cstr(n)?,
                _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "unsupported path component")),
            };
            cur = openat(&cur, &name, libc::O_RDONLY | libc::O_DIRECTORY)?;
        }
        Ok((cur, cstr(last)?))
    }

    /// Opens a file non-blocking; a symlink there fails with `ELOOP`.
    pub fn open_file(dir: &File, name: &std::ffi::CStr) -> io::Result<File> {
        openat(dir, name, libc::O_RDONLY | libc::O_NONBLOCK)
    }

    pub fn readlink(dir: &File, name: &std::ffi::CStr) -> io::Result<PathBuf> {
        let mut buf = vec![0u8; libc::PATH_MAX as usize];
        let n = unsafe { libc::readlinkat(dir.as_raw_fd(), name.as_ptr(), buf.as_mut_ptr().cast(), buf.len()) };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        buf.truncate(n as usize);
        Ok(PathBuf::from(std::ffi::OsString::from_vec(buf)))
    }

    /// Error for a failed walk: missing stays "missing", a symlink on the way is named as such.
    pub fn explain(e: io::Error, path: &Path, what: &str) -> anyhow::Error {
        if e.kind() == io::ErrorKind::NotFound {
            anyhow::anyhow!("{} is missing", path.display())
        } else if matches!(e.raw_os_error(), Some(libc::ELOOP) | Some(libc::ENOTDIR)) {
            anyhow::anyhow!("{} {what}", path.display())
        } else {
            anyhow::Error::new(e).context(format!("opening {}", path.display()))
        }
    }
    use std::os::unix::ffi::OsStringExt;
}

/// Reads at most `cap` bytes of UTF-8. A source that yields more (a file that grew after its size
/// was checked) is an error, never a silently truncated prefix that may still parse.
pub fn read_capped(r: impl std::io::Read, cap: u64, what: &str) -> Result<String> {
    use std::io::Read;
    let mut buf = Vec::new();
    r.take(cap + 1).read_to_end(&mut buf).with_context(|| format!("reading {what}"))?;
    ensure!(buf.len() as u64 <= cap, "{what} grew past the {cap}-byte metadata limit while it was read");
    String::from_utf8(buf).with_context(|| format!("{what} is not UTF-8"))
}

/// Resolves a checkpoint directory once and opens it, so the config, the index and every shard
/// come from one generation even if the directory or a symlink on its path is replaced mid-load.
pub fn resolve_checkpoint_dir(dir: &Path) -> Result<CheckpointDir> {
    let path = std::fs::canonicalize(dir).with_context(|| format!("resolving checkpoint directory {}", dir.display()))?;
    ensure!(path.is_dir(), "{} is not a directory", path.display());
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW: if the canonical entry became a symlink since `canonicalize`, fail.
        let fd = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(&path).with_context(|| format!("opening checkpoint directory {}", path.display()))?;
        Ok(CheckpointDir { path, fd })
    }
    #[cfg(not(unix))]
    Ok(CheckpointDir { path })
}

/// `config.json` of an opened checkpoint directory, read with the metadata open rules.
pub fn read_config(dir: &CheckpointDir) -> Result<String> {
    dir.read_metadata("config.json")
}

/// Index shard names must be plain relative paths inside the checkpoint directory: no absolute
/// paths, no `..`, no drive prefixes. (Lexical only; symlinked shards are allowed, see `CheckpointDir::open`.)
fn check_shard_name(name: &str) -> Result<()> {
    let p = Path::new(name);
    ensure!(
        !name.is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "index names shard {name:?}, which is not a plain relative path inside the checkpoint directory"
    );
    Ok(())
}

/// A JSON object whose keys must be unique. serde's map types keep the last of repeated keys, which
/// would let a malformed file pass validation with a meaning that depends on the reader.
struct UniqueMap<V>(Vec<(String, V)>);

impl<'de, V: serde::Deserialize<'de>> serde::Deserialize<'de> for UniqueMap<V> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V_<V>(std::marker::PhantomData<V>);
        impl<'de, V: serde::Deserialize<'de>> Visitor<'de> for V_<V> {
            type Value = UniqueMap<V>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> std::result::Result<Self::Value, A::Error> {
                let mut seen = HashSet::new();
                let mut out = vec![];
                while let Some(k) = a.next_key::<String>()? {
                    if !seen.insert(k.clone()) {
                        return Err(serde::de::Error::custom(format!("duplicate key {k:?}")));
                    }
                    out.push((k, a.next_value()?));
                }
                Ok(UniqueMap(out))
            }
        }
        d.deserialize_map(V_(std::marker::PhantomData))
    }
}

/// A safetensors header: tensor entries plus an optional `__metadata__` object, keys unique.
struct Header(Vec<(String, RawTensor)>);

impl<'de> serde::Deserialize<'de> for Header {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct HV;
        impl<'de> Visitor<'de> for HV {
            type Value = Header;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a safetensors header object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> std::result::Result<Header, A::Error> {
                let mut seen = HashSet::new();
                let mut out = vec![];
                while let Some(k) = a.next_key::<String>()? {
                    if !seen.insert(k.clone()) {
                        return Err(serde::de::Error::custom(format!("duplicate tensor name {k:?}")));
                    }
                    if k == "__metadata__" {
                        a.next_value::<IgnoredAny>()?;
                    } else {
                        // RawTensor's derived impl rejects repeated fields such as two `dtype`s.
                        let t: RawTensor = a.next_value().map_err(|e| serde::de::Error::custom(format!("tensor {k}: {e}")))?;
                        out.push((k, t));
                    }
                }
                Ok(Header(out))
            }
        }
        d.deserialize_map(HV)
    }
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub shards: Vec<Shard>,
    pub tensors: HashMap<String, TensorInfo>,
}

fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "BF16" | "F16" | "I16" | "U16" => 2,
        "F32" | "I32" | "U32" => 4,
        "F64" | "I64" | "U64" => 8,
        _ => return None,
    })
}

#[derive(serde::Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn pread(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, at)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], at + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
}

impl Checkpoint {
    /// Opens `dir/model.safetensors.index.json` if present, else `dir/model.safetensors`. With an
    /// index, every entry must name the shard whose header actually holds that tensor, and every
    /// tensor in a shard header must be listed in the index.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_dir(&resolve_checkpoint_dir(dir)?)
    }

    /// `open` on an already opened directory: the index and every shard are opened relative to it.
    pub fn open_dir(dir: &CheckpointDir) -> Result<Self> {
        const INDEX: &str = "model.safetensors.index.json";
        let index_path = dir.entry(INDEX);
        // Presence is the directory entry itself: a dangling index symlink is an error, never a
        // silent fallback to a single-file checkpoint that may be stale.
        let index: Option<BTreeMap<String, String>> = if dir.has_entry(INDEX)? {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: UniqueMap<String>,
            }
            let idx: Index = serde_json::from_str(&dir.read_metadata(INDEX)?)
                .with_context(|| format!("parsing {}", index_path.display()))?;
            for (_, shard) in &idx.weight_map.0 {
                check_shard_name(shard)?;
            }
            Some(idx.weight_map.0.into_iter().collect())
        } else {
            None
        };
        let shard_files: Vec<String> = match &index {
            Some(m) => {
                let mut f: Vec<String> = m.values().cloned().collect();
                f.sort();
                f.dedup();
                f
            }
            None => vec!["model.safetensors".into()],
        };
        let mut ck = Checkpoint { dir: dir.path.clone(), shards: vec![], tensors: HashMap::new() };
        for f in &shard_files {
            let path = dir.entry(f);
            let file = dir.open(f).context("opening a shard")?;
            let stamp = FileStamp::of(&file)?;
            let len = stamp.len;
            let shard = ck.shards.len();
            for t in read_header(&path, &file, len, shard)? {
                if let Some(m) = &index {
                    match m.get(&t.name) {
                        Some(declared) if declared == f => {}
                        Some(declared) => bail!("index lists {} in {declared}, but it is in {}", t.name, path.display()),
                        None => bail!("{} holds {}, which the index does not list", path.display(), t.name),
                    }
                }
                if let Some(prev) = ck.tensors.get(&t.name) {
                    bail!("tensor {} appears in both {} and {}", t.name, ck.shards[prev.shard].path.display(), path.display());
                }
                ck.tensors.insert(t.name.clone(), t);
            }
            // A write between the open and the header read would make the header disagree with the
            // stamp every later read is checked against.
            ensure!(FileStamp::of(&file)? == stamp, "{} changed while its header was being read", path.display());
            ck.shards.push(Shard { path, len, stamp, file });
        }
        if let Some(m) = &index {
            for (name, declared) in m {
                ensure!(ck.tensors.contains_key(name), "index lists {name} in {declared}, but that shard's header has no such tensor");
            }
        }
        Ok(ck)
    }

    /// Reads `buf.len()` bytes of tensor `t` starting `off` bytes into it. Fails, rather than
    /// faulting, if the range is outside the tensor or the shard was modified since it was opened.
    ///
    /// The stamp is checked *after* the read: if it still matches the one taken at open, no write
    /// reached the file (to timestamp granularity) before these bytes were read, so every chunk of a
    /// load comes from the version that was validated. A same-length rewrite between two chunks
    /// fails the second chunk instead of silently mixing versions.
    pub fn read_range(&self, t: &TensorInfo, off: u64, buf: &mut [u8]) -> Result<()> {
        let end = off.checked_add(buf.len() as u64).context("read range overflows")?;
        ensure!(end <= t.len(), "read of {}..{end} is outside tensor {} ({} bytes)", off, t.name, t.len());
        let shard = &self.shards[t.shard];
        let before = FileStamp::of(&shard.file)?;
        ensure!(before.len == shard.len, "{} changed size since it was opened ({} -> {} bytes)", shard.path.display(), shard.len, before.len);
        pread(&shard.file, buf, t.start + off).with_context(|| format!("reading {} from {}", t.name, shard.path.display()))?;
        let after = FileStamp::of(&shard.file)?;
        ensure!(after.len == shard.len, "{} changed size since it was opened ({} -> {} bytes)", shard.path.display(), shard.len, after.len);
        ensure!(after == shard.stamp, "{} was modified since it was opened; refusing to mix checkpoint versions", shard.path.display());
        Ok(())
    }

    /// Reads a whole tensor into `buf` (resized to fit).
    pub fn read_into(&self, t: &TensorInfo, buf: &mut Vec<u8>) -> Result<()> {
        buf.resize(usize::try_from(t.len())?, 0);
        self.read_range(t, 0, buf)
    }
}

fn read_header(path: &Path, file: &File, file_len: u64, shard: usize) -> Result<Vec<TensorInfo>> {
    ensure!(file_len >= 8, "{}: shorter than a safetensors header", path.display());
    let mut n8 = [0u8; 8];
    pread(file, &mut n8, 0)?;
    let n = u64::from_le_bytes(n8);
    ensure!(n <= MAX_HEADER_BYTES, "{}: header length {n} exceeds the {MAX_HEADER_BYTES}-byte limit", path.display());
    let data_start = 8u64.checked_add(n).context("header length overflows")?;
    ensure!(data_start <= file_len, "{}: header length {n} runs past the end of the file", path.display());
    let mut header = vec![0u8; n as usize];
    pread(file, &mut header, 8)?;
    parse_header(path, &header, data_start, file_len, shard)
}

/// Validates every tensor entry against the data section `[data_start, file_len)` before forming
/// absolute offsets, then requires the non-empty ranges to tile the data section exactly.
fn parse_header(path: &Path, header: &[u8], data_start: u64, file_len: u64, shard: usize) -> Result<Vec<TensorInfo>> {
    let data_len = file_len - data_start;
    let raw: Header = serde_json::from_slice(header).with_context(|| format!("{}: parsing header JSON", path.display()))?;
    let mut out = vec![];
    for (name, t) in raw.0 {
        let [b, e] = t.data_offsets;
        ensure!(b <= e && e <= data_len, "{}: tensor {name} byte range {b}..{e} lies outside the file's {data_len}-byte data section", path.display());
        let elem = dtype_size(&t.dtype).with_context(|| format!("{}: tensor {name} has unknown dtype {}", path.display(), t.dtype))?;
        let want = t
            .shape
            .iter()
            .try_fold(elem, |acc, &d| acc.checked_mul(d))
            .with_context(|| format!("{}: tensor {name} shape {:?} overflows", path.display(), t.shape))?;
        ensure!(e - b == want, "{}: tensor {name} has {} bytes but {} {:?} needs {want}", path.display(), e - b, t.dtype, t.shape);
        let shape = t.shape.iter().map(|&d| usize::try_from(d)).collect::<std::result::Result<Vec<_>, _>>()?;
        out.push(TensorInfo { name, shard, dtype: t.dtype, shape, start: data_start + b, end: data_start + e });
    }
    // The tensors must cover the data section exactly: no overlaps (one tensor aliasing another)
    // and no unaccounted bytes. Empty tensors take no space and are skipped.
    let mut ranges: Vec<&TensorInfo> = out.iter().filter(|t| !t.is_empty()).collect();
    ranges.sort_by_key(|t| (t.start, t.end));
    let mut cursor = data_start;
    for t in &ranges {
        ensure!(t.start >= cursor, "{}: tensor {} ({}..{}) overlaps the tensor before it", path.display(), t.name, t.start - data_start, t.end - data_start);
        ensure!(t.start == cursor, "{}: {} unaccounted bytes before tensor {}", path.display(), t.start - cursor, t.name);
        cursor = t.end;
    }
    ensure!(cursor == file_len, "{}: {} unaccounted bytes after the last tensor", path.display(), file_len - cursor);
    Ok(out)
}

/// f64 reference for a bound linear's GEMV, `y = W x`, read from the checkpoint in bounded chunks
/// so peak host memory is about `threads * REF_CHUNK_BYTES`, independent of the tensor size.
pub const REF_CHUNK_BYTES: usize = 4 << 20;
pub const REF_THREADS: usize = 16;

pub fn reference_gemv(ck: &Checkpoint, l: &BoundLinear, x: &[f64]) -> Result<Vec<f64>> {
    ensure!(x.len() == l.k, "input has {} elements, linear needs {}", x.len(), l.k);
    let row_bytes = l.k * 2;
    // (tensor, first row within that tensor, output row) for every part, so threads can split rows.
    let mut parts = vec![];
    let mut out_row = 0;
    for name in &l.parts {
        let t = &ck.tensors[name];
        let rows = usize::try_from(t.len())? / row_bytes;
        parts.push((t, out_row, rows));
        out_row += rows;
    }
    ensure!(out_row == l.n, "{} has {out_row} rows, plan expects {}", l.key(), l.n);
    let mut y = vec![0f64; l.n];
    let per = l.n.div_ceil(REF_THREADS);
    let rows_per_read = (REF_CHUNK_BYTES / row_bytes).max(1);
    std::thread::scope(|sc| -> Result<()> {
        let handles: Vec<_> = y
            .chunks_mut(per)
            .enumerate()
            .map(|(ci, out)| {
                let parts = &parts;
                sc.spawn(move || -> Result<()> {
                    let first = ci * per;
                    let mut buf = vec![0u8; rows_per_read * row_bytes];
                    let mut r = 0;
                    while r < out.len() {
                        let global = first + r;
                        let &(t, base, rows) = parts.iter().find(|(_, b, n)| global >= *b && global < b + n).unwrap();
                        let local = global - base;
                        let take = rows_per_read.min(out.len() - r).min(rows - local);
                        let bytes = &mut buf[..take * row_bytes];
                        ck.read_range(t, (local * row_bytes) as u64, bytes)?;
                        for (j, row) in bytes.chunks_exact(row_bytes).enumerate() {
                            out[r + j] = row
                                .chunks_exact(2)
                                .zip(x)
                                .map(|(c, &xv)| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64 * xv)
                                .sum();
                        }
                        r += take;
                    }
                    Ok(())
                })
            })
            .collect();
        for h in handles {
            h.join().expect("reference thread panicked")?;
        }
        Ok(())
    })?;
    Ok(y)
}

/// One planned linear: the checkpoint tensors that make it, stacked by rows in this order.
#[derive(Debug, Clone)]
pub struct BoundLinear {
    /// Plan linear name (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `lm_head`).
    pub linear: String,
    pub layer: Option<u32>,
    pub parts: Vec<String>,
    pub n: usize,
    pub k: usize,
}

impl BoundLinear {
    pub fn fused(&self) -> bool {
        self.parts.len() > 1
    }
    pub fn key(&self) -> String {
        match self.layer {
            Some(l) => format!("layers.{l}.{}", self.linear),
            None => self.linear.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Binding {
    pub linears: Vec<BoundLinear>,
    /// Non-linear tensors the model needs (norms, embedding, optional q/k norms and biases).
    pub others: Vec<String>,
    /// Tensors in the checkpoint that the plan does not use.
    pub unexpected: Vec<String>,
}

/// Validates the checkpoint against the spec and returns which tensors make each linear.
pub fn bind(ck: &Checkpoint, m: &ModelSpec) -> Result<Binding> {
    if m.hybrid.is_some() || !m.formats.is_empty() || m.declared_mixed {
        bail!("{} checkpoints (hybrid layout or mixed-precision weights) do not load yet; the mixed-format loader is a later ticket", m.arch);
    }
    if m.moe.is_some() {
        bail!("MoE weight loading is not supported yet ({})", m.arch);
    }
    if m.weights != WeightFormat::Bf16 {
        bail!("weight format {:?} is not supported yet; only BF16 checkpoints load today", m.weights);
    }
    let (h, q, kv, inter) = (m.hidden as usize, m.q_dim() as usize, m.kv_dim() as usize, m.intermediate as usize);
    let mut need: Vec<(String, Vec<usize>)> = vec![];
    let mut linears = vec![];
    let mut others = vec![];
    let mut lin = |linear: &str, layer: Option<u32>, parts: Vec<(String, usize)>, k: usize, need: &mut Vec<(String, Vec<usize>)>| {
        for (name, rows) in &parts {
            need.push((name.clone(), vec![*rows, k]));
        }
        let n = parts.iter().map(|p| p.1).sum();
        linears.push(BoundLinear { linear: linear.into(), layer, parts: parts.into_iter().map(|p| p.0).collect(), n, k });
    };
    for l in 0..m.layers {
        let p = format!("model.layers.{l}");
        let a = format!("{p}.self_attn");
        let f = format!("{p}.mlp");
        lin("qkv_proj", Some(l), vec![(format!("{a}.q_proj.weight"), q), (format!("{a}.k_proj.weight"), kv), (format!("{a}.v_proj.weight"), kv)], h, &mut need);
        lin("o_proj", Some(l), vec![(format!("{a}.o_proj.weight"), h)], q, &mut need);
        lin("gate_up_proj", Some(l), vec![(format!("{f}.gate_proj.weight"), inter), (format!("{f}.up_proj.weight"), inter)], h, &mut need);
        lin("down_proj", Some(l), vec![(format!("{f}.down_proj.weight"), h)], inter, &mut need);
        let mut add = |name: String, shape: Vec<usize>| {
            others.push(name.clone());
            need.push((name, shape));
        };
        add(format!("{p}.input_layernorm.weight"), vec![h]);
        add(format!("{p}.post_attention_layernorm.weight"), vec![h]);
        if m.qk_norm {
            add(format!("{a}.q_norm.weight"), vec![m.head_dim as usize]);
            add(format!("{a}.k_norm.weight"), vec![m.head_dim as usize]);
        }
        if m.qkv_bias {
            add(format!("{a}.q_proj.bias"), vec![q]);
            add(format!("{a}.k_proj.bias"), vec![kv]);
            add(format!("{a}.v_proj.bias"), vec![kv]);
        }
    }
    let embed = "model.embed_tokens.weight".to_string();
    others.push(embed.clone());
    need.push((embed.clone(), vec![m.vocab as usize, h]));
    others.push("model.norm.weight".into());
    need.push(("model.norm.weight".into(), vec![h]));
    let head = if m.tie_embeddings { embed } else { "lm_head.weight".to_string() };
    if !m.tie_embeddings {
        need.push((head.clone(), vec![m.vocab as usize, h]));
    }
    linears.push(BoundLinear { linear: "lm_head".into(), layer: None, parts: vec![head], n: m.vocab as usize, k: h });

    let mut problems = vec![];
    for (name, shape) in &need {
        match ck.tensors.get(name) {
            None => problems.push(format!("missing {name} (expected BF16 {shape:?})")),
            Some(t) if &t.shape != shape => problems.push(format!("{name}: expected shape {shape:?}, found {:?}", t.shape)),
            Some(t) if t.dtype != "BF16" => problems.push(format!("{name}: expected dtype BF16, found {}", t.dtype)),
            Some(_) => {}
        }
    }
    if !problems.is_empty() {
        bail!("checkpoint {} does not match the plan:\n  {}", ck.dir.display(), problems.join("\n  "));
    }
    let used: std::collections::HashSet<&str> = need.iter().map(|(n, _)| n.as_str()).collect();
    let mut unexpected: Vec<String> = ck.tensors.keys().filter(|k| !used.contains(k.as_str())).cloned().collect();
    unexpected.sort();
    Ok(Binding { linears, others, unexpected })
}

/// Bytes `MemAvailable` must keep after a load (design decision 5).
pub const HEADROOM_BYTES: u64 = 4 << 30;

/// Host staging buffer for uploads: tensors are copied to the GPU in pieces of at most this size.
pub const STAGING_BYTES: u64 = 64 << 20;

/// Host memory a load needs beyond the copied weights: the staging buffer plus, when verifying,
/// the reference's per-thread read buffers and its f64 output (one value per output row).
pub fn load_overhead_bytes(b: &Binding, verifying: bool) -> u64 {
    let refs = if verifying {
        let max_n = b.linears.iter().map(|l| l.n).max().unwrap_or(0) as u64;
        (REF_THREADS * REF_CHUNK_BYTES) as u64 + max_n * 8
    } else {
        0
    };
    STAGING_BYTES + refs
}

/// Refuses a load whose copied bytes would leave less than `HEADROOM_BYTES` available.
pub fn check_headroom(copy_bytes: u64, mem_available: u64) -> Result<()> {
    ensure!(
        copy_bytes + HEADROOM_BYTES <= mem_available,
        "refusing to copy {:.2} GiB: MemAvailable is {:.2} GiB and {:.0} GiB must stay free (GB10 pinned/device memory bypasses the OOM killer)",
        copy_bytes as f64 / (1u64 << 30) as f64,
        mem_available as f64 / (1u64 << 30) as f64,
        HEADROOM_BYTES as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

/// Proof that this process passed the memory guard for `copy_bytes` plus `overhead`. Hold it until
/// the admitted allocations are made: a second sparkforge load blocks in `admit` until then, so its
/// `MemAvailable` reading already reflects this one. Without it, two loads started together could
/// both read the same `MemAvailable` and both pass (codex-review cycle 2).
pub struct Admission {
    _locks: Vec<File>,
    copy_bytes: u64,
    overhead: u64,
}

impl Admission {
    /// Bytes of weights this admission covers.
    pub fn copy_bytes(&self) -> u64 {
        self.copy_bytes
    }

    /// Re-checks, under the held locks and right before allocating, that this admission covers the
    /// workload actually being run (its copied bytes and its overhead, which includes verification
    /// scratch when verifying), and that `MemAvailable` still leaves the headroom: CUDA startup
    /// between admission and upload uses memory too (codex-review of SHA-765).
    pub fn confirm(&self, copy_bytes: u64, overhead: u64, mem: impl Fn() -> Option<u64>) -> Result<()> {
        ensure!(
            copy_bytes <= self.copy_bytes && overhead <= self.overhead,
            "this load needs {copy_bytes} bytes copied plus {overhead} of overhead but its admission covers only {} plus {}; take admission for the binding and mode being run",
            self.copy_bytes, self.overhead
        );
        let avail = mem().context("cannot read MemAvailable from /proc/meminfo; refusing to allocate without the memory guard")?;
        check_headroom(copy_bytes + overhead, avail)
    }
}

/// Longest wait for another process's admission before giving up, so a stuck or hostile holder
/// turns into an error instead of an indefinite hang.
pub const ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// Opens an admission lock without following a final symlink and without blocking, and requires a
/// regular file: a FIFO there fails promptly. A missing per-user lock is created private to this
/// user first. This user's own locks open read-write, since an exclusive lock over NFS needs a
/// writable descriptor; the root-owned node lock opens read-only, which `flock` on tmpfs accepts.
fn open_lock(path: &Path, writable: bool, create: bool) -> Result<File> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    if create && matches!(std::fs::symlink_metadata(path), Err(e) if e.kind() == std::io::ErrorKind::NotFound) {
        let mut create = std::fs::OpenOptions::new();
        create.write(true).create_new(true);
        #[cfg(unix)]
        create.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        match create.open(path) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e).with_context(|| format!("creating the admission lock {}", path.display())),
        }
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true).write(writable);
    #[cfg(unix)]
    opts.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let lock = match opts.open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => bail!(
            "this account cannot open the admission lock {}; the node lock is shared through its group, so ask an administrator to add this user to that group",
            path.display()
        ),
        Err(e) => return Err(e).with_context(|| format!("opening the admission lock {}", path.display())),
    };
    let m = lock.metadata().with_context(|| format!("inspecting {}", path.display()))?;
    ensure!(m.file_type().is_file(), "the admission lock {} is not a regular file", path.display());
    Ok(lock)
}

/// Takes the `flock` on one lock, waiting at most `wait` for the current holder.
fn take_lock(l: &LockFile, wait: std::time::Duration) -> Result<File> {
    let path = l.path.as_path();
    let lock = open_lock(path, l.kind != LockKind::Node, l.kind == LockKind::User)?;
    let deadline = std::time::Instant::now() + wait;
    let mut told = false;
    loop {
        match lock.try_lock() {
            Ok(()) => return Ok(lock),
            Err(std::fs::TryLockError::WouldBlock) => {
                if !told {
                    eprintln!("waiting for another sparkforge load to finish allocating ({})", path.display());
                    told = true;
                }
                ensure!(std::time::Instant::now() < deadline,
                    "gave up after {} s waiting for the admission lock {}; another process has held it that long", wait.as_secs(), path.display());
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(std::fs::TryLockError::Error(e)) => return Err(e).with_context(|| format!("locking {}", path.display())),
        }
    }
}

/// Takes every lock in `locks`, in order (see `AdmissionLocks::files`), then runs the memory guard
/// against `mem()` for `copy_bytes + overhead`. The locks are advisory `flock`s, so they coordinate
/// sparkforge processes; other programs are seen only through `MemAvailable`.
pub fn admit(locks: &[LockFile], copy_bytes: u64, overhead: u64, mem: impl Fn() -> Option<u64>, wait: std::time::Duration) -> Result<Admission> {
    ensure!(!locks.is_empty(), "no admission lock to take");
    let held = locks.iter().map(|l| take_lock(l, wait)).collect::<Result<Vec<_>>>()?;
    let avail = mem().context("cannot read MemAvailable from /proc/meminfo; refusing to allocate without the memory guard")?;
    check_headroom(copy_bytes + overhead, avail)?;
    Ok(Admission { _locks: held, copy_bytes, overhead })
}

/// Root-provisioned directory holding the node-wide admission lock (a deploy step on each GB10).
/// The lock is shared through the `sparkforge` group, never with every account, so a local account
/// outside that group cannot hold it and stall other users' loads. `/run` is tmpfs, so a boot-time
/// `tmpfiles.d` entry recreates it:
/// `d /run/sparkforge 0755 root root -` and `f /run/sparkforge/admission.lock 0640 root sparkforge -`.
pub const NODE_LOCK_DIR: &str = "/run/sparkforge";
/// The group the node lock is shared through.
pub const NODE_LOCK_GROUP: &str = "sparkforge";
const NODE_LOCK_NAME: &str = "admission.lock";
const USER_LOCK_NAME: &str = "sparkforge-gpu-admission.lock";

/// Which processes an admission coordinates with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockScope {
    /// Every user's sparkforge processes on this node (the node-wide lock is provisioned).
    Node,
    /// Only this user's processes; `MemAvailable` is machine-wide, so loads by different users can
    /// still pass together.
    User,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockKind {
    /// A lock an older loader (before SHA-765) left in this user's runtime or temp directory; taken
    /// while it exists so a load by the old binary still blocks a new one during an upgrade. Never
    /// created.
    Legacy,
    /// This user's lock under the password-database home directory.
    User,
    /// The node-wide lock.
    Node,
}

/// One lock an admission takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockFile {
    pub path: PathBuf,
    pub kind: LockKind,
}

/// The locks one admission takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionLocks {
    /// Locks of older loaders that exist now, owned by this user in private directories.
    pub legacy: Vec<PathBuf>,
    /// This user's lock under the password-database home directory. It doesn't depend on a login
    /// session, so a background job and a login shell always meet here. Optional only when the
    /// node lock is available.
    pub user: Option<PathBuf>,
    /// The node-wide lock, when provisioned.
    pub node: Option<PathBuf>,
}

impl AdmissionLocks {
    pub fn scope(&self) -> LockScope {
        if self.node.is_some() { LockScope::Node } else { LockScope::User }
    }

    /// The order they are taken: legacy, user, node. Taking this user's locks first also covers
    /// turning the node lock on, or upgrading the binary, while this user's earlier load (which held
    /// only those) is still allocating.
    pub fn files(&self) -> Vec<LockFile> {
        let f = |p: &PathBuf, kind| LockFile { path: p.clone(), kind };
        self.legacy.iter().map(|p| f(p, LockKind::Legacy))
            .chain(self.user.iter().map(|p| f(p, LockKind::User)))
            .chain(self.node.iter().map(|p| f(p, LockKind::Node)))
            .collect()
    }
}

/// Where the locks are, decided from ownership only, never from environment variables.
/// - The node lock when `node_dir` is a real directory owned by `trusted_owner` (root in
///   production) and closed to writes by others, holding a regular `admission.lock` owned by
///   `trusted_owner`, in group `node_group` (`sparkforge`), granting no access to other accounts.
///   A `node_dir` that exists but fails these checks is an error, not a silent fallback.
/// - The user lock in `<home>/.local/state/sparkforge` (created 0700). `home` is consulted only
///   here, so a numeric uid without a password entry can still use a provisioned node lock.
/// - Legacy locks: `USER_LOCK_NAME` in each of `legacy_dirs` when the directory and the file are
///   this user's, the directory isn't writable by others, and the file is regular.
pub fn admission_locks_in(
    node_dir: &Path,
    trusted_owner: u32,
    node_group: impl FnOnce() -> Option<u32>,
    uid: u32,
    home: impl FnOnce() -> Option<PathBuf>,
    legacy_dirs: &[PathBuf],
) -> Result<AdmissionLocks> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt};
        let private_dir = |d: &Path| std::fs::symlink_metadata(d).is_ok_and(|m| m.file_type().is_dir() && m.uid() == uid && m.mode() & 0o022 == 0);
        let node = match std::fs::symlink_metadata(node_dir) {
            Ok(m) => {
                ensure!(m.file_type().is_dir() && m.uid() == trusted_owner && m.mode() & 0o022 == 0,
                    "{} exists but is not a directory owned by uid {trusted_owner} and closed to writes by others; fix the provisioning or remove it", node_dir.display());
                let lock = node_dir.join(NODE_LOCK_NAME);
                let lm = std::fs::symlink_metadata(&lock).with_context(|| format!("{} is provisioned without {NODE_LOCK_NAME}", node_dir.display()))?;
                ensure!(lm.file_type().is_file() && lm.uid() == trusted_owner,
                    "{} must be a regular file owned by uid {trusted_owner}", lock.display());
                ensure!(lm.mode() & 0o007 == 0,
                    "{} grants access to every account (mode {:o}), so any local account could hold it and stall every load; share it through group {NODE_LOCK_GROUP} (mode 0640)", lock.display(), lm.mode() & 0o777);
                let group = node_group().with_context(|| format!("{} is provisioned but group {NODE_LOCK_GROUP} does not exist", lock.display()))?;
                ensure!(lm.gid() == group,
                    "{} belongs to gid {}, not group {NODE_LOCK_GROUP} (gid {group}), so accounts outside that group could hold it; fix the provisioning", lock.display(), lm.gid());
                Some(lock)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).with_context(|| format!("inspecting {}", node_dir.display())),
        };
        let user = (|| -> Result<PathBuf> {
            let home = home().with_context(|| format!("no home directory for uid {uid} in the password database, and {} is not provisioned", node_dir.display()))?;
            let d = home.join(".local/state/sparkforge");
            std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&d).with_context(|| format!("creating {}", d.display()))?;
            ensure!(private_dir(&d), "{} must be a directory owned by uid {uid} and not writable by others", d.display());
            Ok(d.join(USER_LOCK_NAME))
        })();
        // With the node lock, the user lock only adds same-user cutover cover, so it is optional.
        let user = if node.is_some() { user.ok() } else { Some(user?) };
        let legacy = legacy_dirs.iter().filter(|d| private_dir(d)).map(|d| d.join(USER_LOCK_NAME))
            .filter(|p| std::fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_file() && m.uid() == uid))
            .filter(|p| user.as_ref() != Some(p))
            .collect();
        Ok(AdmissionLocks { legacy, user, node })
    }
    #[cfg(not(unix))]
    {
        let _ = (node_dir, trusted_owner, node_group, uid, legacy_dirs);
        let home = home().unwrap_or_else(std::env::temp_dir);
        Ok(AdmissionLocks { legacy: vec![], user: Some(home.join(USER_LOCK_NAME)), node: None })
    }
}

/// The production lock locations: `/run/sparkforge` owned by root in group `sparkforge`, this
/// process's effective uid, the home directory from the password database (not `$HOME`), and the
/// directories older loaders used (`/run/user/<uid>`, `<tmp>/sparkforge-<uid>`).
pub fn admission_locks() -> Result<AdmissionLocks> {
    #[cfg(unix)]
    {
        let uid = unsafe { libc::geteuid() };
        let legacy = [PathBuf::from(format!("/run/user/{uid}")), std::env::temp_dir().join(format!("sparkforge-{uid}"))];
        admission_locks_in(Path::new(NODE_LOCK_DIR), 0, || gid_of(NODE_LOCK_GROUP), uid, || home_dir_of(uid), &legacy)
    }
    #[cfg(not(unix))]
    {
        admission_locks_in(Path::new(NODE_LOCK_DIR), 0, || None, 0, || None, &[])
    }
}

#[cfg(unix)]
fn gid_of(group: &str) -> Option<u32> {
    let name = std::ffi::CString::new(group).ok()?;
    let mut buf: Vec<libc::c_char> = vec![0; 16 * 1024];
    let mut gr: libc::group = unsafe { std::mem::zeroed() };
    let mut out: *mut libc::group = std::ptr::null_mut();
    let rc = unsafe { libc::getgrnam_r(name.as_ptr(), &mut gr, buf.as_mut_ptr(), buf.len(), &mut out) };
    if rc != 0 || out.is_null() {
        return None;
    }
    Some(gr.gr_gid)
}

#[cfg(unix)]
fn home_dir_of(uid: u32) -> Option<PathBuf> {
    use std::ffi::CStr;
    let mut buf: Vec<libc::c_char> = vec![0; 16 * 1024];
    let mut pw: libc::passwd = unsafe { std::mem::zeroed() };
    let mut out: *mut libc::passwd = std::ptr::null_mut();
    let rc = unsafe { libc::getpwuid_r(uid, &mut pw, buf.as_mut_ptr(), buf.len(), &mut out) };
    if rc != 0 || out.is_null() || pw.pw_dir.is_null() {
        return None;
    }
    let dir = unsafe { CStr::from_ptr(pw.pw_dir) }.to_str().ok()?;
    Some(PathBuf::from(dir))
}

/// Bytes a load copies into GPU allocations.
pub fn bytes_to_copy(ck: &Checkpoint, b: &Binding) -> u64 {
    b.linears.iter().flat_map(|l| l.parts.iter()).map(|p| ck.tensors[p].len()).sum()
}

/// Admission for loading `b`, taken on the host before any CUDA call: the `locks` and the memory
/// guard for the copied bytes plus the load's overhead. Hold the result until the upload has made
/// and filled every allocation; the upload confirms it against what it actually copies.
pub fn admit_load_with(ck: &Checkpoint, b: &Binding, verifying: bool, locks: &[LockFile], mem: impl Fn() -> Option<u64>) -> Result<Admission> {
    admit(locks, bytes_to_copy(ck, b), load_overhead_bytes(b, verifying), mem, ADMISSION_WAIT)
}

/// `admit_load_with` at the production lock locations with `/proc/meminfo`. Warns when only the
/// per-user lock is available.
pub fn admit_load(ck: &Checkpoint, b: &Binding, verifying: bool) -> Result<Admission> {
    let locks = admission_locks()?;
    if locks.scope() == LockScope::User {
        eprintln!("note: {NODE_LOCK_DIR}/{NODE_LOCK_NAME} is not provisioned; loads are serialized only among this user's processes");
    }
    admit_load_with(ck, b, verifying, &locks.files(), mem_available)
}

/// `MemAvailable` from /proc/meminfo in bytes (Linux only).
pub fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

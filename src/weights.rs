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
    /// `<repo>/blobs` of a Hugging Face snapshot, opened from the same pinned `<repo>` descriptor as
    /// the snapshot itself. Directory pinning alone can't make the revision and the store a
    /// consistent pair (entries can be swapped and swapped back between any two checks), so every
    /// blob is also verified against its name (`verify_blob`): the revision's links, read from the
    /// opened revision, then fix the bytes, whichever store they are found in (codex-review of
    /// SHA-788). `None` for a directory that is not a snapshot. An open error is kept, not raised,
    /// and reported only when a blob link needs the store, so a checkpoint of plain files loads
    /// beside a blob store it doesn't use.
    #[cfg(unix)]
    blobs: Option<std::result::Result<File, String>>,
    /// The cache's shared, deduplicated store (`<repo>/../blobs`), when it exists. A blob may link
    /// one hop into it, as `../../blobs/<xx>/<name>`, and nowhere else.
    #[cfg(unix)]
    shared: Option<File>,
}

impl CheckpointDir {
    fn entry(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Opens `name` inside the directory without blocking (a FIFO with no writer would otherwise
    /// hang `open`) and requires the opened descriptor to be a regular file.
    ///
    /// No symlink on the way is followed, except one at the entry itself, and only when its target
    /// stays inside this directory or is a Hugging Face cache blob (see `link_target`). The target
    /// is resolved lexically, so the filesystem never walks `..`: an inside target is walked from
    /// this directory's descriptor, and a blob is opened in the blob store pinned at resolution.
    /// That walk is one component at a time with `O_NOFOLLOW`, so a target that passes through
    /// another symlink (for example a `current` alias that a publisher switches) is refused rather
    /// than followed into whatever generation the alias names now (codex-review of SHA-765).
    /// `open`, without verifying a blob: returns the Hugging Face blob name the entry links to, if
    /// any, so the caller can verify several shards in parallel.
    fn open_entry(&self, name: &str) -> Result<(File, Option<String>)> {
        let path = self.entry(name);
        #[allow(unused_mut)]
        let mut blob_name = None;
        #[cfg(unix)]
        let opened = {
            use std::io::ErrorKind;
            let (parent, last) = nofollow::walk(&self.fd, Path::new(name))
                .map_err(|e| nofollow::explain(e, &path, "passes through a symlink or a missing directory"))?;
            match nofollow::open_file(&parent, &last) {
                Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
                    let target = nofollow::readlink(&parent, &last).with_context(|| format!("reading the link {}", path.display()))?;
                    let hop = match link_target(Path::new(name), &target, self.is_hf_snapshot()) {
                        LinkTarget::Inside(rel) => nofollow::walk(&self.fd, &rel).and_then(|(p, l)| nofollow::open_file(&p, &l)),
                        LinkTarget::HfBlob(blob) => match &self.blobs {
                            Some(Ok(blobs)) => {
                                blob_name = blob.to_str().map(str::to_owned);
                                let c = nofollow::cstr(&blob)?;
                                match nofollow::open_file(blobs, &c) {
                                    // The blob may itself link one hop into the shared store
                                    // (spark-5's cache links `<repo>/blobs/<sha256>` on to
                                    // `hub/blobs/<xx>/<id>`), walked there without following any
                                    // further link. Its bytes are verified against the name.
                                    Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
                                        let t = nofollow::readlink(blobs, &c).with_context(|| format!("reading the blob link behind {}", path.display()))?;
                                        let (Some(rel), Some(shared)) = (shared_store_target(&t), &self.shared) else {
                                            bail!("{} links to blob {}, which links on to {}; a blob may link only into the cache's shared store, as ../../blobs/<xx>/<name>", path.display(), blob.to_string_lossy(), t.display());
                                        };
                                        nofollow::walk(shared, &rel).and_then(|(d, l)| nofollow::open_file(&d, &l))
                                    }
                                    r => r,
                                }
                            }
                            Some(Err(why)) => bail!("{} links into the blob store, but {why}", path.display()),
                            None => Err(std::io::Error::from(ErrorKind::NotFound)),
                        },
                        LinkTarget::Outside => bail!(
                            "{} links to {}, outside the checkpoint directory; a directory there can be replaced by another generation mid-load, so only links that stay inside the checkpoint, or a Hugging Face snapshot's links to ../../blobs/<hash>, are followed",
                            path.display(), target.display()
                        ),
                    };
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
        Ok((file, blob_name))
    }

    /// Whether this is `<repo>/snapshots/<revision>` of a Hugging Face cache.
    #[cfg(unix)]
    fn is_hf_snapshot(&self) -> bool {
        self.path.parent().and_then(Path::file_name).is_some_and(|n| n == "snapshots")
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
    /// A Hugging Face blob is verified against its name over exactly the bytes returned.
    pub fn read_metadata(&self, name: &str) -> Result<String> {
        let path = self.entry(name);
        let (file, blob) = self.open_entry(name)?;
        let len = file.metadata()?.len();
        ensure!(len <= MAX_METADATA_BYTES, "{} is {len} bytes, above the {MAX_METADATA_BYTES}-byte metadata limit", path.display());
        let text = read_capped(file, MAX_METADATA_BYTES, &path.display().to_string())?;
        if let Some(blob) = blob {
            let got = blob_digest(&blob, text.len() as u64, &path, |update| {
                update(text.as_bytes());
                Ok(())
            })?;
            check_blob_digest(&got, &blob, &path)?;
        }
        Ok(text)
    }
}

/// Checks that a Hugging Face blob's bytes match its name: SHA256 of the content for a 64-digit
/// name (LFS files), or the Git object SHA1 (`blob <len>\0` then the content) for a 40-digit one.
/// Blob names are what make a snapshot's links mean fixed bytes; this makes that hold even if the
/// blob store was replaced or rearranged around the load. The file's stamp must not change while
/// it is hashed, and later reads are checked against that same stamp.
fn verify_blob(file: &File, name: &str, path: &Path) -> Result<FileStamp> {
    let stamp = FileStamp::of(file).with_context(|| format!("inspecting {}", path.display()))?;
    let mut buf = vec![0u8; REF_CHUNK_BYTES];
    let got = blob_digest(name, stamp.len, path, |update| {
        let mut at = 0u64;
        while at < stamp.len {
            let n = (stamp.len - at).min(buf.len() as u64) as usize;
            pread(file, &mut buf[..n], at).with_context(|| format!("reading {} to verify it", path.display()))?;
            update(&buf[..n]);
            at += n as u64;
        }
        Ok(())
    })?;
    check_blob_digest(&got, name, path)?;
    ensure!(FileStamp::of(file)? == stamp, "{} changed while it was being verified", path.display());
    Ok(stamp)
}

/// The digest a blob named `name` must have, over `len` bytes that `feed` passes to its callback.
fn blob_digest(name: &str, len: u64, path: &Path, feed: impl FnOnce(&mut dyn FnMut(&[u8])) -> Result<()>) -> Result<String> {
    use sha2::Digest;
    Ok(match name.len() {
        64 => {
            let mut h = sha2::Sha256::new();
            feed(&mut |b| h.update(b))?;
            hex(&h.finalize())
        }
        40 => {
            let mut h = sha1::Sha1::new();
            h.update(format!("blob {len}\0").as_bytes());
            feed(&mut |b| h.update(b))?;
            hex(&h.finalize())
        }
        n => bail!("{} links to a blob name of {n} characters, not a SHA1 or SHA256", path.display()),
    })
}

fn check_blob_digest(got: &str, name: &str, path: &Path) -> Result<()> {
    ensure!(got.eq_ignore_ascii_case(name),
        "{} links to blob {name}, but that blob's bytes do not hash to its name: the blob store holds other bytes under that name (replaced, rearranged or corrupt), so they are not used", path.display());
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `<xx>/<name>` when a blob's own link is exactly `../../blobs/<xx>/<name>`: a two-hex-digit
/// shard directory of the shared store, then a plain name.
#[cfg(unix)]
fn shared_store_target(t: &Path) -> Option<PathBuf> {
    let c: Vec<Component> = t.components().collect();
    match c.as_slice() {
        [Component::ParentDir, Component::ParentDir, Component::Normal(b), Component::Normal(xx), Component::Normal(name)]
            if *b == "blobs" && xx.len() == 2 && xx.to_str().is_some_and(|x| x.bytes().all(|b| b.is_ascii_hexdigit())) && !name.is_empty() =>
        {
            Some(Path::new(xx).join(name))
        }
        _ => None,
    }
}

/// Where a shard link's target lands, judged lexically from the link's own directory.
#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
enum LinkTarget {
    /// Never leaves the checkpoint directory; the path relative to it, with no `..` left. Walking
    /// that from the pinned descriptor, rather than the link's text, means a subdirectory renamed
    /// mid-walk cannot carry a `..` outside (codex-review cycle 1 on SHA-787).
    Inside(PathBuf),
    /// `../../blobs/<sha1 or sha256>` from a Hugging Face snapshot: the blob's name, opened in the
    /// blob store pinned at resolution. Blobs are named by content hash, so within that store the
    /// name keeps meaning the same bytes.
    HfBlob(std::ffi::OsString),
    /// Anything else, including an absolute target or one that climbs out and back in: a real
    /// directory on that path can be replaced between the config read and the shard open
    /// (codex-review round 3 on #13).
    Outside,
}

#[cfg(unix)]
fn link_target(entry: &Path, target: &Path, hf_snapshot: bool) -> LinkTarget {
    let mut depth: Vec<&std::ffi::OsStr> = entry.parent().into_iter().flat_map(Path::components).filter_map(|c| match c {
        Component::Normal(n) => Some(n),
        _ => None,
    }).collect();
    let mut above = 0usize; // levels above the checkpoint directory
    let mut tail: Vec<&std::ffi::OsStr> = vec![];
    for c in target.components() {
        match c {
            Component::CurDir => {}
            Component::Normal(n) if above == 0 => depth.push(n),
            Component::Normal(n) => tail.push(n),
            Component::ParentDir if above == 0 && !depth.is_empty() => {
                depth.pop();
            }
            Component::ParentDir if tail.is_empty() => above += 1,
            // `..` after leaving and re-entering: outside either way.
            Component::ParentDir => return LinkTarget::Outside,
            Component::RootDir | Component::Prefix(_) => return LinkTarget::Outside,
        }
    }
    let is_hash = |n: &std::ffi::OsStr| n.to_str().is_some_and(|h| matches!(h.len(), 40 | 64) && h.bytes().all(|b| b.is_ascii_hexdigit()));
    match (above, depth.is_empty(), tail.as_slice()) {
        (0, false, _) => LinkTarget::Inside(depth.iter().collect()),
        (2, true, [blobs, hash]) if hf_snapshot && *blobs == "blobs" && is_hash(hash) => LinkTarget::HfBlob(hash.to_os_string()),
        _ => LinkTarget::Outside,
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

    /// Flags for a directory used only as a base for `openat`. On Linux, `O_PATH` needs only search
    /// permission, so a mode 0710 or 0711 directory works as it does for a path-based open
    /// (codex-review round 3 on #13). Elsewhere, `O_RDONLY` (which also needs read permission).
    /// `O_DIRECTORY` is what keeps `O_NOFOLLOW` strict here: `O_PATH | O_NOFOLLOW` alone would open a
    /// symlink itself, while with `O_DIRECTORY` a symlink fails with `ENOTDIR`.
    #[cfg(target_os = "linux")]
    pub const DIR_FLAGS: i32 = libc::O_PATH | libc::O_DIRECTORY;
    #[cfg(not(target_os = "linux"))]
    pub const DIR_FLAGS: i32 = libc::O_RDONLY | libc::O_DIRECTORY;

    pub fn cstr(s: &std::ffi::OsStr) -> io::Result<CString> {
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
            cur = openat(&cur, &name, DIR_FLAGS)?;
        }
        Ok((cur, cstr(last)?))
    }

    /// Opens the directory at `path` from `start` with no symlink anywhere on the way.
    pub fn open_dir(start: &File, path: &Path) -> io::Result<File> {
        let (parent, last) = walk(start, path)?;
        openat(&parent, &last, DIR_FLAGS)
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
        let open = |p: &Path| std::fs::OpenOptions::new().read(true).custom_flags(nofollow::DIR_FLAGS | libc::O_NOFOLLOW).open(p);
        let snapshot = path.parent().filter(|s| s.file_name().is_some_and(|n| n == "snapshots"));
        let (fd, blobs, shared) = match (snapshot.and_then(Path::parent), path.file_name()) {
            // A Hugging Face snapshot: pin `<repo>` and open `snapshots/<rev>` and `blobs` downward
            // from it. Which bytes a blob link may yield is settled by `verify_blob`, not by this.
            (Some(repo), Some(rev)) => {
                let repo_fd = open(repo).with_context(|| format!("opening the cache repository {}", repo.display()))?;
                let fd = nofollow::open_dir(&repo_fd, &Path::new("snapshots").join(rev))
                    .map_err(|e| nofollow::explain(e, &path, "is not a real directory under its repository"))?;
                let blobs = nofollow::open_dir(&repo_fd, Path::new("blobs")).map_err(|e| match e.kind() {
                    std::io::ErrorKind::NotFound => format!("{} is missing", repo.join("blobs").display()),
                    _ => format!("{} cannot be opened as a real directory ({e})", repo.join("blobs").display()),
                });
                let shared = nofollow::open_dir(&repo_fd, Path::new("../blobs")).ok();
                (fd, Some(blobs), shared)
            }
            _ => (open(&path).with_context(|| format!("opening checkpoint directory {}", path.display()))?, None, None),
        };
        Ok(CheckpointDir { path, fd, blobs, shared })
    }
    #[cfg(not(unix))]
    Ok(CheckpointDir { path })
}

/// `config.json` of an opened checkpoint directory, read with the metadata open rules.
pub fn read_config(dir: &CheckpointDir) -> Result<String> {
    dir.read_metadata("config.json")
}

/// The spec of the checkpoint opened at `dir`, for planning with a checkpoint: parsed from its own
/// `config.json` (with `fmt` as an override) and resolved onto its tensor names, both read through
/// the one opened directory, so a plan never pairs one generation's format map with another's
/// modules (codex-review of #17). A caller holding a config file of its own passes its text as
/// `given`, which must equal the checkpoint's.
pub fn spec_from_checkpoint(dir: &CheckpointDir, fmt: Option<WeightFormat>, given: Option<&str>) -> Result<ModelSpec> {
    let text = read_config(dir)?;
    if let Some(given) = given {
        let ours: serde_json::Value = serde_json::from_str(given).context("parsing the config")?;
        let theirs: serde_json::Value = serde_json::from_str(&text).context("parsing the checkpoint's config.json")?;
        ensure!(ours == theirs, "the config differs from {}'s own config.json; plan with that file, or with the checkpoint it belongs to", dir.path.display());
    }
    let mut spec = ModelSpec::from_json(&text, fmt)?;
    let names = tensor_names(dir)?;
    spec.resolve_modules(names.iter().map(String::as_str))?;
    Ok(spec)
}

/// The tensor names of a checkpoint, for resolving its module identities: the keys of
/// `model.safetensors.index.json` when present (no shard is read), else the header of
/// `model.safetensors`. `Checkpoint::open_dir` later checks that the index and the headers agree.
pub fn tensor_names(dir: &CheckpointDir) -> Result<Vec<String>> {
    const INDEX: &str = "model.safetensors.index.json";
    if dir.has_entry(INDEX)? {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: UniqueMap<String>,
        }
        let idx: Index = serde_json::from_str(&dir.read_metadata(INDEX)?).with_context(|| format!("parsing {}", dir.entry(INDEX).display()))?;
        // The same shard-name rule the loader applies, so planning never accepts an index the
        // loader refuses (no shard is opened here).
        for (_, shard) in &idx.weight_map.0 {
            check_shard_name(shard)?;
        }
        return Ok(idx.weight_map.0.into_iter().map(|(k, _)| k).collect());
    }
    let ck = Checkpoint::open_dir(dir)?;
    Ok(ck.tensors.into_keys().collect())
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
    // Other targets (for example wasm32) have no positioned reads; this keeps the non-Unix code
    // paths compiling, which `cargo check --target wasm32-unknown-unknown` verifies.
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, at);
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "positioned reads are not supported on this target"))
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
        Self::open_dir_with(dir, std::thread::available_parallelism().map_or(4, |n| n.get()))
    }

    /// `open_dir` verifying Hugging Face blobs with at most `workers` threads; 0 verifies them all
    /// on the calling thread, as happens for any blob whose worker the OS refuses to create.
    #[doc(hidden)]
    pub fn open_dir_with(dir: &CheckpointDir, workers: usize) -> Result<Self> {
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
        let mut opened = Vec::with_capacity(shard_files.len());
        for f in &shard_files {
            let (file, blob) = dir.open_entry(f).context("opening a shard")?;
            opened.push((f, dir.entry(f), file, blob));
        }
        // Hugging Face blobs are verified against their names in parallel: hashing runs at about
        // 2.2 GB/s per thread on GB10 (SHA2 instructions), so one thread per shard keeps it to
        // roughly the time of the largest shard.
        let mut verified: Vec<Option<FileStamp>> = Vec::with_capacity(opened.len());
        for group in opened.chunks(workers.max(1)) {
            // A worker the OS refuses to create (for example in a container at its task limit) is
            // not an error: that blob is verified on this thread instead (codex-review of #15).
            let stamps = std::thread::scope(|sc| {
                enum Job<'s> {
                    None,
                    Spawned(std::thread::ScopedJoinHandle<'s, Result<FileStamp>>),
                    Here,
                }
                let jobs: Vec<Job> = group.iter().map(|(_, path, file, blob)| match blob {
                    None => Job::None,
                    Some(_) if workers == 0 => Job::Here,
                    Some(b) => match std::thread::Builder::new().spawn_scoped(sc, move || verify_blob(file, b, path)) {
                        Ok(h) => Job::Spawned(h),
                        Err(_) => Job::Here,
                    },
                }).collect();
                jobs.into_iter().zip(group).map(|(j, (_, path, file, blob))| match j {
                    Job::None => Ok(None),
                    Job::Spawned(h) => h.join().map_err(|_| anyhow::anyhow!("a blob verification thread panicked"))?.map(Some),
                    Job::Here => verify_blob(file, blob.as_deref().expect("a blob job has a name"), path).map(Some),
                }).collect::<Result<Vec<_>>>()
            })?;
            verified.extend(stamps);
        }
        for ((f, path, file, _), verified) in opened.into_iter().zip(verified) {
            let stamp = FileStamp::of(&file)?;
            // A verified blob must still be the version that was hashed.
            ensure!(verified.as_ref().is_none_or(|v| *v == stamp), "{} changed after it was verified against its blob name", path.display());
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
    if m.hybrid.is_some() {
        bail!("{} checkpoints are checked tensor by tensor (`sparkforge weights`) but do not load into GPU memory yet", m.arch);
    }
    if !m.formats.is_empty() || m.declared_mixed {
        bail!("{} checkpoints with mixed-precision weights do not load yet; only the hybrid layout reads per-module formats", m.arch);
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

/// A hybrid checkpoint checked tensor by tensor against the plan.
#[derive(Debug)]
pub struct HybridCheck {
    /// Tensors checked against the plan, and their bytes.
    pub tensors: usize,
    pub bytes: u64,
    /// Bytes per inventory group (`routed_experts`, `ple_table`, ...), largest first.
    pub groups: Vec<(&'static str, u64)>,
    /// Tensors of towers the text model doesn't load (the vision tower), and their bytes.
    pub skipped: usize,
    pub skipped_bytes: u64,
    /// Rows in each PLE table shard, derived from the checkpoint's per-head table sizes.
    pub table_rows: usize,
}

/// Canonical prefix of the towers a text-only load skips.
const SKIPPED_TOWERS: [&str; 1] = ["visual."];

/// Checks every tensor of a hybrid checkpoint against the plan for `m`, which must be resolved
/// onto this checkpoint (`ModelSpec::resolve_modules`): each tensor the plan needs exists once,
/// with the dtype and shape of the format it is priced at, and every other tensor belongs to a
/// skipped tower. Any problem fails, listing the tensors and the expected and found values.
pub fn check_hybrid(ck: &Checkpoint, m: &ModelSpec) -> Result<HybridCheck> {
    let Some(h) = &m.hybrid else { bail!("{} is not a hybrid model", m.arch) };
    let Some(r) = &m.resolved else { bail!("the spec must be resolved onto this checkpoint's modules before its tensors are checked") };
    // Canonical name -> tensor, refusing two spellings of one tensor (`model.layers.0...` beside
    // `model.language_model.layers.0...`): only one of them could be loaded.
    let mut by_name: HashMap<&str, &TensorInfo> = HashMap::new();
    let (mut skipped, mut skipped_bytes) = (0, 0);
    for t in ck.tensors.values() {
        let c = crate::modules::canonical(&t.name);
        if SKIPPED_TOWERS.iter().any(|p| c.starts_with(p)) {
            skipped += 1;
            skipped_bytes += t.len();
            continue;
        }
        if let Some(other) = by_name.insert(c, t) {
            let (a, b) = if other.name < t.name { (&other.name, &t.name) } else { (&t.name, &other.name) };
            bail!("the checkpoint holds {a} and {b}, two names for the tensor {c}; only one can be loaded");
        }
    }
    let ple = crate::inventory::ple_paths(&r.table, h)?;
    let table_rows = ple_table_rows(ck, &by_name, &ple.embedding, &h.ple)?;
    let f = crate::plan::Formats::new(m);
    let wants = crate::inventory::hybrid(m, &f, &crate::inventory::Layout { ple, table_rows: Some(table_rows) })?;
    let mut problems = vec![];
    let mut groups: Vec<(&'static str, u64)> = vec![];
    let mut bytes = 0;
    for w in &wants {
        match by_name.remove(w.name.as_str()) {
            None => problems.push(format!("missing {} (expected {} {:?})", w.name, w.dtype, w.shape)),
            Some(t) if t.dtype != w.dtype || t.shape != w.shape => {
                problems.push(format!("{}: expected {} {:?}, found {} {:?}", t.name, w.dtype, w.shape, t.dtype, t.shape))
            }
            Some(t) => {
                bytes += t.len();
                match groups.iter_mut().find(|(g, _)| *g == w.group) {
                    Some(g) => g.1 += t.len(),
                    None => groups.push((w.group, t.len())),
                }
            }
        }
    }
    let mut unexpected: Vec<&str> = by_name.values().map(|t| t.name.as_str()).collect();
    unexpected.sort_unstable();
    problems.extend(unexpected.iter().map(|n| format!("unexpected {n}: the plan has no use for it, and it is not in a skipped tower")));
    if !problems.is_empty() {
        let shown = problems.len().min(20);
        bail!("checkpoint {} does not match the plan ({} problem(s)):\n  {}{}", ck.dir.display(), problems.len(), problems[..shown].join("\n  "),
            if problems.len() > shown { format!("\n  ... and {} more", problems.len() - shown) } else { String::new() });
    }
    groups.sort_by_key(|g| std::cmp::Reverse(g.1));
    Ok(HybridCheck { tensors: wants.len(), bytes, groups, skipped, skipped_bytes, table_rows })
}

/// Prices `m`'s PLE table at the rows `c` found in the checkpoint, so a plan built afterwards
/// budgets the table as stored rather than from `ngram_vocab_size_base`.
pub fn apply_checked_table(m: &mut ModelSpec, c: &HybridCheck) {
    if let Some(h) = &mut m.hybrid {
        h.ple.checked_rows = h.ple.split_parts.map(|parts| parts as u64 * c.table_rows as u64);
    }
}

/// Rows in each shard of the PLE n-gram table. The per-head tables are prime-sized near
/// `ngram_vocab_size_base`, so their sizes come from the checkpoint (`ngram_heads_vocab_sizes`,
/// with `ngram_heads_offsets` their running sum); the total is padded to a multiple of
/// `make_ngram_vocab_size_divisible_by` and split evenly into `split_ngram_parts` shards.
fn ple_table_rows(ck: &Checkpoint, by_name: &HashMap<&str, &TensorInfo>, embedding: &str, p: &crate::config::Ple) -> Result<usize> {
    let heads = p.heads() as usize;
    let read = |rel: &str| -> Result<Vec<i64>> {
        let name = format!("{embedding}.{rel}");
        let t = by_name.get(name.as_str()).with_context(|| format!("missing {name} (expected I64 [{heads}]), which sizes the PLE n-gram table"))?;
        ensure!(t.dtype == "I64" && t.shape == [heads], "{}: expected I64 [{heads}], found {} {:?}", t.name, t.dtype, t.shape);
        let mut buf = vec![];
        ck.read_into(t, &mut buf)?;
        Ok(buf.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let (sizes, offsets) = (read("ngram_heads_vocab_sizes")?, read("ngram_heads_offsets")?);
    let mut total: u64 = 0;
    for (i, (&s, &o)) in sizes.iter().zip(&offsets).enumerate() {
        ensure!(u64::try_from(s).is_ok_and(|s| s >= p.vocab_base), "PLE head {i} has {s} rows, below ngram_vocab_size_base {}", p.vocab_base);
        ensure!(u64::try_from(o) == Ok(total), "PLE head {i} starts at row {o}, but the heads before it hold {total} rows");
        total = total.checked_add(s as u64).context("the PLE n-gram table's per-head sizes overflow a 64-bit row count")?;
    }
    let multiple = p.vocab_multiple.unwrap_or(1) as u64;
    let parts = p.split_parts.context("config.json has no split_ngram_parts, so the PLE n-gram table's shards can't be checked")? as u64;
    let padded = total.div_ceil(multiple).checked_mul(multiple)
        .with_context(|| format!("the PLE n-gram table's {total} rows, padded to a multiple of {multiple}, overflow a 64-bit row count"))?;
    ensure!(padded % parts == 0, "the PLE n-gram table has {padded} rows (padded to a multiple of {multiple}), which do not split evenly into {parts} shards");
    Ok(usize::try_from(padded / parts)?)
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
    _lock: File,
    copy_bytes: u64,
    overhead: u64,
}

impl Admission {
    /// Bytes of weights this admission covers.
    pub fn copy_bytes(&self) -> u64 {
        self.copy_bytes
    }

    /// Re-checks, under the held lock and right before allocating, that this admission covers the
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

/// Opens this user's admission lock without following a final symlink and without blocking, and
/// requires a regular file: a FIFO there fails promptly. A missing lock is created private to this
/// user first. It opens read-write, since an exclusive lock over NFS needs a writable descriptor.
fn open_lock(path: &Path) -> Result<File> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    if matches!(std::fs::symlink_metadata(path), Err(e) if e.kind() == std::io::ErrorKind::NotFound) {
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
    opts.read(true).write(true);
    #[cfg(unix)]
    opts.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let lock = match opts.open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => bail!("this account cannot open the admission lock {} read-write", path.display()),
        Err(e) => return Err(e).with_context(|| format!("opening the admission lock {}", path.display())),
    };
    let m = lock.metadata().with_context(|| format!("inspecting {}", path.display()))?;
    ensure!(m.file_type().is_file(), "the admission lock {} is not a regular file", path.display());
    Ok(lock)
}

/// Takes the admission lock at `lock` (see `admission_lock`), waiting at most `wait` for another
/// load to finish allocating, then runs the memory guard against `mem()` for
/// `copy_bytes + overhead`. The lock is an advisory `flock`, so it coordinates sparkforge
/// processes; other programs are seen only through `MemAvailable`.
///
/// - Once the lock has been found held, an acquisition after the deadline is a timeout too, so a
///   holder that outlasts `wait` never leads to an admission.
/// - After the lock is taken, the path must still name the locked file. If the file was replaced
///   before this load got it, the load locks the new file instead, within the same deadline
///   (codex-review of #16).
/// - Deleting the lock file, or `~/.local/state/sparkforge`, while a load is admitted is not
///   supported: no lock survives its own deletion, so a load started afterwards locks a new file
///   and isn't coordinated with the admitted one. Nothing on the Sparks deletes that directory;
///   don't remove it while sparkforge runs.
pub fn admit(lock: &Path, copy_bytes: u64, overhead: u64, mem: impl Fn() -> Option<u64>, wait: std::time::Duration) -> Result<Admission> {
    admit_with(lock, copy_bytes, overhead, mem, wait, &AdmitHooks::default())
}

/// Test seams for `admit_with`: `on_wait` runs the first time each opened lock file is found held
/// (it proves which file a waiting load opened), and `sleep` replaces the pause between attempts
/// (so a test can release the lock at an exact point).
#[doc(hidden)]
pub struct AdmitHooks<'a> {
    pub on_wait: &'a dyn Fn(),
    pub sleep: &'a dyn Fn(std::time::Duration),
}

impl Default for AdmitHooks<'_> {
    fn default() -> Self {
        AdmitHooks { on_wait: &|| {}, sleep: &std::thread::sleep }
    }
}

/// `admit` with test seams.
#[doc(hidden)]
pub fn admit_with(lock: &Path, copy_bytes: u64, overhead: u64, mem: impl Fn() -> Option<u64>, wait: std::time::Duration, hooks: &AdmitHooks) -> Result<Admission> {
    let deadline = std::time::Instant::now() + wait;
    let timeout = || anyhow::anyhow!("gave up after {} s waiting for the admission lock {}; another process has held it that long", wait.as_secs(), lock.display());
    let mut contended = false;
    let file = loop {
        let file = open_lock(lock)?;
        let mut told = false;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) => {
                    if !told {
                        if !contended {
                            eprintln!("waiting for another sparkforge load to finish allocating ({})", lock.display());
                        }
                        (hooks.on_wait)();
                        told = true;
                        contended = true;
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err(timeout());
                    }
                    (hooks.sleep)(std::time::Duration::from_millis(20));
                }
                Err(std::fs::TryLockError::Error(e)) => return Err(e).with_context(|| format!("locking {}", lock.display())),
            }
        }
        if contended && std::time::Instant::now() >= deadline {
            return Err(timeout());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let held = file.metadata().with_context(|| format!("inspecting {}", lock.display()))?;
            let now = std::fs::symlink_metadata(lock).ok().map(|m| (m.dev(), m.ino()));
            if now != Some((held.dev(), held.ino())) {
                eprintln!("{} was replaced while this load waited; locking the new file", lock.display());
                contended = true;
                continue;
            }
        }
        break file;
    };
    let avail = mem().context("cannot read MemAvailable from /proc/meminfo; refusing to allocate without the memory guard")?;
    check_headroom(copy_bytes + overhead, avail)?;
    Ok(Admission { _lock: file, copy_bytes, overhead })
}

const USER_LOCK_NAME: &str = "sparkforge-gpu-admission.lock";

/// This user's admission lock: `<home>/.local/state/sparkforge/sparkforge-gpu-admission.lock`, with
/// the directory created 0700 and required to be this user's and not writable by others. `home`
/// comes from the password database, never from environment variables, so a login shell and a
/// background job always meet at the same lock.
///
/// The lock coordinates one account's loads. Every Spark runs sparkforge under the single account
/// `spark`, so that is every load on the node. A node-wide lock shared between accounts was built
/// in #13 to #15 and removed in SHA-789, since no Spark has a second account. A second account
/// loading weights on the same GB10 at the same time would need it back.
pub fn admission_lock_in(uid: u32, home: impl FnOnce() -> Option<PathBuf>) -> Result<PathBuf> {
    let home = home().with_context(|| format!("no home directory for uid {uid} in the password database"))?;
    let d = home.join(".local/state/sparkforge");
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt};
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&d).with_context(|| format!("creating {}", d.display()))?;
        let private = std::fs::symlink_metadata(&d).is_ok_and(|m| m.file_type().is_dir() && m.uid() == uid && m.mode() & 0o022 == 0);
        ensure!(private, "{} must be a directory owned by uid {uid} and not writable by others", d.display());
    }
    #[cfg(not(unix))]
    {
        let _ = uid;
        std::fs::create_dir_all(&d).with_context(|| format!("creating {}", d.display()))?;
    }
    Ok(d.join(USER_LOCK_NAME))
}

/// `admission_lock_in` for this process's effective uid and its password-database home.
pub fn admission_lock() -> Result<PathBuf> {
    #[cfg(unix)]
    {
        let uid = unsafe { libc::geteuid() };
        admission_lock_in(uid, || home_dir_of(uid))
    }
    #[cfg(not(unix))]
    {
        admission_lock_in(0, || std::env::var_os("USERPROFILE").map(PathBuf::from))
    }
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

/// Admission for loading `b`, taken on the host before any CUDA call: the lock at `lock` and the
/// memory guard for the copied bytes plus the load's overhead. Hold the result until the upload has made
/// and filled every allocation; the upload confirms it against what it actually copies.
pub fn admit_load_with(ck: &Checkpoint, b: &Binding, verifying: bool, lock: &Path, mem: impl Fn() -> Option<u64>) -> Result<Admission> {
    admit(lock, bytes_to_copy(ck, b), load_overhead_bytes(b, verifying), mem, ADMISSION_WAIT)
}

/// `admit_load_with` at this user's admission lock with `/proc/meminfo`.
pub fn admit_load(ck: &Checkpoint, b: &Binding, verifying: bool) -> Result<Admission> {
    admit_load_with(ck, b, verifying, &admission_lock()?, mem_available)
}

/// `MemAvailable` from /proc/meminfo in bytes (Linux only).
pub fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

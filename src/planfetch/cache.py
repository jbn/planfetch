# cache.py
"""S3 cache (NVMe/local disk) management for prefetching."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import boto3
from botocore.config import Config as BotoConfig
from concurrent.futures import Future, ThreadPoolExecutor
from filelock import FileLock

# Try to import Rust extension for high-performance downloads
try:
    from .planfetch_rs import RustS3Cache  # type: ignore[import-not-found]

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def has_rust_backend() -> bool:
    """Check if the Rust S3 backend is available."""
    return HAS_RUST


def assert_rust_backend() -> None:
    """
    Assert that the Rust S3 backend is available.

    Raises RuntimeError if the Rust extension is not compiled/installed.
    Use this at startup to fail fast if you require the high-performance
    Rust backend and don't want to fall back to the slower Python implementation.

    Example:
        import planfetch as pf
        pf.assert_rust_backend()  # Fail early if Rust not available
    """
    if not HAS_RUST:
        raise RuntimeError(
            "Rust S3 backend (planfetch_rs) is not available. "
            "The Python backend uses ThreadPoolExecutor which may be too slow "
            "for production workloads. Build with `uv build` or install a wheel "
            "with the Rust extension included."
        )


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")
    rest = uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key


STAT_KEYS = (
    "attempts",
    "retries",
    "throttled",
    "failures",
    "downloaded",
    "bytes",
)

_INTERNAL_DIRS = frozenset({"_tmp", "_locks"})

# Attributes that hold per-process state (thread pool, tokio runtime, locks).
# They are rebuilt lazily in whichever process first uses the cache after a
# fork or an unpickle, so one SharedDiskCache can be shared with DataLoader
# workers regardless of the start method.
_RUNTIME_ATTRS = (
    "_rust_cache",
    "s3",
    "_pool",
    "_inflight",
    "_inflight_lock",
    "_meta",
    "_meta_lock",
    "_total_bytes",
    "_py_stats",
    "_py_stats_lock",
)


@dataclass
class CacheMeta:
    path: str
    size: int
    atime: float  # last access time (best-effort)


class SharedDiskCache:
    """
    Shared NVMe (or local disk) cache for S3 objects.

    Properties:
    - Atomic downloads: tmp -> rename (no partial files exposed).
    - Dedupe across processes: per-object file lock.
    - Concurrency: ThreadPoolExecutor wrapping boto3 (sync), or the Rust backend.
    - Pruning:
        drop(uris): unlink specific objects (what PrefetchCoordinator uses per step)
        mode="window": keep exactly a keep-set (full sweep; use rarely)
        mode="pressure": evict only under max_bytes pressure (LRU-ish)

    Multi-worker note:
    - The cache is fork- and pickle-safe: the download backend (thread pool and
      boto3 client, or the Rust tokio runtime) is created per process on first
      use, so DataLoader workers that inherit or unpickle the cache get their
      own. A user-supplied `s3_client` is reused as-is and must itself be
      fork-safe or picklable.
    - In-memory metadata is per process. For a global sweep across processes,
      use prune_backend="scan" (filesystem scan).
    """

    def __init__(
        self,
        cache_dir: Union[str, os.PathLike],
        *,
        max_bytes: int,
        download_threads: int = 64,
        s3_max_pool_connections: int = 256,
        min_age_seconds_before_delete: float = 2.0,
        s3_client=None,
        verbose: bool = False,
        use_rust: bool = True,
        retry_backoff_seconds: float = 0.25,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "_tmp").mkdir(exist_ok=True)
        (self.cache_dir / "_locks").mkdir(exist_ok=True)

        self.max_bytes = int(max_bytes)
        self.min_age_seconds_before_delete = float(min_age_seconds_before_delete)
        self.verbose = bool(verbose)
        self.retry_backoff_seconds = float(retry_backoff_seconds)

        self._download_threads = int(download_threads)
        self._s3_max_pool_connections = int(s3_max_pool_connections)
        self._user_s3_client = s3_client

        # Use Rust backend if available and requested, but not if a custom s3_client is provided
        # (Rust uses its own AWS SDK client)
        self._use_rust = bool(use_rust) and HAS_RUST and s3_client is None

        self._pid: Optional[int] = None
        self._init_backend()

    # ------------------------------------------------------------------ #
    # Per-process backend lifecycle
    # ------------------------------------------------------------------ #

    def _init_backend(self) -> None:
        """Create the download backend and bookkeeping for the current process."""
        self._pid = os.getpid()
        if self._use_rust:
            self._rust_cache = RustS3Cache(str(self.cache_dir), self._download_threads)
        else:
            s3_client = self._user_s3_client
            if s3_client is None:
                cfg = BotoConfig(max_pool_connections=self._s3_max_pool_connections)
                s3_client = boto3.client("s3", config=cfg)
            self.s3 = s3_client

            self._pool = ThreadPoolExecutor(max_workers=self._download_threads)
            self._inflight: Dict[str, Future] = {}
            self._inflight_lock = threading.Lock()

        self._meta: Dict[str, CacheMeta] = {}
        self._meta_lock = threading.Lock()
        self._total_bytes = 0
        # Python-backend download counters (the Rust backend keeps its own).
        self._py_stats: Dict[str, int] = {k: 0 for k in STAT_KEYS}
        self._py_stats_lock = threading.Lock()

    def _ensure_backend(self) -> None:
        """Rebuild the backend if this process did not create it (fork / unpickle)."""
        if self._pid != os.getpid():
            self._init_backend()

    def __getstate__(self) -> Dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k not in _RUNTIME_ATTRS}
        state["_pid"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pid = None  # backend is built on first use

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SharedDiskCache] {msg}")

    def _bump(self, key: str, n: int = 1) -> None:
        with self._py_stats_lock:
            self._py_stats[key] += n

    def stats(self) -> Dict[str, int]:
        """
        Download counters for this process's backend.

        Keys: attempts, retries, throttled, failures, downloaded, bytes, inflight.
        A non-zero `throttled` or `retries` means S3 is pushing back (503 SlowDown)
        and what looks like slow S3 is really retry backoff.
        """
        self._ensure_backend()
        if self._use_rust:
            return dict(self._rust_cache.stats())
        with self._py_stats_lock:
            out = dict(self._py_stats)
        with self._inflight_lock:
            out["inflight"] = len(self._inflight)
        return out

    def _submit_if_absent(self, key: str, s3_uri: str) -> Future:
        """
        Return the in-flight download for `key`, submitting one if none exists.

        The done-callback that removes the entry is registered *after* the
        in-flight lock is released: Future.add_done_callback runs the callback
        inline when the future has already completed (a fast or already-cached
        download), and the callback needs the same lock. Registering it under
        the lock deadlocked the calling thread.
        """
        with self._inflight_lock:
            fut = self._inflight.get(key)
            if fut is not None:
                return fut
            fut = self._pool.submit(self._download_if_needed, s3_uri)
            self._inflight[key] = fut
        self._track_future(key, fut)
        return fut

    def _track_future(self, key: str, fut: Future) -> None:
        """Ensure inflight entries get removed when the Future completes."""

        def _cleanup(_f: Future) -> None:
            with self._inflight_lock:
                if self._inflight.get(key) is _f:
                    self._inflight.pop(key, None)

        fut.add_done_callback(_cleanup)

    def _path_for(self, s3_uri: str) -> Path:
        """Where `s3_uri` lives on disk. Pure: creates nothing."""
        h = _sha256_hex(s3_uri)
        return self.cache_dir / h[:2] / h[2:4] / h

    def path_for(self, s3_uri: str) -> str:
        """Local path for an S3 URI (does not download or create directories)."""
        return str(self._path_for(s3_uri))

    def _lock_for(self, s3_uri: str) -> FileLock:
        return FileLock(
            str(self.cache_dir / "_locks" / (_sha256_hex(s3_uri) + ".lock"))
        )

    def _touch(self, s3_uri: str, path: Path) -> None:
        h = _sha256_hex(s3_uri)
        try:
            st = path.stat()
        except FileNotFoundError:
            return
        now = time.time()
        with self._meta_lock:
            prev = self._meta.get(h)
            if prev is None:
                self._meta[h] = CacheMeta(path=str(path), size=st.st_size, atime=now)
                self._total_bytes += st.st_size
            else:
                if prev.size != st.st_size:
                    self._total_bytes += st.st_size - prev.size
                    prev.size = st.st_size
                prev.atime = now
                prev.path = str(path)

    def _forget(self, h: str) -> None:
        with self._meta_lock:
            prev = self._meta.pop(h, None)
            if prev is not None:
                self._total_bytes -= prev.size

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #

    def ensure_local(self, s3_uri: str) -> str:
        """
        Return a local on-disk path for the given S3 URI, downloading if necessary.
        Blocks until the file is present.
        """
        self._ensure_backend()
        if self._use_rust:
            path = self._rust_cache.ensure_local(s3_uri)
            self._touch(s3_uri, Path(path))
            return path

        dst = self._path_for(s3_uri)
        if dst.exists():
            self._touch(s3_uri, dst)
            return str(dst)

        key = _sha256_hex(s3_uri)
        fut = self._submit_if_absent(key, s3_uri)

        # Wait for completion (may raise)
        path = fut.result()

        # Best-effort cleanup (callback usually handles it)
        with self._inflight_lock:
            self._inflight.pop(key, None)
        return path

    def prime_prefetch(self, s3_uris: Sequence[str]) -> None:
        """
        Schedule downloads for a list of S3 URIs. Does not block.
        """
        self._ensure_backend()
        if self._use_rust:
            self._rust_cache.prime_prefetch(list(s3_uris))
            return

        for uri in s3_uris:
            if self._path_for(uri).exists():
                continue
            self._submit_if_absent(_sha256_hex(uri), uri)

    def wait_for_inflight(self, *, raise_errors: bool = False) -> None:
        """
        Block until all currently scheduled downloads complete.

        If raise_errors=True, re-raise the first download exception encountered.
        """
        self._ensure_backend()
        if self._use_rust:
            self._rust_cache.wait_for_inflight()
            return

        while True:
            with self._inflight_lock:
                futs = list(self._inflight.values())
            if not futs:
                return
            for f in futs:
                try:
                    f.result()
                except Exception:
                    if raise_errors:
                        raise
            time.sleep(0.01)

    def _download_if_needed(self, s3_uri: str) -> str:
        dst = self._path_for(s3_uri)
        if dst.exists():
            self._touch(s3_uri, dst)
            return str(dst)

        with self._lock_for(s3_uri):
            if dst.exists():
                self._touch(s3_uri, dst)
                return str(dst)

            bucket, key = parse_s3_uri(s3_uri)
            tmp = (
                self.cache_dir
                / "_tmp"
                / f"{dst.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )

            backoff = self.retry_backoff_seconds
            for attempt in range(6):
                try:
                    self._bump("attempts")
                    # Use get_object() instead of download_file() for less overhead
                    response = self.s3.get_object(Bucket=bucket, Key=key)
                    n = 0
                    with open(tmp, "wb") as f:
                        # Read in chunks to avoid holding large objects in memory
                        for chunk in response["Body"].iter_chunks(
                            chunk_size=1024 * 1024
                        ):
                            f.write(chunk)
                            n += len(chunk)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(str(tmp), str(dst))  # atomic
                    self._bump("downloaded")
                    self._bump("bytes", n)
                    self._touch(s3_uri, dst)
                    return str(dst)
                except Exception as e:
                    msg = repr(e)
                    if "SlowDown" in msg or "503" in msg or "Throttl" in msg:
                        self._bump("throttled")
                    if attempt == 5:
                        self._bump("failures")
                    else:
                        self._bump("retries")
                    self._log(f"download failed attempt {attempt+1}: {s3_uri} -> {e}")
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    if attempt < 5 and backoff > 0:
                        time.sleep(backoff)
                    backoff *= 2.0

            raise RuntimeError(f"Failed to download after retries: {s3_uri}")

    # ------------------------------------------------------------------ #
    # Eviction
    # ------------------------------------------------------------------ #

    def drop(self, s3_uris: Sequence[str]) -> int:
        """
        Unlink the cached files for `s3_uris`. Missing files are ignored.

        This is O(len(s3_uris)) with no directory walk, which is what makes it
        suitable for calling once per training step. Returns the number of
        files removed. URIs whose download is still in flight (Python backend)
        are skipped so the download is not raced.
        """
        self._ensure_backend()
        removed = 0
        for uri in s3_uris:
            h = _sha256_hex(uri)
            if not self._use_rust:
                with self._inflight_lock:
                    if h in self._inflight:
                        continue
            p = self.cache_dir / h[:2] / h[2:4] / h
            try:
                os.unlink(p)
                removed += 1
            except FileNotFoundError:
                pass
            except Exception as e:
                self._log(f"drop failed: {p} -> {e}")
                continue
            self._forget(h)
        return removed

    def prune(
        self,
        *,
        keep_uris: Sequence[str],
        mode: str = "window",  # "window" or "pressure"
        backend: str = "meta",  # "meta" or "scan"
    ) -> None:
        """
        Prune cache entries (full sweep).

        mode:
          - "window": delete anything not in keep_uris (subject to inflight + min_age safety)
          - "pressure": delete not-in-keep only if total_bytes > max_bytes (LRU-ish by atime)

        backend:
          - "meta": use this process's in-memory metadata (fast, but not global across processes)
          - "scan": scan filesystem for hashed files (global, slower). scan supports mode="window".

        A scan walks every cache directory, so run it rarely (epoch boundaries),
        not every few batches; use drop() for per-step eviction.

        If keep_uris exceeds max_bytes (in bytes), "window" will still keep it.
        In that case: shrink your prefetch window or increase max_bytes.
        """
        self._ensure_backend()
        keep_hashes = {_sha256_hex(u) for u in keep_uris}

        if mode not in {"window", "pressure"}:
            raise ValueError("mode must be 'window' or 'pressure'")
        if backend not in {"meta", "scan"}:
            raise ValueError("backend must be 'meta' or 'scan'")

        if backend == "scan":
            if mode != "window":
                raise ValueError(
                    "backend='scan' currently supports mode='window' only."
                )
            self._prune_scan_window(keep_hashes)
            return

        self._prune_meta(keep_hashes, mode=mode)

    def _prune_meta(self, keep_hashes: set, *, mode: str) -> None:
        now = time.time()
        with self._meta_lock:
            items = list(self._meta.items())
            total = self._total_bytes

        items.sort(key=lambda kv: kv[1].atime)  # oldest first

        def can_delete(h: str, meta: CacheMeta) -> bool:
            if h in keep_hashes:
                return False
            # Only check inflight if using Python backend (Rust backend manages its own inflight)
            if not self._use_rust:
                with self._inflight_lock:
                    if h in self._inflight:
                        return False
            if (now - meta.atime) < self.min_age_seconds_before_delete:
                return False
            return True

        for h, meta in items:
            if mode == "pressure" and total <= self.max_bytes:
                break
            if not can_delete(h, meta):
                continue
            try:
                p = Path(meta.path)
                if p.exists():
                    p.unlink()
                total -= meta.size
                with self._meta_lock:
                    self._meta.pop(h, None)
                    self._total_bytes = total
            except Exception as e:
                self._log(f"prune failed: {meta.path} -> {e}")

    def _prune_scan_window(self, keep_hashes: set) -> None:
        """
        Filesystem-based pruning: delete hashed cache files not in keep_hashes.

        This is robust across processes because it does not rely on in-memory meta.
        It uses os.scandir so directory entries are typed without a stat call;
        only deletion candidates are stat'ed (for the min-age check).
        """
        now = time.time()
        min_age = self.min_age_seconds_before_delete

        def walk(d: str) -> None:
            try:
                it = os.scandir(d)
            except OSError as e:
                self._log(f"scan prune: cannot list {d} -> {e}")
                return
            with it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in _INTERNAL_DIRS:
                                walk(entry.path)
                            continue
                        name = entry.name
                        # Only files named like our sha256 hex key are cache entries.
                        if len(name) != 64 or name in keep_hashes:
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if min_age > 0:
                            age = now - entry.stat(follow_symlinks=False).st_mtime
                            if age < min_age:
                                continue
                        os.unlink(entry.path)
                        self._forget(name)
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        self._log(f"scan prune failed: {entry.path} -> {e}")

        walk(str(self.cache_dir))

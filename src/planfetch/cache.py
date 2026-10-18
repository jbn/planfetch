# cache.py
"""S3 cache (NVMe/local disk) management for prefetching."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
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


class S3TerminalError(RuntimeError):
    """
    A GET that must not be retried: the object or bucket does not exist, we are
    not allowed to read it, or S3 rejected the request (any other 4xx).
    Carries `uri`, `code` (S3 error code) and `status` (HTTP status, 0 if unknown).
    """

    def __init__(self, uri: str, code: str, status: int, message: Optional[str] = None):
        self.uri = uri
        self.code = code
        self.status = int(status)
        super().__init__(message or f"{code} ({status}): {uri}")


class S3ObjectMissing(S3TerminalError):
    """NoSuchKey / NoSuchBucket / 404."""


class S3AccessDenied(S3TerminalError):
    """AccessDenied / 403."""


_TERMINAL_BY_TAG = {
    "NotFound": S3ObjectMissing,
    "Forbidden": S3AccessDenied,
    "ClientError": S3TerminalError,
}
# "NotFound: NoSuchKey (404): s3://bucket/key", as the Rust backend formats it.
_RUST_TERMINAL_RE = re.compile(
    r"^(NotFound|Forbidden|ClientError): (\S+) \((\d+)\): (s3://.*)$"
)

_NOT_FOUND_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "NotFound", "404"})
_FORBIDDEN_CODES = frozenset(
    {
        "AccessDenied",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "ExpiredToken",
        "403",
    }
)
_THROTTLE_CODES = frozenset(
    {"SlowDown", "Throttling", "RequestLimitExceeded", "503", "429"}
)


def _classify_boto_error(e: BaseException) -> Tuple[str, str, int]:
    """
    Map a boto3 exception to (class, code, status). class is one of
    "retry", "throttled", "not_found", "forbidden", "client".
    """
    if isinstance(e, ClientError):
        err = e.response.get("Error", {}) if hasattr(e, "response") else {}
        code = str(err.get("Code", ""))
        status = int(
            e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0
        )
        if code in _NOT_FOUND_CODES or status == 404:
            return "not_found", code or "HTTP404", status
        if code in _FORBIDDEN_CODES or status == 403:
            return "forbidden", code or "HTTP403", status
        if code in _THROTTLE_CODES or status in (503, 429):
            return "throttled", code or f"HTTP{status}", status
        if 400 <= status < 500:
            return "client", code or f"HTTP{status}", status
        return "retry", code or type(e).__name__, status
    msg = repr(e)
    if "SlowDown" in msg or "503" in msg or "Throttl" in msg:
        return "throttled", "SlowDown", 503
    return "retry", type(e).__name__, 0


def _rust_error(e: BaseException, uri: str) -> BaseException:
    """Turn the Rust backend's tagged RuntimeError into a typed exception."""
    m = _RUST_TERMINAL_RE.match(str(e))
    if m is None:
        return e
    cls = _TERMINAL_BY_TAG[m.group(1)]
    return cls(m.group(4) or uri, m.group(2), int(m.group(3)), str(e))


STAT_KEYS = (
    "attempts",
    "retries",
    "throttled",
    "failures",
    "not_found",
    "downloaded",
    "bytes",
)

_INTERNAL_DIRS = frozenset({"_tmp", "_locks", "_stats"})

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
    "_drop_queue",
    "_drop_thread",
    "_last_stats_dump",
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
    - A process other than the one that built the cache (a DataLoader worker)
      gets a smaller backend: `worker_download_threads` threads and, for Rust,
      a `worker_runtime_threads`-thread tokio runtime. Workers mostly wait for
      the parent's prefetch (`wait_local`) and rarely download.
    - In-memory metadata is per process. For a global sweep across processes,
      use prune_backend="scan" (filesystem scan).
    - Counters are per process. Every process writes its counters to
      `<cache_dir>/_stats/<pid>.json` after a download (at most once per
      `stats_dump_interval` seconds); `stats(all_processes=True)` sums them.

    Errors:
    - A key that does not exist (NoSuchKey / 404) raises `S3ObjectMissing` on
      the first attempt; AccessDenied / 403 raises `S3AccessDenied`; any other
      4xx raises `S3TerminalError`. None of these are retried. 5xx, SlowDown
      and connection errors are retried 6 times with backoff.
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
        worker_download_threads: int = 8,
        worker_runtime_threads: Optional[int] = 2,
        stats_dump_interval: float = 2.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "_tmp").mkdir(exist_ok=True)
        (self.cache_dir / "_locks").mkdir(exist_ok=True)
        stats_dir = self.cache_dir / "_stats"
        stats_dir.mkdir(exist_ok=True)
        # Only the constructing process gets here (workers inherit or unpickle),
        # so dumps from an earlier run do not leak into this run's totals.
        for old in stats_dir.glob("*.json"):
            try:
                old.unlink()
            except OSError:
                pass

        self.max_bytes = int(max_bytes)
        self.min_age_seconds_before_delete = float(min_age_seconds_before_delete)
        self.verbose = bool(verbose)
        self.retry_backoff_seconds = float(retry_backoff_seconds)

        self._download_threads = int(download_threads)
        self._worker_download_threads = int(worker_download_threads)
        self._worker_runtime_threads = (
            int(worker_runtime_threads) if worker_runtime_threads is not None else None
        )
        self._stats_dump_interval = float(stats_dump_interval)
        self._s3_max_pool_connections = int(s3_max_pool_connections)
        self._user_s3_client = s3_client
        self._owner_pid = os.getpid()

        # Use Rust backend if available and requested, but not if a custom s3_client is provided
        # (Rust uses its own AWS SDK client)
        self._use_rust = bool(use_rust) and HAS_RUST and s3_client is None

        self._pid: Optional[int] = None
        self._init_backend()

    # ------------------------------------------------------------------ #
    # Per-process backend lifecycle
    # ------------------------------------------------------------------ #

    def _is_worker(self) -> bool:
        """True in any process other than the one that constructed the cache."""
        return os.getpid() != getattr(self, "_owner_pid", os.getpid())

    def _init_backend(self) -> None:
        """Create the download backend and bookkeeping for the current process."""
        self._pid = os.getpid()
        worker = self._is_worker()
        threads = self._worker_download_threads if worker else self._download_threads
        if self._use_rust:
            self._rust_cache = RustS3Cache(
                str(self.cache_dir),
                threads,
                self._worker_runtime_threads if worker else None,
            )
        else:
            s3_client = self._user_s3_client
            if s3_client is None:
                cfg = BotoConfig(max_pool_connections=self._s3_max_pool_connections)
                s3_client = boto3.client("s3", config=cfg)
            self.s3 = s3_client

            self._pool = ThreadPoolExecutor(max_workers=threads)
            self._inflight: Dict[str, Future] = {}
            self._inflight_lock = threading.Lock()
        self._drop_queue: Optional[Any] = (
            None  # built lazily by drop_async (Python backend)
        )
        self._drop_thread: Optional[threading.Thread] = None
        self._last_stats_dump = 0.0

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

    def stats(self, all_processes: bool = False) -> Dict[str, int]:
        """
        Download counters.

        Keys: attempts, retries, throttled, failures, not_found, downloaded,
        bytes, inflight. A non-zero `throttled` or `retries` means S3 is pushing
        back (503 SlowDown) and what looks like slow S3 is really retry backoff.
        `not_found` counts keys S3 said do not exist (also in `failures`).

        all_processes=False: this process's backend only.
        all_processes=True: this process's live counters plus the last dump of
        every other process under `<cache_dir>/_stats/` (DataLoader workers write
        one after each download), with an extra key `processes`.
        """
        self._ensure_backend()
        if self._use_rust:
            local = dict(self._rust_cache.stats())
        else:
            with self._py_stats_lock:
                local = dict(self._py_stats)
            with self._inflight_lock:
                local["inflight"] = len(self._inflight)
        if not all_processes:
            return local

        total = dict(local)
        processes = 1
        me = os.getpid()
        for f in (self.cache_dir / "_stats").glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if int(d.get("pid", -1)) == me:
                continue  # live counters above are newer than our own dump
            processes += 1
            for k in local:
                total[k] += int(d.get(k, 0))
        total["processes"] = processes
        return total

    def dump_stats(self) -> str:
        """Write this process's counters to `<cache_dir>/_stats/<pid>.json`; returns the path."""
        self._ensure_backend()
        path = self.cache_dir / "_stats" / f"{os.getpid()}.json"
        if self._use_rust:
            self._rust_cache.stats_dump(str(path))
        else:
            d = self.stats()
            d["pid"] = os.getpid()
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(d))
            os.replace(tmp, path)
        self._last_stats_dump = time.monotonic()
        return str(path)

    def _maybe_dump_stats(self) -> None:
        """Dump after a download, at most once per `stats_dump_interval` seconds.

        DataLoader workers end with os._exit, so an atexit hook never runs there;
        dumping on the download path is what leaves their counters behind."""
        now = time.monotonic()
        if now - self._last_stats_dump >= self._stats_dump_interval:
            try:
                self.dump_stats()
            except OSError as e:
                self._log(f"stats dump failed: {e}")

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
        dst = self._path_for(s3_uri)
        if dst.exists():
            self._touch(s3_uri, dst)
            return str(dst)

        if self._use_rust:
            try:
                path = self._rust_cache.ensure_local(s3_uri)
            except RuntimeError as e:
                self._maybe_dump_stats()
                raise _rust_error(e, s3_uri) from None
            self._touch(s3_uri, Path(path))
            self._maybe_dump_stats()
            return path

        key = _sha256_hex(s3_uri)
        fut = self._submit_if_absent(key, s3_uri)

        # Wait for completion (may raise)
        try:
            path = fut.result()
        finally:
            # Best-effort cleanup (callback usually handles it)
            with self._inflight_lock:
                self._inflight.pop(key, None)
            self._maybe_dump_stats()
        return path

    def wait_local(self, s3_uri: str, timeout: float) -> bool:
        """
        Block until the cached file for `s3_uri` exists or `timeout` seconds pass.
        Never downloads and needs no backend, so a forked worker can wait for the
        parent's prefetch without building a runtime of its own. Returns True when
        the file is there.
        """
        dst = self._path_for(s3_uri)
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            if dst.exists():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

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
                break
            for f in futs:
                try:
                    f.result()
                except Exception:
                    if raise_errors:
                        raise
            time.sleep(0.01)
        if self._drop_queue is not None:
            self._drop_queue.join()

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
            last = ""
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
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    klass, code, status = _classify_boto_error(e)
                    last = f"{code} ({status})"
                    if klass == "throttled":
                        self._bump("throttled")
                    elif klass in ("not_found", "forbidden", "client"):
                        # Terminal: S3 gave a definite answer, retrying cannot help.
                        self._bump("failures")
                        if klass == "not_found":
                            self._bump("not_found")
                            raise S3ObjectMissing(s3_uri, code, status) from e
                        if klass == "forbidden":
                            raise S3AccessDenied(s3_uri, code, status) from e
                        raise S3TerminalError(s3_uri, code, status) from e
                    if attempt == 5:
                        self._bump("failures")
                    else:
                        self._bump("retries")
                    self._log(f"download failed attempt {attempt+1}: {s3_uri} -> {e}")
                    if attempt < 5 and backoff > 0:
                        time.sleep(backoff)
                    backoff *= 2.0

            raise RuntimeError(f"Failed to download after retries: {s3_uri}: {last}")

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

    def drop_async(self, s3_uris: Sequence[str]) -> None:
        """
        Unlink the cached files for `s3_uris` off the calling thread; returns at
        once. This is what PrefetchCoordinator uses per step so eviction does not
        cost training-thread time. `wait_for_inflight()` waits for pending
        unlinks too. Missing files are ignored.
        """
        self._ensure_backend()
        uris = list(s3_uris)
        if not uris:
            return
        for uri in uris:
            self._forget(_sha256_hex(uri))
        if self._use_rust:
            self._rust_cache.drop_paths(uris)
            return
        with self._inflight_lock:
            uris = [u for u in uris if _sha256_hex(u) not in self._inflight]
        if self._drop_queue is None:
            import queue

            self._drop_queue = queue.Queue()
            t = threading.Thread(
                target=self._drop_worker, name="planfetch-drop", daemon=True
            )
            self._drop_thread = t
            t.start()
        self._drop_queue.put([self._path_for(u) for u in uris])

    def _drop_worker(self) -> None:
        q = self._drop_queue
        assert q is not None
        while True:
            paths = q.get()
            try:
                for p in paths:
                    try:
                        os.unlink(p)
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        self._log(f"drop failed: {p} -> {e}")
            finally:
                q.task_done()

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

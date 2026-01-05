# cache.py
"""S3 cache (NVMe/local disk) management for prefetching."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import boto3
from botocore.config import Config as BotoConfig
from concurrent.futures import Future, ThreadPoolExecutor
from filelock import FileLock

# Try to import Rust extension for high-performance downloads
try:
    from .planfetch_rs import RustS3Cache
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
    - Concurrency: ThreadPoolExecutor wrapping boto3 (sync).
    - Pruning:
        mode="window": keep exactly a keep-set (great for huge datasets)
        mode="pressure": evict only under max_bytes pressure (LRU-ish)

    Multi-worker note:
    - In a multi-process DataLoader, each process has its own in-memory metadata.
    - For truly global pruning across processes, use prune_backend="scan" (filesystem scan),
      or implement a shared metadata store (SQLite).
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
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "_tmp").mkdir(exist_ok=True)
        (self.cache_dir / "_locks").mkdir(exist_ok=True)

        self.max_bytes = int(max_bytes)
        self.min_age_seconds_before_delete = float(min_age_seconds_before_delete)
        self.verbose = bool(verbose)

        # Use Rust backend if available and requested, but not if a custom s3_client is provided
        # (Rust uses its own AWS SDK client)
        self._use_rust = use_rust and HAS_RUST and s3_client is None
        if self._use_rust:
            self._rust_cache = RustS3Cache(str(cache_dir), download_threads)
        else:
            if s3_client is None:
                cfg = BotoConfig(max_pool_connections=int(s3_max_pool_connections))
                s3_client = boto3.client("s3", config=cfg)
            self.s3 = s3_client

            self._pool = ThreadPoolExecutor(max_workers=int(download_threads))
            self._inflight: Dict[str, Future] = {}
            self._inflight_lock = threading.Lock()

        self._meta: Dict[str, CacheMeta] = {}
        self._meta_lock = threading.Lock()
        self._total_bytes = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SharedDiskCache] {msg}")

    def _track_future(self, key: str, fut: Future) -> None:
        """Ensure inflight entries get removed when the Future completes."""
        def _cleanup(_f: Future) -> None:
            with self._inflight_lock:
                if self._inflight.get(key) is _f:
                    self._inflight.pop(key, None)
        fut.add_done_callback(_cleanup)

    def _path_for(self, s3_uri: str) -> Path:
        h = _sha256_hex(s3_uri)
        sub = self.cache_dir / h[:2] / h[2:4]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / h

    def _lock_for(self, s3_uri: str) -> FileLock:
        return FileLock(str(self.cache_dir / "_locks" / (_sha256_hex(s3_uri) + ".lock")))

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
                    self._total_bytes += (st.st_size - prev.size)
                    prev.size = st.st_size
                prev.atime = now
                prev.path = str(path)

    def ensure_local(self, s3_uri: str) -> str:
        """
        Return a local on-disk path for the given S3 URI, downloading if necessary.
        Blocks until the file is present.
        """
        if self._use_rust:
            path = self._rust_cache.ensure_local(s3_uri)
            self._touch(s3_uri, Path(path))
            return path

        dst = self._path_for(s3_uri)
        if dst.exists():
            self._touch(s3_uri, dst)
            return str(dst)

        key = _sha256_hex(s3_uri)
        with self._inflight_lock:
            fut = self._inflight.get(key)
            if fut is None:
                fut = self._pool.submit(self._download_if_needed, s3_uri)
                self._inflight[key] = fut
                self._track_future(key, fut)

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
        if self._use_rust:
            self._rust_cache.prime_prefetch(list(s3_uris))
            return

        for uri in s3_uris:
            if self._path_for(uri).exists():
                continue
            key = _sha256_hex(uri)
            with self._inflight_lock:
                if key in self._inflight:
                    continue
                fut = self._pool.submit(self._download_if_needed, uri)
                self._inflight[key] = fut
                self._track_future(key, fut)

    def wait_for_inflight(self, *, raise_errors: bool = False) -> None:
        """
        Block until all currently scheduled downloads complete.

        If raise_errors=True, re-raise the first download exception encountered.
        """
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
                except Exception as e:
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
            tmp = self.cache_dir / "_tmp" / f"{dst.name}.{os.getpid()}.{time.time_ns()}.tmp"

            backoff = 0.25
            for attempt in range(6):
                try:
                    # Use get_object() instead of download_file() for less overhead
                    response = self.s3.get_object(Bucket=bucket, Key=key)
                    with open(tmp, 'wb') as f:
                        # Read in chunks to avoid holding large objects in memory
                        for chunk in response['Body'].iter_chunks(chunk_size=1024 * 1024):
                            f.write(chunk)
                    os.replace(str(tmp), str(dst))  # atomic
                    self._touch(s3_uri, dst)
                    return str(dst)
                except Exception as e:
                    self._log(f"download failed attempt {attempt+1}: {s3_uri} -> {e}")
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    time.sleep(backoff)
                    backoff *= 2.0

            raise RuntimeError(f"Failed to download after retries: {s3_uri}")

    def prune(
        self,
        *,
        keep_uris: Sequence[str],
        mode: str = "window",         # "window" or "pressure"
        backend: str = "meta",        # "meta" or "scan"
    ) -> None:
        """
        Prune cache entries.

        mode:
          - "window": delete anything not in keep_uris (subject to inflight + min_age safety)
          - "pressure": delete not-in-keep only if total_bytes > max_bytes (LRU-ish by atime)

        backend:
          - "meta": use this process's in-memory metadata (fast, but not global across processes)
          - "scan": scan filesystem for hashed files (global, slower). scan supports mode="window".

        If keep_uris exceeds max_bytes (in bytes), "window" will still keep it.
        In that case: shrink your prefetch window or increase max_bytes.
        """
        keep_hashes = {_sha256_hex(u) for u in keep_uris}

        if mode not in {"window", "pressure"}:
            raise ValueError("mode must be 'window' or 'pressure'")
        if backend not in {"meta", "scan"}:
            raise ValueError("backend must be 'meta' or 'scan'")

        if backend == "scan":
            if mode != "window":
                raise ValueError("backend='scan' currently supports mode='window' only.")
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

        This is more robust across multiple processes because it does not rely on in-memory meta.
        It is slower because it stats files on disk.
        """
        now = time.time()

        # Only consider files whose name looks like our sha256 hex key.
        # Skip internal directories.
        internal = {"_tmp", "_locks"}

        for p in self.cache_dir.rglob("*"):
            if not p.is_file():
                continue
            # skip lock files and tmp
            if p.parent.name in internal:
                continue
            name = p.name
            if len(name) != 64:
                continue
            if name in keep_hashes:
                continue

            try:
                st = p.stat()
                age = now - st.st_mtime
                if age < self.min_age_seconds_before_delete:
                    continue
                p.unlink()
            except Exception as e:
                self._log(f"scan prune failed: {p} -> {e}")

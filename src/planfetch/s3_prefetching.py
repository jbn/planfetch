# s3_prefetching.py
from __future__ import annotations

import csv
import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

import boto3
from botocore.config import Config as BotoConfig
from concurrent.futures import Future, ThreadPoolExecutor
from filelock import FileLock

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


# ============================================================
# 1) CSV utilities
# ============================================================

def load_index_csv(
    csv_path: Union[str, os.PathLike],
    *,
    s3_uri_col: str,
    class_col: str,
) -> Tuple[List[str], np.ndarray]:
    """
    Load a CSV that contains (at minimum) two columns:
      - s3_uri_col: an S3 URI like "s3://bucket/key"
      - class_col: an integer-like class id

    Returns:
      s3_uris: list[str] length N
      labels:  np.ndarray[int64] shape [N]

    Note:
      This loads the whole CSV into memory. If your CSV is huge, consider:
      - Parquet + memory-mapped columns
      - sharded CSVs (per class or per prefix)
      - a database or key-value index
    """
    s3_uris: List[str] = []
    labels: List[int] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        if s3_uri_col not in reader.fieldnames or class_col not in reader.fieldnames:
            raise ValueError(f"CSV missing required columns. Found: {reader.fieldnames}")

        for row in reader:
            s3_uris.append(row[s3_uri_col])
            labels.append(int(row[class_col]))

    return s3_uris, np.asarray(labels, dtype=np.int64)


# ============================================================
# 2) S3 cache (NVMe/local disk)
# ============================================================

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
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "_tmp").mkdir(exist_ok=True)
        (self.cache_dir / "_locks").mkdir(exist_ok=True)

        self.max_bytes = int(max_bytes)
        self.min_age_seconds_before_delete = float(min_age_seconds_before_delete)
        self.verbose = bool(verbose)

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
                    self.s3.download_file(bucket, key, str(tmp))
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


# ============================================================
# 3) PlanView and plan providers
# ============================================================

class PlanView:
    """
    A dense, batch-aware epoch plan.

    Stores a matrix of indices: [num_batches, batch_size] as np.int64.

    Benefits:
    - ~16MB for 2M indices (instead of hundreds of MB as Python ints).
    - Fast iteration, fast window queries for prefetching.
    """
    def __init__(self, batches: np.ndarray):
        if batches.ndim != 2 or batches.dtype != np.int64:
            raise ValueError("PlanView expects a 2D np.int64 array [num_batches, batch_size].")
        self._batches = batches

    def __len__(self) -> int:
        return int(self._batches.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self._batches.shape[1])

    def batch(self, i: int) -> np.ndarray:
        return self._batches[int(i)]  # view

    def range_flat(self, lo: int, hi: int) -> np.ndarray:
        lo = max(0, int(lo))
        hi = min(len(self), int(hi))
        if hi <= lo:
            return np.empty((0,), dtype=np.int64)
        return self._batches[lo:hi].reshape(-1)


class PlanProvider:
    def make_epoch_plan(self, epoch: int) -> PlanView:
        raise NotImplementedError


class SequentialPlanProvider(PlanProvider):
    """
    Deterministic sequential plan, useful for val/test.
    """
    def __init__(
        self,
        n: int,
        batch_size: int,
        *,
        epoch_samples: Optional[int] = None,
        drop_last: bool = True,
    ):
        self.n = int(n)
        self.bs = int(batch_size)
        self.epoch_samples = int(epoch_samples) if epoch_samples is not None else self.n
        self.drop_last = bool(drop_last)

    def make_epoch_plan(self, epoch: int) -> PlanView:
        total = min(self.epoch_samples, self.n)
        if self.drop_last:
            nb = total // self.bs
            total2 = nb * self.bs
            idxs = np.arange(total2, dtype=np.int64)
        else:
            nb = math.ceil(total / self.bs)
            total2 = nb * self.bs
            idxs = np.arange(total2, dtype=np.int64)
            idxs[idxs >= total] = idxs[idxs >= total] % total
        batches = idxs.reshape(nb, self.bs)
        return PlanView(batches)


class ShuffleOncePerEpochPlanProvider(PlanProvider):
    """
    Default "shuffle once per epoch" plan.

    For large datasets, it avoids permuting the full [0..n) by using
    choice-without-replacement for epoch_samples.
    """
    def __init__(
        self,
        n: int,
        batch_size: int,
        *,
        seed: int = 123,
        epoch_samples: int = 2_000_000,
        drop_last: bool = True,
        full_permutation_threshold: int = 10_000_000,
    ):
        self.n = int(n)
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.epoch_samples = int(epoch_samples)
        self.drop_last = bool(drop_last)
        self.full_perm_thresh = int(full_permutation_threshold)

    def make_epoch_plan(self, epoch: int) -> PlanView:
        rng = np.random.default_rng(self.seed + int(epoch))

        total = min(self.epoch_samples, self.n)
        if self.drop_last:
            nb = total // self.bs
            total2 = nb * self.bs
        else:
            nb = math.ceil(total / self.bs)
            total2 = nb * self.bs

        if self.n <= self.full_perm_thresh:
            perm = rng.permutation(self.n).astype(np.int64, copy=False)
            seq = perm[:total2]
        else:
            seq = rng.choice(self.n, size=total2, replace=False).astype(np.int64, copy=False)

        if not self.drop_last and total2 > total:
            seq[total:] = seq[total:] % total

        batches = seq.reshape(nb, self.bs)
        return PlanView(batches)


def _choose_coprime_step(n: int, rng: np.random.Generator) -> int:
    if n <= 1:
        return 1
    step = int(rng.integers(1, n))
    for _ in range(256):
        if math.gcd(step, n) == 1:
            return step
        step += 1
        if step >= n:
            step = 1
    return 1


class _CyclicShuffler:
    """
    Memory-thrifty permutation-like stream over a fixed pool array.

    It never copies the pool. Each cycle uses:
      idx(pos) = (start + step*pos) mod n
    with gcd(step, n) == 1, yielding a full cycle.

    When the cycle finishes, it picks a new start/step.
    """

    def __init__(self, pool: np.ndarray, *, seed: int, epoch: int, class_id: int):
        self.pool = pool
        self.n = int(pool.shape[0])
        if self.n <= 0:
            raise ValueError("Empty class pool.")
        ss = np.random.SeedSequence([int(seed), int(epoch), int(class_id)])
        self.rng = np.random.default_rng(ss)
        self.start = int(self.rng.integers(0, self.n))
        self.step = _choose_coprime_step(self.n, self.rng)
        self.pos = 0

    def _next_cycle(self) -> None:
        self.start = int(self.rng.integers(0, self.n))
        self.step = _choose_coprime_step(self.n, self.rng)
        self.pos = 0

    def take(self, k: int) -> np.ndarray:
        k = int(k)
        if k <= 0:
            return np.empty((0,), dtype=np.int64)

        out = np.empty((k,), dtype=np.int64)
        written = 0
        while written < k:
            rem = self.n - self.pos
            take_now = min(k - written, rem)

            positions = self.pos + np.arange(take_now, dtype=np.int64)
            idxs = (self.start + (self.step * positions)) % self.n
            out[written:written + take_now] = self.pool[idxs]

            self.pos += take_now
            written += take_now

            if self.pos >= self.n and written < k:
                self._next_cycle()

        return out


def _allocate_fixed_counts(
    probs: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Allocate stable per-class counts for each batch, summing to batch_size.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    expected = probs * float(batch_size)
    base = np.floor(expected).astype(np.int64)
    rem = int(batch_size - int(base.sum()))
    if rem <= 0:
        return base

    frac = expected - base
    order = np.arange(len(probs))
    rng.shuffle(order)
    ranked = sorted(order.tolist(), key=lambda i: (frac[i],), reverse=True)
    base[ranked[:rem]] += 1
    return base


class StreamingStratifiedBatchPlanProvider(PlanProvider):
    """
    Memory-thrifty stratified batch plan provider.

    Key upgrade:
      - Avoids pool.copy() and per-epoch full shuffles of large class pools.
      - Uses a small per-class modular permutation stream instead.

    This keeps per-epoch allocations small even when class pools are enormous.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        *,
        seed: int = 123,
        epoch_samples: int = 2_000_000,
        drop_last: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        shuffle_within_batch: bool = True,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.epoch_samples = int(epoch_samples)
        self.drop_last = bool(drop_last)
        self.shuffle_within_batch = bool(shuffle_within_batch)

        classes = np.unique(self.labels)
        self.classes = np.asarray(classes, dtype=np.int64)

        # Pools built once
        self.class_to_pool: Dict[int, np.ndarray] = {}
        for c in self.classes:
            ci = int(c)
            self.class_to_pool[ci] = np.where(self.labels == c)[0].astype(np.int64)

        # Class probabilities (uniform over classes unless weighted)
        if class_weights is None:
            weights = {int(c): 1.0 for c in self.classes}
        else:
            weights = {int(k): float(v) for k, v in class_weights.items()}
            for c in self.classes:
                weights.setdefault(int(c), 0.0)

        w = np.array([weights[int(c)] for c in self.classes], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("class_weights sum to 0; provide positive weights.")
        self.class_probs = w / w.sum()

    def make_epoch_plan(self, epoch: int) -> PlanView:
        total = int(self.epoch_samples)  # may exceed dataset size; repeats are allowed
        if self.drop_last:
            nb = total // self.bs
            total2 = nb * self.bs
        else:
            nb = math.ceil(total / self.bs)
            total2 = nb * self.bs

        rng = np.random.default_rng(self.seed + int(epoch))

        counts = _allocate_fixed_counts(self.class_probs, self.bs, rng)

        active: List[Tuple[int, int]] = []
        for i, c in enumerate(self.classes):
            k = int(counts[i])
            if k > 0:
                active.append((int(c), k))
        if sum(k for _, k in active) != self.bs:
            raise RuntimeError("Internal error: counts do not sum to batch_size.")

        shufflers: Dict[int, _CyclicShuffler] = {}
        for c, _k in active:
            shufflers[c] = _CyclicShuffler(self.class_to_pool[c], seed=self.seed, epoch=int(epoch), class_id=int(c))

        out = np.empty((nb, self.bs), dtype=np.int64)

        for b in range(nb):
            pos = 0
            for c, k in active:
                idxs = shufflers[c].take(k)
                out[b, pos:pos + k] = idxs
                pos += k

            if self.shuffle_within_batch:
                perm = rng.permutation(self.bs)
                out[b, :] = out[b, perm]

        if not self.drop_last and total2 > total:
            out_flat = out.reshape(-1)
            out_flat[total:] = out_flat[total:] % int(len(self.labels))
            out = out_flat.reshape(nb, self.bs)

        return PlanView(out)


# ============================================================
# 4) DataLoader integration (PlanBatchSampler + Prefetch wrapper)
# ============================================================

class PlanBatchSampler(BatchSampler):
    """
    BatchSampler that yields batches from a PlanView.
    """
    def __init__(self, plan: PlanView):
        self.plan = plan

    def set_plan(self, plan: PlanView) -> None:
        self.plan = plan

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self.plan)):
            yield self.plan.batch(i)

    def __len__(self) -> int:
        return len(self.plan)


class PrefetchCoordinator:
    """
    Sliding-window prefetch + prune.

    Keeps batches:
      [batch_idx - B_back, batch_idx + B_ahead)

    This is designed to keep training loops unobtrusive:
      - PrefetchingLoader calls on_batch_yielded automatically
    """

    def __init__(
        self,
        cache: SharedDiskCache,
        s3_uris: Sequence[str],
        *,
        batches_ahead: int,
        batches_back: int,
        prune_every: int = 5,
        prune_mode: str = "window",
        prune_backend: str = "meta",
    ):
        self.cache = cache
        self.s3_uris = list(s3_uris)
        self.Ba = int(batches_ahead)
        self.Bb = int(batches_back)
        self.prune_every = int(prune_every)
        self.prune_mode = str(prune_mode)
        self.prune_backend = str(prune_backend)

        self.plan: Optional[PlanView] = None
        self._step = 0

    def set_plan(self, plan: PlanView) -> None:
        self.plan = plan
        self._step = 0

    def _window_bounds(self, batch_idx: int) -> Tuple[int, int]:
        assert self.plan is not None
        lo = max(0, int(batch_idx) - self.Bb)
        hi = min(len(self.plan), int(batch_idx) + self.Ba)
        return lo, hi

    def _window_uris(self, batch_idx: int) -> List[str]:
        assert self.plan is not None
        lo, hi = self._window_bounds(batch_idx)
        idxs = self.plan.range_flat(lo, hi)
        return [self.s3_uris[int(i)] for i in idxs]

    def prime(self) -> None:
        if self.plan is None:
            return
        keep = self._window_uris(0)
        self.cache.prime_prefetch(keep)

    def on_batch_yielded(self, batch_idx: int) -> None:
        if self.plan is None:
            return
        keep = self._window_uris(batch_idx)
        self.cache.prime_prefetch(keep)

        self._step += 1
        if self._step % self.prune_every == 0:
            self.cache.prune(keep_uris=keep, mode=self.prune_mode, backend=self.prune_backend)


class PrefetchingLoader:
    """
    Wrap a DataLoader and automatically advance prefetching per yielded batch.
    """
    def __init__(self, dataloader: DataLoader, prefetch: PrefetchCoordinator):
        self.dataloader = dataloader
        self.prefetch = prefetch

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        for bi, batch in enumerate(self.dataloader):
            self.prefetch.on_batch_yielded(bi)
            yield batch


class PrefetchController:
    """
    One method per epoch: set_epoch(epoch).
    """
    def __init__(self, plan_provider: PlanProvider, batch_sampler: PlanBatchSampler, prefetch: PrefetchCoordinator):
        self.plan_provider = plan_provider
        self.batch_sampler = batch_sampler
        self.prefetch = prefetch

    def set_epoch(self, epoch: int) -> None:
        plan = self.plan_provider.make_epoch_plan(epoch)
        self.batch_sampler.set_plan(plan)
        self.prefetch.set_plan(plan)
        self.prefetch.prime()


# ============================================================
# 5) Dataset wrapper
# ============================================================

class S3PathDataset(Dataset):
    """
    Dataset: idx -> (image, label)

    The Dataset owns preprocessing.
    The cache owns S3 fetching and local storage.
    """
    def __init__(
        self,
        s3_uris: Sequence[str],
        labels: Sequence[int],
        *,
        cache: SharedDiskCache,
        image_loader,
        transform=None,
    ):
        self.s3_uris = list(s3_uris)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.cache = cache
        self.image_loader = image_loader
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.s3_uris))

    def __getitem__(self, idx):
        i = int(idx)
        uri = self.s3_uris[i]
        path = self.cache.ensure_local(uri)
        img = self.image_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


# ============================================================
# 6) Convenience builder (least-obtrusive wiring)
# ============================================================

def build_prefetching_dataloader(
    *,
    dataset: Dataset,
    s3_uris: Sequence[str],
    plan_provider: PlanProvider,
    cache: SharedDiskCache,
    num_workers: int,
    batches_ahead: int,
    batches_back: int,
    prune_every: int = 5,
    prune_mode: str = "window",
    prune_backend: str = "meta",
    pin_memory: bool = True,
    persistent_workers: bool = True,
    worker_init_fn=None,
) -> Tuple[PrefetchingLoader, PrefetchController]:
    """
    Create:
      - DataLoader configured with a PlanBatchSampler
      - PrefetchingLoader wrapper
      - PrefetchController to advance epoch plans

    You call controller.set_epoch(epoch) once per epoch.
    """
    plan0 = plan_provider.make_epoch_plan(0)
    sampler = PlanBatchSampler(plan0)

    prefetch = PrefetchCoordinator(
        cache=cache,
        s3_uris=s3_uris,
        batches_ahead=batches_ahead,
        batches_back=batches_back,
        prune_every=prune_every,
        prune_mode=prune_mode,
        prune_backend=prune_backend,
    )

    controller = PrefetchController(plan_provider, sampler, prefetch)

    dl = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        worker_init_fn=worker_init_fn,
    )

    return PrefetchingLoader(dl, prefetch), controller


# ============================================================
# 7) Optional Lightning callback
# ============================================================

try:
    import pytorch_lightning as pl  # type: ignore

    class S3PrefetchLightningCallback(pl.Callback):  # type: ignore
        """
        Lightning callback that advances prefetch plans at the right time.

        Typical setup: one PrefetchController per split (train/val/test), because each split
        usually has its own plan provider and prefetch window.
        """

        def __init__(
            self,
            *,
            train_controller: Optional[PrefetchController] = None,
            val_controller: Optional[PrefetchController] = None,
            test_controller: Optional[PrefetchController] = None,
        ):
            super().__init__()
            self.train_controller = train_controller
            self.val_controller = val_controller
            self.test_controller = test_controller

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            if self.train_controller is not None:
                self.train_controller.set_epoch(trainer.current_epoch)

        def on_validation_epoch_start(self, trainer, pl_module) -> None:
            if self.val_controller is not None:
                self.val_controller.set_epoch(trainer.current_epoch)

        def on_test_epoch_start(self, trainer, pl_module) -> None:
            if self.test_controller is not None:
                self.test_controller.set_epoch(trainer.current_epoch)

except Exception:
    # Lightning is optional.
    pass

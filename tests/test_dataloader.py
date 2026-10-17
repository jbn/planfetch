"""Tests for planfetch.dataloader module (integration tests)."""

import threading
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest

from planfetch.cache import SharedDiskCache
from planfetch.dataloader import PrefetchCoordinator
from planfetch.plans import SequentialPlanProvider


class FakeStreamingBody:
    """Mock for boto3 StreamingBody."""

    def __init__(self, data: bytes):
        self._data = data

    def iter_chunks(self, chunk_size: int = 1024):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


class FakeS3Client:
    """
    Minimal mock for boto3 S3 client. Stores objects in-memory as bytes.
    """

    def __init__(self, objects: Dict[Tuple[str, str], bytes]):
        self.objects = dict(objects)
        self.calls = []
        self.lock = threading.Lock()

    def get_object(self, Bucket: str, Key: str) -> dict:
        with self.lock:
            self.calls.append((Bucket, Key))
        data = self.objects[(Bucket, Key)]
        return {"Body": FakeStreamingBody(data)}


def make_cache(tmp_path, fake_s3, *, max_bytes=10_000_000, min_age=0.0, threads=4):
    return SharedDiskCache(
        tmp_path / "cache",
        max_bytes=max_bytes,
        download_threads=threads,
        s3_client=fake_s3,
        min_age_seconds_before_delete=min_age,
    )


def test_prefetch_window_prunes_previous_batches(tmp_path):
    # Make 6 unique URIs, 3 batches of size 2
    objects = {("b", f"k{i}"): bytes([i]) for i in range(6)}
    fake = FakeS3Client(objects)
    cache = make_cache(tmp_path, fake, min_age=0.0, threads=4)

    s3_uris = [f"s3://b/k{i}" for i in range(6)]

    plan_provider = SequentialPlanProvider(
        n=6, batch_size=2, epoch_samples=6, drop_last=True
    )
    plan = plan_provider.make_epoch_plan(0)

    prefetch = PrefetchCoordinator(
        cache=cache,
        s3_uris=s3_uris,
        batches_ahead=2,  # keep current + next
        batches_back=0,
        prune_every=1,
        prune_mode="window",
        prune_backend="meta",
    )
    prefetch.set_plan(plan)

    # Batch 0: keep indices [0,1,2,3]
    prefetch.on_batch_yielded(0)
    cache.wait_for_inflight(raise_errors=True)

    keep0 = set(s3_uris[i] for i in [0, 1, 2, 3])
    for uri in keep0:
        assert cache._path_for(uri).exists()

    # Batch 1: keep indices [2,3,4,5], so [0,1] should get pruned
    prefetch.on_batch_yielded(1)
    cache.wait_for_inflight(raise_errors=True)

    assert not cache._path_for("s3://b/k0").exists()
    assert not cache._path_for("s3://b/k1").exists()
    assert cache._path_for("s3://b/k2").exists()
    assert cache._path_for("s3://b/k3").exists()
    assert cache._path_for("s3://b/k4").exists()
    assert cache._path_for("s3://b/k5").exists()


# ---------------------------------------------------------------------------
# Incremental coordinator (FSx benchmark 2026-10-16: the old coordinator
# re-primed the whole window every step and walked the cache dir every 5
# batches, ~163 ms/batch on the training thread).
# ---------------------------------------------------------------------------

import os

from planfetch.dataloader import build_prefetching_dataloader
from planfetch.dataset import S3PathDataset
from planfetch.plans import PlanView


class RecordingCache:
    """Duck-typed cache that records what the coordinator asks of it."""

    def __init__(self):
        self.primed: list = []
        self.dropped: list = []
        self.pruned: list = []

    def prime_prefetch(self, s3_uris: Sequence[str]) -> None:
        self.primed.append(list(s3_uris))

    def drop(self, s3_uris: Sequence[str]) -> int:
        self.dropped.append(list(s3_uris))
        return len(s3_uris)

    def prune(
        self, *, keep_uris: Sequence[str], mode: str = "window", backend: str = "meta"
    ) -> None:
        self.pruned.append((set(keep_uris), mode, backend))


def _uris(n):
    return [f"s3://b/k{i}" for i in range(n)]


def _seq_plan(n, bs):
    return SequentialPlanProvider(n=n, batch_size=bs, drop_last=True).make_epoch_plan(0)


def _batch_uris(uris, plan, b):
    return [uris[int(i)] for i in plan.batch(b)]


def test_coordinator_primes_one_batch_in_and_drops_one_batch_out():
    uris = _uris(20)
    plan = _seq_plan(20, 2)  # 10 batches
    cache = RecordingCache()
    coord = PrefetchCoordinator(cache, uris, batches_ahead=3, batches_back=1)
    coord.set_plan(plan)

    coord.prime()
    assert cache.primed == [uris[0:6]]  # batches 0,1,2

    coord.on_batch_yielded(0)
    assert cache.primed == [uris[0:6]]  # window [0,3) already primed
    assert cache.dropped == []

    coord.on_batch_yielded(1)
    assert cache.primed[-1] == _batch_uris(uris, plan, 3)
    assert cache.dropped == []

    coord.on_batch_yielded(2)
    assert cache.primed[-1] == _batch_uris(uris, plan, 4)
    assert cache.dropped == [_batch_uris(uris, plan, 0)]

    for b in range(3, 10):
        coord.on_batch_yielded(b)

    # Every URI primed exactly once, per-step calls are O(batch), not O(window).
    flat = [u for call in cache.primed for u in call]
    assert sorted(flat) == sorted(uris)
    assert all(len(call) <= 6 for call in cache.primed)
    assert max(len(call) for call in cache.primed[1:]) == 2
    # Batches 0..7 dropped (window at the last step is [8, 10)).
    assert cache.dropped == [_batch_uris(uris, plan, b) for b in range(8)]
    assert all(len(call) == 2 for call in cache.dropped)
    assert cache.pruned == []  # prune_every defaults to off


def test_coordinator_handles_forward_jump():
    uris = _uris(20)
    plan = _seq_plan(20, 2)
    cache = RecordingCache()
    coord = PrefetchCoordinator(cache, uris, batches_ahead=3, batches_back=1)
    coord.set_plan(plan)
    coord.prime()
    for b in range(3):
        coord.on_batch_yielded(b)
    cache.primed.clear()
    cache.dropped.clear()

    coord.on_batch_yielded(6)  # window [5, 9): primed so far [0,5), dropped [0,1)
    primed = {u for call in cache.primed for u in call}
    assert primed == {u for b in (5, 6, 7, 8) for u in _batch_uris(uris, plan, b)}
    dropped = {u for call in cache.dropped for u in call}
    assert dropped == {u for b in (1, 2, 3, 4) for u in _batch_uris(uris, plan, b)}


def test_coordinator_restart_from_earlier_batch_reprimes_and_drops_stale():
    uris = _uris(20)
    plan = _seq_plan(20, 2)
    cache = RecordingCache()
    coord = PrefetchCoordinator(cache, uris, batches_ahead=3, batches_back=1)
    coord.set_plan(plan)
    coord.prime()
    coord.on_batch_yielded(6)  # window [5, 9)
    cache.primed.clear()
    cache.dropped.clear()

    coord.on_batch_yielded(0)  # window [0, 3)
    primed = {u for call in cache.primed for u in call}
    assert primed == set(uris[0:6])
    dropped = {u for call in cache.dropped for u in call}
    assert dropped == {u for b in (5, 6, 7, 8) for u in _batch_uris(uris, plan, b)}

    # And it keeps going incrementally from there.
    coord.on_batch_yielded(1)
    assert cache.primed[-1] == _batch_uris(uris, plan, 3)


def test_coordinator_keeps_uri_shared_by_two_batches_until_both_leave():
    uris = _uris(8)
    # k0 is in batch 0 and batch 3.
    plan = PlanView(np.array([[0, 1], [2, 3], [4, 5], [0, 6]], dtype=np.int64))
    cache = RecordingCache()
    coord = PrefetchCoordinator(cache, uris, batches_ahead=4, batches_back=0)
    coord.set_plan(plan)
    coord.prime()
    for b in range(4):
        coord.on_batch_yielded(b)
    assert cache.dropped == [[uris[1]], [uris[2], uris[3]], [uris[4], uris[5]]]


def test_set_plan_drops_previous_epoch_files_not_in_new_window():
    uris = _uris(12)
    plan0 = _seq_plan(12, 2)  # 6 batches
    cache = RecordingCache()
    coord = PrefetchCoordinator(cache, uris, batches_ahead=2, batches_back=0)
    coord.set_plan(plan0)
    coord.prime()
    for b in range(6):
        coord.on_batch_yielded(b)
    cache.primed.clear()
    cache.dropped.clear()
    # Batch 5 (k10, k11) is still on disk at the end of the epoch.

    plan1 = PlanView(
        np.array([[10, 0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 11]], dtype=np.int64)
    )
    coord.set_plan(plan1)
    coord.prime()
    assert cache.primed == [[uris[10], uris[0], uris[1], uris[2]]]
    assert cache.dropped == [[uris[11]]]  # k10 is in the new window, k11 is not


def test_set_plan_without_prime_still_flushes_on_first_step():
    uris = _uris(12)
    cache = RecordingCache()
    coord = PrefetchCoordinator(cache, uris, batches_ahead=2, batches_back=0)
    coord.set_plan(_seq_plan(12, 2))
    coord.prime()
    for b in range(6):
        coord.on_batch_yielded(b)
    cache.dropped.clear()

    coord.set_plan(_seq_plan(12, 2))
    coord.on_batch_yielded(0)  # window [0,2) = k0..k3; stale k10,k11
    assert cache.dropped == [[uris[10], uris[11]]]


def test_prune_every_sweeps_with_window_keep_set():
    uris = _uris(20)
    plan = _seq_plan(20, 2)
    cache = RecordingCache()
    coord = PrefetchCoordinator(
        cache,
        uris,
        batches_ahead=3,
        batches_back=1,
        prune_every=2,
        prune_mode="window",
        prune_backend="scan",
    )
    coord.set_plan(plan)
    coord.prime()
    coord.on_batch_yielded(0)
    assert cache.pruned == []
    coord.on_batch_yielded(1)
    assert len(cache.pruned) == 1
    keep, mode, backend = cache.pruned[0]
    assert keep == set(uris[0:8])  # window [0, 4)
    assert (mode, backend) == ("window", "scan")
    coord.on_batch_yielded(2)
    coord.on_batch_yielded(3)
    assert len(cache.pruned) == 2
    assert cache.pruned[1][0] == {
        u for b in range(2, 6) for u in _batch_uris(uris, plan, b)
    }


def test_coordinator_rejects_window_smaller_than_loader_depth():
    coord = PrefetchCoordinator(
        RecordingCache(), _uris(8), batches_ahead=8, batches_back=0
    )
    with pytest.raises(ValueError, match="batches_ahead"):
        coord.check_loader_depth(num_workers=4, prefetch_factor=2)
    coord.check_loader_depth(num_workers=4, prefetch_factor=1)
    coord.check_loader_depth(num_workers=0, prefetch_factor=2)
    PrefetchCoordinator(
        RecordingCache(), _uris(8), batches_ahead=9, batches_back=0
    ).check_loader_depth(num_workers=4, prefetch_factor=2)


def _picklable_objects(n):
    return {("b", f"k{i}"): i.to_bytes(2, "big") for i in range(n)}


class LocklessFakeS3Client:
    """Fork-friendly fake: no lock to inherit mid-acquire."""

    def __init__(self, objects):
        self.objects = dict(objects)

    def get_object(self, Bucket: str, Key: str) -> dict:
        return {"Body": FakeStreamingBody(self.objects[(Bucket, Key)])}


def _load_int(path: str) -> int:
    return int.from_bytes(Path(path).read_bytes(), "big")


def test_build_prefetching_dataloader_rejects_small_window(tmp_path):
    n = 40
    uris = _uris(n)
    cache = SharedDiskCache(
        tmp_path / "cache",
        max_bytes=10_000_000,
        download_threads=2,
        s3_client=LocklessFakeS3Client(_picklable_objects(n)),
        min_age_seconds_before_delete=0.0,
    )
    ds = S3PathDataset(uris, np.arange(n), cache=cache, image_loader=_load_int)
    provider = SequentialPlanProvider(n=n, batch_size=4, drop_last=True)
    with pytest.raises(ValueError, match="batches_ahead"):
        build_prefetching_dataloader(
            dataset=ds,
            s3_uris=uris,
            plan_provider=provider,
            cache=cache,
            num_workers=4,
            batches_ahead=8,  # == 4 workers x prefetch_factor 2
            batches_back=0,
            pin_memory=False,
            persistent_workers=False,
        )


def test_end_to_end_with_forked_dataloader_workers(tmp_path):
    n = 40
    uris = _uris(n)
    cache = SharedDiskCache(
        tmp_path / "cache",
        max_bytes=10_000_000,
        download_threads=2,
        s3_client=LocklessFakeS3Client(_picklable_objects(n)),
        min_age_seconds_before_delete=0.0,
    )
    ds = S3PathDataset(uris, np.arange(n), cache=cache, image_loader=_load_int)
    provider = SequentialPlanProvider(n=n, batch_size=4, drop_last=True)
    loader, controller = build_prefetching_dataloader(
        dataset=ds,
        s3_uris=uris,
        plan_provider=provider,
        cache=cache,
        num_workers=2,
        batches_ahead=6,
        batches_back=1,
        pin_memory=False,
        persistent_workers=False,
    )
    controller.set_epoch(0)
    seen = []
    for x, y in loader:
        assert x.tolist() == y.tolist()
        seen.extend(y.tolist())
    assert seen == list(range(n))

    # Only the last window [8, 10) is left on disk: 2 batches x 4 files.
    left = [
        p
        for d, _, files in os.walk(tmp_path / "cache")
        for p in files
        if len(p) == 64 and Path(d).name not in ("_tmp", "_locks")
    ]
    assert len(left) == 8
    # The parent's cache was not replaced by the workers' copies.
    assert cache._pid == os.getpid()

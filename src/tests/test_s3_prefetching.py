
import threading
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

import s3_prefetching as m


class FakeS3Client:
    """
    Minimal mock for boto3 S3 client. Stores objects in-memory as bytes.
    """
    def __init__(self, objects: Dict[Tuple[str, str], bytes]):
        self.objects = dict(objects)
        self.calls = []
        self.lock = threading.Lock()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        with self.lock:
            self.calls.append((bucket, key))
        data = self.objects[(bucket, key)]
        with open(filename, "wb") as f:
            f.write(data)


def make_cache(tmp_path, fake_s3, *, max_bytes=10_000_000, min_age=0.0, threads=4):
    return m.SharedDiskCache(
        tmp_path / "cache",
        max_bytes=max_bytes,
        download_threads=threads,
        s3_client=fake_s3,
        min_age_seconds_before_delete=min_age,
    )


def test_planview_range_flat():
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    pv = m.PlanView(arr)
    assert len(pv) == 3
    assert pv.batch_size == 4
    assert np.all(pv.batch(1) == np.array([4, 5, 6, 7], dtype=np.int64))

    flat = pv.range_flat(1, 3)
    assert flat.shape == (8,)
    assert np.all(flat == np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64))


def test_streaming_stratified_counts_and_determinism():
    # Two balanced classes, batch_size divisible by num_classes -> exact per-batch stratification.
    labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    prov = m.StreamingStratifiedBatchPlanProvider(labels, batch_size=10, epoch_samples=100, seed=123)

    plan0_a = prov.make_epoch_plan(0)
    plan0_b = prov.make_epoch_plan(0)
    assert np.array_equal(
        plan0_a.range_flat(0, len(plan0_a)),
        plan0_b.range_flat(0, len(plan0_b)),
    )

    plan1 = prov.make_epoch_plan(1)
    assert not np.array_equal(plan0_a.range_flat(0, 3), plan1.range_flat(0, 3))

    # Check each batch has 5 of each class
    for bi in range(len(plan0_a)):
        idxs = plan0_a.batch(bi)
        batch_labels = labels[idxs]
        c0 = int(np.sum(batch_labels == 0))
        c1 = int(np.sum(batch_labels == 1))
        assert c0 == 5
        assert c1 == 5


def test_shared_disk_cache_ensure_local_dedup(tmp_path):
    objects = {("b", "k1"): b"hello"}
    fake = FakeS3Client(objects)
    cache = make_cache(tmp_path, fake)

    uri = "s3://b/k1"
    p1 = cache.ensure_local(uri)
    assert Path(p1).exists()
    assert Path(p1).read_bytes() == b"hello"
    assert len(fake.calls) == 1

    # Second call should hit disk, not re-download
    p2 = cache.ensure_local(uri)
    assert p2 == p1
    assert len(fake.calls) == 1


def test_shared_disk_cache_concurrent_dedup(tmp_path):
    objects = {("b", "k1"): b"x" * 1024}
    fake = FakeS3Client(objects)
    cache = make_cache(tmp_path, fake, threads=8)

    uri = "s3://b/k1"
    results = []

    def worker():
        results.append(cache.ensure_local(uri))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert results[0] == results[1]
    assert len(fake.calls) == 1


def test_prime_prefetch_and_wait(tmp_path):
    objects = {("b", "k1"): b"a", ("b", "k2"): b"b"}
    fake = FakeS3Client(objects)
    cache = make_cache(tmp_path, fake, threads=4)

    cache.prime_prefetch(["s3://b/k1", "s3://b/k2"])
    cache.wait_for_inflight(raise_errors=True)

    p1 = cache.ensure_local("s3://b/k1")
    p2 = cache.ensure_local("s3://b/k2")
    assert Path(p1).read_bytes() == b"a"
    assert Path(p2).read_bytes() == b"b"
    # exactly two downloads total
    assert len(fake.calls) == 2


def test_prune_window_meta(tmp_path):
    objects = {
        ("b", "k1"): b"x" * 10,
        ("b", "k2"): b"y" * 10,
        ("b", "k3"): b"z" * 10,
    }
    fake = FakeS3Client(objects)
    cache = make_cache(tmp_path, fake, max_bytes=10_000_000, min_age=0.0)

    u1, u2, u3 = "s3://b/k1", "s3://b/k2", "s3://b/k3"
    p1 = cache.ensure_local(u1)
    p2 = cache.ensure_local(u2)
    p3 = cache.ensure_local(u3)
    assert Path(p1).exists() and Path(p2).exists() and Path(p3).exists()

    cache.prune(keep_uris=[u2], mode="window", backend="meta")
    assert not Path(p1).exists()
    assert Path(p2).exists()
    assert not Path(p3).exists()


def test_prune_window_scan_removes_orphans(tmp_path):
    objects = {("b", "k1"): b"hello"}
    fake = FakeS3Client(objects)
    cache = make_cache(tmp_path, fake, min_age=0.0)

    # Create an "orphan" cache file that meta doesn't know about
    orphan_uri = "s3://b/orphan"
    orphan_path = cache._path_for(orphan_uri)  # internal helper
    orphan_path.parent.mkdir(parents=True, exist_ok=True)
    orphan_path.write_bytes(b"orphan")

    assert orphan_path.exists()
    # scan prune should remove it if not kept
    cache.prune(keep_uris=[], mode="window", backend="scan")
    assert not orphan_path.exists()


def test_prefetch_window_prunes_previous_batches(tmp_path):
    # Make 6 unique URIs, 3 batches of size 2
    objects = {("b", f"k{i}"): bytes([i]) for i in range(6)}
    fake = FakeS3Client(objects)
    cache = make_cache(tmp_path, fake, min_age=0.0, threads=4)

    s3_uris = [f"s3://b/k{i}" for i in range(6)]

    plan_provider = m.SequentialPlanProvider(n=6, batch_size=2, epoch_samples=6, drop_last=True)
    plan = plan_provider.make_epoch_plan(0)

    prefetch = m.PrefetchCoordinator(
        cache=cache,
        s3_uris=s3_uris,
        batches_ahead=2,   # keep current + next
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

"""Tests for planfetch.dataloader module (integration tests)."""
import threading
from pathlib import Path
from typing import Dict, Tuple

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
            yield self._data[i:i + chunk_size]


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

    plan_provider = SequentialPlanProvider(n=6, batch_size=2, epoch_samples=6, drop_last=True)
    plan = plan_provider.make_epoch_plan(0)

    prefetch = PrefetchCoordinator(
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

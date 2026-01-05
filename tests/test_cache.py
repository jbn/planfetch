"""Tests for planfetch.cache module."""

import threading
from pathlib import Path
from typing import Dict, Tuple

import pytest

from planfetch.cache import SharedDiskCache, has_rust_backend, assert_rust_backend


def test_has_rust_backend_returns_bool():
    """has_rust_backend() should return a boolean."""
    result = has_rust_backend()
    assert isinstance(result, bool)


def test_assert_rust_backend_when_available():
    """assert_rust_backend() should not raise if Rust is available."""
    if has_rust_backend():
        # Should not raise
        assert_rust_backend()
    else:
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="planfetch_rs"):
            assert_rust_backend()


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

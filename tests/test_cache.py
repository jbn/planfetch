"""Tests for planfetch.cache module."""

import json
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


# ---------------------------------------------------------------------------
# Fixes from the FSx benchmark (2026-10-16): path_for, drop, stats, scan prune,
# fork/pickle safety.
# ---------------------------------------------------------------------------

import multiprocessing
import os
import pickle
import time


class PicklableFakeS3Client(FakeS3Client):
    """FakeS3Client whose lock survives pickling (dropped and rebuilt)."""

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("lock", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.lock = threading.Lock()


class FlakyS3Client(FakeS3Client):
    """Fails the first `fail_first` calls per key, then succeeds."""

    def __init__(self, objects, fail_first: int, message: str = "boom"):
        super().__init__(objects)
        self.fail_first = fail_first
        self.message = message
        self.failures: Dict[Tuple[str, str], int] = {}

    def get_object(self, Bucket: str, Key: str) -> dict:
        with self.lock:
            self.calls.append((Bucket, Key))
            n = self.failures.get((Bucket, Key), 0)
            if n < self.fail_first:
                self.failures[(Bucket, Key)] = n + 1
                raise Exception(self.message)
        return {"Body": FakeStreamingBody(self.objects[(Bucket, Key)])}


def test_path_for_is_pure_and_matches_download_location(tmp_path):
    fake = FakeS3Client({("b", "k1"): b"hello"})
    cache = make_cache(tmp_path, fake)
    uri = "s3://b/k1"

    p = cache.path_for(uri)
    # Computing the path must not create directories (13k mkdirs/step otherwise).
    assert not Path(p).parent.exists()
    assert cache.ensure_local(uri) == p
    assert Path(p).exists()


def test_drop_unlinks_files_and_ignores_missing(tmp_path):
    fake = FakeS3Client({("b", "k1"): b"x" * 10, ("b", "k2"): b"y" * 10})
    cache = make_cache(tmp_path, fake)
    p1 = cache.ensure_local("s3://b/k1")
    p2 = cache.ensure_local("s3://b/k2")

    removed = cache.drop(["s3://b/k1", "s3://b/never-downloaded"])
    assert removed == 1
    assert not Path(p1).exists()
    assert Path(p2).exists()
    # meta bookkeeping follows the unlink
    assert cache._total_bytes == 10
    assert cache.drop(["s3://b/k1"]) == 0


def test_drop_skips_inflight_downloads(tmp_path):
    fake = FakeS3Client({("b", "k1"): b"x"})
    cache = make_cache(tmp_path, fake)
    cache.prime_prefetch(["s3://b/k1"])
    # Whether or not the download has landed yet, drop must not interfere with it.
    cache.drop(["s3://b/k1"])
    cache.wait_for_inflight(raise_errors=True)
    assert Path(cache.ensure_local("s3://b/k1")).read_bytes() == b"x"


def test_stats_count_downloads_and_bytes(tmp_path):
    fake = FakeS3Client({("b", "k1"): b"x" * 100, ("b", "k2"): b"y" * 50})
    cache = make_cache(tmp_path, fake)
    assert cache.stats() == {
        "attempts": 0,
        "retries": 0,
        "throttled": 0,
        "failures": 0,
        "downloaded": 0,
        "bytes": 0,
        "inflight": 0,
    }
    cache.ensure_local("s3://b/k1")
    cache.ensure_local("s3://b/k2")
    cache.ensure_local("s3://b/k1")  # cached: no attempt
    s = cache.stats()
    assert s["attempts"] == 2
    assert s["downloaded"] == 2
    assert s["bytes"] == 150
    assert s["retries"] == 0
    assert s["failures"] == 0
    assert s["inflight"] == 0


def test_stats_count_retries_and_throttles(tmp_path):
    fake = FlakyS3Client({("b", "k1"): b"x"}, fail_first=2, message="SlowDown 503")
    cache = SharedDiskCache(
        tmp_path / "cache",
        max_bytes=1_000,
        download_threads=2,
        s3_client=fake,
        retry_backoff_seconds=0.0,
    )
    cache.ensure_local("s3://b/k1")
    s = cache.stats()
    assert s["attempts"] == 3
    assert s["retries"] == 2
    assert s["throttled"] == 2
    assert s["failures"] == 0
    assert s["downloaded"] == 1


def test_stats_count_failures_after_retries_exhausted(tmp_path):
    fake = FlakyS3Client({("b", "k1"): b"x"}, fail_first=100)
    cache = SharedDiskCache(
        tmp_path / "cache",
        max_bytes=1_000,
        download_threads=2,
        s3_client=fake,
        retry_backoff_seconds=0.0,
    )
    with pytest.raises(RuntimeError):
        cache.ensure_local("s3://b/k1")
    s = cache.stats()
    assert s["attempts"] == 6
    assert s["retries"] == 5
    assert s["failures"] == 1
    assert s["inflight"] == 0


def test_prune_scan_keeps_window_and_skips_internal_dirs(tmp_path):
    fake = FakeS3Client({("b", "k1"): b"a", ("b", "k2"): b"b"})
    cache = make_cache(tmp_path, fake, min_age=0.0)
    p1 = Path(cache.ensure_local("s3://b/k1"))
    p2 = Path(cache.ensure_local("s3://b/k2"))
    tmp_file = cache.cache_dir / "_tmp" / ("a" * 64)
    tmp_file.write_bytes(b"partial")
    lock_file = cache.cache_dir / "_locks" / ("b" * 64)
    lock_file.write_bytes(b"")
    # A sibling tmp file from the Rust backend (<hash>.tmp.<pid>) is not a cache entry
    rust_tmp = p1.parent / (p1.name + ".tmp.123")
    rust_tmp.write_bytes(b"partial")

    cache.prune(keep_uris=["s3://b/k2"], mode="window", backend="scan")

    assert not p1.exists()
    assert p2.exists()
    assert tmp_file.exists()
    assert lock_file.exists()
    assert rust_tmp.exists()


def test_prune_scan_respects_min_age(tmp_path):
    fake = FakeS3Client({("b", "k1"): b"a"})
    cache = make_cache(tmp_path, fake, min_age=60.0)
    p1 = Path(cache.ensure_local("s3://b/k1"))
    cache.prune(keep_uris=[], mode="window", backend="scan")
    assert p1.exists()


def test_cache_pickle_roundtrip_rebuilds_backend(tmp_path):
    fake = PicklableFakeS3Client({("b", "k1"): b"hello"})
    cache = make_cache(tmp_path, fake)
    cache.ensure_local("s3://b/k1")

    clone = pickle.loads(pickle.dumps(cache))
    assert clone.cache_dir == cache.cache_dir
    # The clone has its own thread pool / inflight table and works.
    assert Path(clone.ensure_local("s3://b/k1")).read_bytes() == b"hello"
    assert clone._pool is not cache._pool


def _child_check(cache, uri, parent_pool_id, q):
    try:
        path = cache.ensure_local(uri)
        q.put(
            {
                "path": path,
                "data": Path(path).read_bytes(),
                "pool_id": id(cache._pool),
                "pid_ok": cache._pid == os.getpid(),
            }
        )
    except Exception as e:  # pragma: no cover - reported to the parent
        q.put({"error": repr(e)})


def test_forked_child_rebuilds_backend(tmp_path):
    fake = FakeS3Client({("b", "k1"): b"hello", ("b", "k2"): b"world"})
    cache = make_cache(tmp_path, fake)
    cache.ensure_local("s3://b/k1")  # pool and locks exist in the parent
    parent_pool_id = id(cache._pool)

    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_check, args=(cache, "s3://b/k2", parent_pool_id, q))
    p.daemon = True
    p.start()
    try:
        res = q.get(timeout=20)
    except Exception:
        res = {"error": "child did not answer (hang in forked worker)"}
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
    assert "error" not in res, res
    assert res["data"] == b"world"
    assert res["pid_ok"]
    assert res["pool_id"] != parent_pool_id
    # The parent is untouched and still works.
    assert cache._pid == os.getpid()
    assert Path(cache.ensure_local("s3://b/k2")).read_bytes() == b"world"


def _child_rust_stats(cache, q):
    try:
        s = cache.stats()  # rebuilds the backend in the child
        q.put({"pid_ok": cache._pid == os.getpid(), "inflight": s["inflight"]})
    except Exception as e:  # pragma: no cover - reported to the parent
        q.put({"error": repr(e)})


@pytest.mark.skipif(not has_rust_backend(), reason="Rust backend not built")
def test_rust_backend_forked_child_does_not_hang_dropping_parent_runtime(
    tmp_path, monkeypatch
):
    """A forked DataLoader worker inherits the parent's RustS3Cache and replaces
    it on first use. Dropping the inherited tokio runtime in the child blocks
    forever: tokio's blocking pool waits for worker threads that only exist in
    the parent. The child must leak the inherited runtime instead."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    cache = SharedDiskCache(tmp_path / "cache", max_bytes=1_000, download_threads=2)
    assert cache._use_rust
    cache.stats()  # the runtime exists in the parent before the fork

    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_rust_stats, args=(cache, q))
    p.daemon = True
    p.start()
    try:
        res = q.get(timeout=30)
    except Exception:
        res = {"error": "child did not answer (hang dropping the inherited runtime)"}
    p.join(timeout=10)
    alive = p.is_alive()
    if alive:
        p.kill()
    assert "error" not in res, res
    assert res["pid_ok"]
    assert res["inflight"] == 0
    assert not alive, "child did not exit (hang at interpreter shutdown)"
    # The parent still owns a working runtime.
    assert cache._pid == os.getpid()
    assert cache.stats()["inflight"] == 0


@pytest.mark.skipif(not has_rust_backend(), reason="Rust backend not built")
def test_rust_backend_stats_and_path_for(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    cache = SharedDiskCache(tmp_path / "cache", max_bytes=1_000, download_threads=2)
    assert cache._use_rust
    s = cache.stats()
    assert set(s) == {
        "attempts",
        "retries",
        "throttled",
        "failures",
        "not_found",
        "downloaded",
        "bytes",
        "inflight",
    }
    assert all(v == 0 for v in s.values())
    # Rust and Python agree on where a URI lives on disk.
    uri = "s3://b/some/key.jpg"
    assert cache.path_for(uri) == cache._rust_cache.path_for(uri)
    # Pickling drops the Rust runtime and rebuilds it lazily.
    clone = pickle.loads(pickle.dumps(cache))
    assert clone.stats()["attempts"] == 0
    assert clone.path_for(uri) == cache.path_for(uri)


def test_ensure_local_does_not_deadlock_when_download_finishes_before_callback(
    tmp_path,
):
    """Regression: a download that completes before add_done_callback runs made the
    cleanup callback run inline on the calling thread, which still held the
    in-flight lock (self-deadlock). Seen as a hung DataLoader worker."""
    fake = FakeS3Client({("b", "k1"): b"x", ("b", "k2"): b"y"})
    cache = make_cache(tmp_path, fake)
    real_submit = cache._pool.submit

    def submit_and_finish(fn, *a, **kw):
        fut = real_submit(fn, *a, **kw)
        fut.result()  # completed before the caller can register its callback
        return fut

    cache._pool.submit = submit_and_finish  # type: ignore[method-assign]
    done = {}

    def run():
        done["ensure"] = Path(cache.ensure_local("s3://b/k1")).read_bytes()
        cache.prime_prefetch(["s3://b/k2"])
        done["prime"] = True

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "ensure_local/prime_prefetch deadlocked"
    assert done == {"ensure": b"x", "prime": True}
    assert cache.stats()["inflight"] == 0


def _child_take_lock(cache, uri, q):
    try:
        with cache._lock_for(uri):
            q.put("acquired")
    except Exception as e:  # pragma: no cover
        q.put(repr(e))


def test_object_lock_held_across_fork_is_released_for_the_child(tmp_path):
    """The parent holds a per-object lock while forking (a download in flight when
    DataLoader starts its workers). The child inherits the lock's descriptor but
    must still get the lock once the parent releases it."""
    fake = FakeS3Client({})
    cache = make_cache(tmp_path, fake)
    uri = "s3://b/k1"
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    lock = cache._lock_for(uri)
    with lock:
        p = ctx.Process(target=_child_take_lock, args=(cache, uri, q))
        p.daemon = True
        p.start()
        time.sleep(0.3)
        assert q.empty()  # child is blocked while the parent holds the lock
    try:
        res = q.get(timeout=10)
    except Exception:
        res = "child never acquired the lock"
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
    assert res == "acquired"


# --------------------------------------------------------------------------- #
# Rust extension API used by the Python side (no S3 needed).
# --------------------------------------------------------------------------- #


def _rust_cache(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    from planfetch.planfetch_rs import RustS3Cache

    return RustS3Cache(str(tmp_path / "cache"), 4, **kw)


@pytest.mark.skipif(not has_rust_backend(), reason="Rust backend not built")
def test_rust_wait_for_path_times_out_then_returns_when_the_file_appears(
    tmp_path, monkeypatch
):
    rc = _rust_cache(tmp_path, monkeypatch)
    uri = "s3://b/waited"
    p = Path(rc.path_for(uri))
    t0 = time.perf_counter()
    assert rc.wait_for_path(uri, 50) is False
    assert 0.04 < time.perf_counter() - t0 < 1.0

    def writer():
        time.sleep(0.1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    threading.Thread(target=writer, daemon=True).start()
    t0 = time.perf_counter()
    assert rc.wait_for_path(uri, 5000) is True
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.skipif(not has_rust_backend(), reason="Rust backend not built")
def test_rust_drop_paths_unlinks_in_the_background_and_wait_covers_it(
    tmp_path, monkeypatch
):
    rc = _rust_cache(tmp_path, monkeypatch)
    uris = [f"s3://b/k{i}" for i in range(50)]
    paths = [Path(rc.path_for(u)) for u in uris]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    rc.drop_paths(uris + ["s3://b/never-existed"])  # missing files are ignored
    rc.wait_for_inflight()  # also waits for pending unlinks
    assert not any(p.exists() for p in paths)
    rc.drop_paths(uris)  # dropping again is a no-op, not an error
    rc.wait_for_inflight()


@pytest.mark.skipif(not has_rust_backend(), reason="Rust backend not built")
def test_rust_stats_dump_writes_json_with_pid(tmp_path, monkeypatch):
    rc = _rust_cache(tmp_path, monkeypatch)
    out = tmp_path / "stats.json"
    rc.stats_dump(str(out))
    d = json.loads(out.read_text())
    assert d["pid"] == os.getpid()
    assert d["attempts"] == 0 and d["not_found"] == 0 and d["inflight"] == 0
    assert set(d) == set(rc.stats()) | {"pid"}
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


@pytest.mark.skipif(not has_rust_backend(), reason="Rust backend not built")
def test_rust_worker_threads_argument(tmp_path, monkeypatch):
    rc = _rust_cache(tmp_path, monkeypatch)
    assert rc.worker_threads is None
    small = _rust_cache(tmp_path, monkeypatch, worker_threads=1)
    assert small.worker_threads == 1
    # A 1-thread runtime still serves requests (a local miss on a fake bucket
    # needs the network, so only exercise the non-network calls here).
    assert small.wait_for_path("s3://b/none", 10) is False
    small.drop_paths(["s3://b/none"])
    small.wait_for_inflight()

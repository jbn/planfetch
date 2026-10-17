"""Terminal S3 errors, cross-process stats, worker wait-not-download, async drop.

These cover the Python side of the 2026-10-17 real-S3 findings: a missing key
must fail fast and be skippable, the parent must be able to see every
process's counters, forked workers should wait for the parent's prefetch
instead of downloading with a full runtime of their own, and per-step
eviction should leave the training thread.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest
from botocore.exceptions import ClientError

import planfetch as pf
from planfetch.cache import (
    S3AccessDenied,
    S3ObjectMissing,
    S3TerminalError,
    SharedDiskCache,
    has_rust_backend,
)
from planfetch.dataloader import PrefetchCoordinator, build_prefetching_dataloader
from planfetch.dataset import S3PathDataset
from planfetch.plans import PlanView, SequentialPlanProvider


class Body:
    def __init__(self, data: bytes):
        self._data = data

    def iter_chunks(self, chunk_size: int = 1024):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "GetObject",
    )


class ErrorS3Client:
    """Objects by (bucket, key); `errors` maps a key to the ClientError to raise.
    `flaky` keys fail with SlowDown once, then succeed. Picklable."""

    def __init__(self, objects: Dict[Tuple[str, str], bytes], errors=None, flaky=()):
        self.objects = dict(objects)
        self.errors = dict(errors or {})
        self.flaky = set(flaky)
        self.calls = []
        self.seen_flaky = set()

    def get_object(self, Bucket: str, Key: str) -> dict:
        self.calls.append((Bucket, Key))
        if (Bucket, Key) in self.errors:
            raise self.errors[(Bucket, Key)]
        if (Bucket, Key) in self.flaky and (Bucket, Key) not in self.seen_flaky:
            self.seen_flaky.add((Bucket, Key))
            raise _client_error("SlowDown", 503)
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", 404)
        return {"Body": Body(self.objects[(Bucket, Key)])}


def make_cache(tmp_path, client, **kw):
    kw.setdefault("max_bytes", 10_000_000)
    kw.setdefault("download_threads", 4)
    kw.setdefault("min_age_seconds_before_delete", 0.0)
    kw.setdefault("retry_backoff_seconds", 0.01)
    return SharedDiskCache(tmp_path / "cache", s3_client=client, **kw)


# --------------------------------------------------------------------------- #
# Terminal errors: no retry, typed exception, counted.
# --------------------------------------------------------------------------- #


def test_missing_key_raises_typed_error_on_first_attempt(tmp_path):
    client = ErrorS3Client({("b", "ok"): b"x"})
    cache = make_cache(tmp_path, client)
    t0 = time.perf_counter()
    with pytest.raises(S3ObjectMissing) as ei:
        cache.ensure_local("s3://b/missing")
    assert time.perf_counter() - t0 < 0.5
    assert client.calls == [("b", "missing")]  # one attempt, no retry
    assert ei.value.uri == "s3://b/missing"
    assert ei.value.code == "NoSuchKey"
    assert ei.value.status == 404
    assert isinstance(ei.value, S3TerminalError)
    s = cache.stats()
    assert s["attempts"] == 1
    assert s["retries"] == 0
    assert s["failures"] == 1
    assert s["not_found"] == 1
    assert s["inflight"] == 0
    # The cache still works.
    assert Path(cache.ensure_local("s3://b/ok")).read_bytes() == b"x"


def test_access_denied_raises_typed_error_without_retry(tmp_path):
    client = ErrorS3Client({}, errors={("b", "k"): _client_error("AccessDenied", 403)})
    cache = make_cache(tmp_path, client)
    with pytest.raises(S3AccessDenied) as ei:
        cache.ensure_local("s3://b/k")
    assert len(client.calls) == 1
    assert ei.value.code == "AccessDenied"
    s = cache.stats()
    assert s["failures"] == 1 and s["not_found"] == 0 and s["retries"] == 0


def test_other_4xx_is_terminal_but_throttling_still_retries(tmp_path):
    client = ErrorS3Client(
        {("b", "flaky"): b"y"},
        errors={("b", "bad"): _client_error("InvalidRequest", 400)},
        flaky=[("b", "flaky")],
    )
    cache = make_cache(tmp_path, client)
    with pytest.raises(S3TerminalError) as ei:
        cache.ensure_local("s3://b/bad")
    assert not isinstance(ei.value, (S3ObjectMissing, S3AccessDenied))
    assert client.calls.count(("b", "bad")) == 1
    # SlowDown once, then success: counted as throttled + retry, not failure.
    assert Path(cache.ensure_local("s3://b/flaky")).read_bytes() == b"y"
    s = cache.stats()
    assert s["throttled"] == 1 and s["retries"] == 1
    assert s["failures"] == 1 and s["downloaded"] == 1


def test_prime_prefetch_swallows_terminal_errors_but_counts_them(tmp_path):
    client = ErrorS3Client({("b", "ok"): b"x"})
    cache = make_cache(tmp_path, client)
    cache.prime_prefetch(["s3://b/ok", "s3://b/gone"])
    cache.wait_for_inflight()
    s = cache.stats()
    assert s["not_found"] == 1 and s["downloaded"] == 1 and s["inflight"] == 0


def test_terminal_error_classes_are_exported():
    assert pf.S3ObjectMissing is S3ObjectMissing
    assert pf.S3AccessDenied is S3AccessDenied
    assert pf.S3TerminalError is S3TerminalError


# --------------------------------------------------------------------------- #
# Stats across processes.
# --------------------------------------------------------------------------- #


def test_stats_all_processes_sums_dumps_from_other_pids(tmp_path):
    client = ErrorS3Client({("b", "k"): b"x"})
    cache = make_cache(tmp_path, client)
    cache.ensure_local("s3://b/k")
    other = {k: 0 for k in cache.stats()}
    other.update(
        {
            "attempts": 5,
            "downloaded": 4,
            "not_found": 1,
            "failures": 1,
            "bytes": 40,
            "pid": 4242,
        }
    )
    (tmp_path / "cache" / "_stats").mkdir(exist_ok=True)
    (tmp_path / "cache" / "_stats" / "4242.json").write_text(json.dumps(other))
    # A stale dump for *this* pid must not double count the live counters.
    mine = dict(other)
    mine.update({"pid": os.getpid(), "attempts": 999})
    (tmp_path / "cache" / "_stats" / f"{os.getpid()}.json").write_text(json.dumps(mine))

    local = cache.stats()
    total = cache.stats(all_processes=True)
    assert local["attempts"] == 1 and local["downloaded"] == 1
    assert total["attempts"] == 6
    assert total["downloaded"] == 5
    assert total["not_found"] == 1 and total["failures"] == 1
    assert total["bytes"] == local["bytes"] + 40
    assert total["processes"] == 2
    assert "pid" not in total


def test_dump_stats_writes_this_process_file_and_scan_prune_ignores_it(tmp_path):
    client = ErrorS3Client({("b", "k"): b"x"})
    cache = make_cache(tmp_path, client)
    cache.ensure_local("s3://b/k")
    p = cache.dump_stats()
    assert Path(p) == tmp_path / "cache" / "_stats" / f"{os.getpid()}.json"
    d = json.loads(Path(p).read_text())
    assert d["pid"] == os.getpid() and d["downloaded"] == 1
    cache.prune(keep_uris=[], mode="window", backend="scan")
    assert Path(p).exists()


def test_constructor_clears_stale_stats_dir(tmp_path):
    d = tmp_path / "cache" / "_stats"
    d.mkdir(parents=True)
    (d / "1.json").write_text("{}")
    make_cache(tmp_path, ErrorS3Client({}))
    assert not list(d.iterdir())


def test_download_in_a_worker_leaves_a_stats_dump_without_atexit(tmp_path):
    """DataLoader workers exit with os._exit, so dumps must happen on download."""
    client = ErrorS3Client({("b", "k"): b"x"})
    cache = make_cache(tmp_path, client, stats_dump_interval=0.0)
    ctx = multiprocessing.get_context("fork")

    def child():
        cache.ensure_local("s3://b/k")
        os._exit(0)

    p = ctx.Process(target=child)
    p.start()
    p.join(20)
    assert p.exitcode == 0
    dumps = list((tmp_path / "cache" / "_stats").glob("*.json"))
    assert [json.loads(f.read_text())["pid"] for f in dumps] == [p.pid]
    assert cache.stats(all_processes=True)["downloaded"] == 1


# --------------------------------------------------------------------------- #
# Workers wait for the parent's prefetch instead of downloading.
# --------------------------------------------------------------------------- #


def test_wait_local_returns_when_file_appears_without_downloading(tmp_path):
    client = ErrorS3Client({("b", "k"): b"x"})
    cache = make_cache(tmp_path, client)
    uri = "s3://b/k"
    assert cache.wait_local(uri, timeout=0.05) is False
    dst = Path(cache.path_for(uri))

    def writer():
        time.sleep(0.1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")

    threading.Thread(target=writer, daemon=True).start()
    assert cache.wait_local(uri, timeout=5.0) is True
    assert client.calls == []  # nothing downloaded


def test_dataset_waits_for_prefetched_file_then_falls_back_to_download(tmp_path):
    client = ErrorS3Client({("b", "k0"): b"a", ("b", "k1"): b"b"})
    cache = make_cache(tmp_path, client)
    ds = S3PathDataset(
        ["s3://b/k0", "s3://b/k1"],
        [0, 1],
        cache=cache,
        image_loader=lambda p: Path(p).read_bytes(),
        miss_wait_seconds=0.3,
    )
    dst = Path(cache.path_for("s3://b/k0"))

    def writer():
        time.sleep(0.1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"a")

    threading.Thread(target=writer, daemon=True).start()
    assert ds[0] == (b"a", 0)
    assert client.calls == []  # waited for the "parent", did not download
    t0 = time.perf_counter()
    assert ds[1] == (b"b", 1)  # nobody writes k1: wait times out, then download
    assert 0.3 <= time.perf_counter() - t0 < 2.0
    assert client.calls == [("b", "k1")]


def test_dataset_default_does_not_wait(tmp_path):
    client = ErrorS3Client({("b", "k"): b"a"})
    cache = make_cache(tmp_path, client)
    ds = S3PathDataset(["s3://b/k"], [0], cache=cache, image_loader=lambda p: 1)
    t0 = time.perf_counter()
    assert ds[0] == (1, 0)
    assert time.perf_counter() - t0 < 0.2
    assert client.calls == [("b", "k")]


def test_dataset_on_error_substitutes_another_index(tmp_path):
    client = ErrorS3Client({("b", "k0"): b"a", ("b", "k2"): b"c"})
    cache = make_cache(tmp_path, client)
    seen = []

    def on_error(idx, exc):
        seen.append((idx, type(exc).__name__))
        return (idx + 1) % 3

    ds = S3PathDataset(
        ["s3://b/k0", "s3://b/k1", "s3://b/k2"],
        [0, 1, 2],
        cache=cache,
        image_loader=lambda p: Path(p).read_bytes(),
        on_error=on_error,
    )
    assert ds[1] == (b"c", 2)
    assert seen == [(1, "S3ObjectMissing")]


def test_dataset_without_on_error_raises_typed_error(tmp_path):
    cache = make_cache(tmp_path, ErrorS3Client({}))
    ds = S3PathDataset(["s3://b/k"], [0], cache=cache, image_loader=lambda p: 1)
    with pytest.raises(S3ObjectMissing):
        ds[0]


def test_dataset_on_error_gives_up_after_a_bounded_number_of_hops(tmp_path):
    cache = make_cache(tmp_path, ErrorS3Client({}))
    ds = S3PathDataset(
        [f"s3://b/k{i}" for i in range(3)],
        [0, 1, 2],
        cache=cache,
        image_loader=lambda p: 1,
        on_error=lambda idx, exc: (idx + 1) % 3,  # every candidate is missing
    )
    with pytest.raises(S3TerminalError):
        ds[0]


def _child_backend_report(cache, q):
    try:
        cache.ensure_local("s3://b/k")
        info = {"pid_ok": cache._pid == os.getpid()}
        if cache._use_rust:
            info["threads"] = cache._rust_cache.worker_threads
        else:
            info["threads"] = cache._pool._max_workers
        q.put(info)
    except Exception as e:  # pragma: no cover
        q.put({"error": repr(e)})


def test_forked_worker_builds_a_smaller_backend(tmp_path):
    client = ErrorS3Client({("b", "k"): b"x"})
    cache = make_cache(tmp_path, client, download_threads=32, worker_download_threads=3)
    cache.stats()
    assert cache._pool._max_workers == 32
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_backend_report, args=(cache, q))
    p.daemon = True
    p.start()
    res = q.get(timeout=20)
    p.join(5)
    assert res == {"pid_ok": True, "threads": 3}


@pytest.mark.skipif(not has_rust_backend(), reason="Rust backend not built")
def test_forked_worker_gets_small_rust_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    cache = SharedDiskCache(
        tmp_path / "cache",
        max_bytes=1_000,
        download_threads=8,
        worker_runtime_threads=2,
    )
    assert cache._use_rust
    cache.stats()
    assert cache._rust_cache.worker_threads is None  # the parent: tokio default
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()

    def child():
        cache.stats()  # rebuild in the child
        q.put(cache._rust_cache.worker_threads)

    p = ctx.Process(target=child)
    p.daemon = True
    p.start()
    assert q.get(timeout=30) == 2
    p.join(5)


# --------------------------------------------------------------------------- #
# Eviction off the training thread.
# --------------------------------------------------------------------------- #


def test_drop_async_unlinks_and_wait_for_inflight_covers_it(tmp_path):
    client = ErrorS3Client({("b", f"k{i}"): b"x" for i in range(20)})
    cache = make_cache(tmp_path, client)
    uris = [f"s3://b/k{i}" for i in range(20)]
    paths = [Path(cache.ensure_local(u)) for u in uris]
    assert all(p.exists() for p in paths)
    cache.drop_async(uris + ["s3://b/never"])
    cache.wait_for_inflight()
    assert not any(p.exists() for p in paths)
    assert cache._total_bytes == 0  # meta bookkeeping still happens


class RecordingSyncCache:
    """A PrefetchCache with only the synchronous drop()."""

    def __init__(self):
        self.calls = []

    def prime_prefetch(self, s3_uris: Sequence[str]) -> None:
        self.calls.append(("prime", list(s3_uris)))

    def drop(self, s3_uris: Sequence[str]) -> int:
        self.calls.append(("drop", list(s3_uris)))
        return len(s3_uris)

    def prune(
        self, *, keep_uris: Sequence[str], mode: str = "window", backend: str = "scan"
    ) -> None:
        self.calls.append(("prune", list(keep_uris)))


class RecordingCache(RecordingSyncCache):
    """…and the asynchronous drop the coordinator should prefer."""

    def drop_async(self, s3_uris: Sequence[str]) -> None:
        self.calls.append(("drop_async", list(s3_uris)))


def _plan(n, bs):
    return PlanView(np.arange(n * bs, dtype=np.int64).reshape(n, bs))


def test_coordinator_uses_drop_async_when_the_cache_has_it():
    uris = [f"s3://b/{i}" for i in range(40)]
    cache = RecordingCache()
    c = PrefetchCoordinator(cache, uris, batches_ahead=2, batches_back=0)
    c.set_plan(_plan(10, 4))
    c.prime()
    c.on_batch_yielded(0)
    c.on_batch_yielded(1)
    kinds = [k for k, _ in cache.calls]
    assert "drop_async" in kinds and "drop" not in kinds
    assert ("drop_async", uris[0:4]) in cache.calls


def test_coordinator_falls_back_to_sync_drop():
    uris = [f"s3://b/{i}" for i in range(40)]
    cache = RecordingSyncCache()
    c = PrefetchCoordinator(cache, uris, batches_ahead=2, batches_back=0)
    c.set_plan(_plan(10, 4))
    c.prime()
    c.on_batch_yielded(1)
    assert ("drop", uris[0:4]) in cache.calls


def test_build_prefetching_dataloader_sets_miss_wait_on_the_dataset(tmp_path):
    client = ErrorS3Client({("b", f"k{i}"): b"x" for i in range(8)})
    cache = make_cache(tmp_path, client)
    uris = [f"s3://b/k{i}" for i in range(8)]
    ds = S3PathDataset(uris, [0] * 8, cache=cache, image_loader=lambda p: 1)
    assert ds.miss_wait_seconds == 0.0
    build_prefetching_dataloader(
        dataset=ds,
        s3_uris=uris,
        plan_provider=SequentialPlanProvider(8, 2),
        cache=cache,
        num_workers=0,
        batches_ahead=2,
        batches_back=0,
        pin_memory=False,
    )
    assert ds.miss_wait_seconds == 30.0
    # An explicit value on the dataset is left alone.
    ds2 = S3PathDataset(
        uris, [0] * 8, cache=cache, image_loader=lambda p: 1, miss_wait_seconds=1.5
    )
    build_prefetching_dataloader(
        dataset=ds2,
        s3_uris=uris,
        plan_provider=SequentialPlanProvider(8, 2),
        cache=cache,
        num_workers=0,
        batches_ahead=2,
        batches_back=0,
        pin_memory=False,
    )
    assert ds2.miss_wait_seconds == 1.5

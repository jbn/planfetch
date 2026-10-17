"""Tests for the planfetch speedtest CLI."""

import threading
from typing import Dict, Tuple

import pytest
from click.testing import CliRunner

import planfetch.cli as cli_mod
from planfetch.cache import SharedDiskCache
from planfetch.cli import cli


class FakeStreamingBody:
    def __init__(self, data: bytes):
        self._data = data

    def iter_chunks(self, chunk_size: int = 1024):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


class FakeS3Client:
    def __init__(self, objects: Dict[Tuple[str, str], bytes]):
        self.objects = dict(objects)
        self.calls = []
        self.lock = threading.Lock()

    def get_object(self, Bucket: str, Key: str) -> dict:
        with self.lock:
            self.calls.append((Bucket, Key))
        return {"Body": FakeStreamingBody(self.objects[(Bucket, Key)])}


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "index.csv"
    lines = ["s3_uri,label"] + [f"s3://b/k{i},{i % 2}" for i in range(12)]
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture
def patched_cache(monkeypatch):
    """Route the CLI's SharedDiskCache to a fake S3 and record constructor kwargs."""
    fake = FakeS3Client({("b", f"k{i}"): bytes([i]) * 100 for i in range(12)})
    seen = {}

    def factory(cache_dir, **kw):
        seen.update(kw)
        kw = dict(kw)
        kw.pop("use_rust", None)
        kw.pop("s3_max_pool_connections", None)
        return SharedDiskCache(cache_dir, s3_client=fake, **kw)

    monkeypatch.setattr(cli_mod, "SharedDiskCache", factory)
    return fake, seen


def test_speedtest_requires_cache_dir(csv_file):
    result = CliRunner().invoke(cli, ["speedtest", str(csv_file)])
    assert result.exit_code != 0
    assert "--cache-dir" in result.output


def test_speedtest_runs_and_reports(csv_file, tmp_path, patched_cache):
    fake, seen = patched_cache
    result = CliRunner().invoke(
        cli,
        [
            "speedtest",
            str(csv_file),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--batch-size",
            "4",
            "--batches-ahead",
            "2",
            "--strategy",
            "sequential",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Samples/sec" in result.output
    assert "P99 file wait" in result.output  # per-file latency, not just per-batch
    assert "Retries" in result.output and "Throttled" in result.output
    assert "Total downloaded" in result.output
    assert "1.20 KB" in result.output or "0.00 GB" in result.output
    # every object fetched exactly once
    assert sorted(fake.calls) == sorted(("b", f"k{i}") for i in range(12))
    assert seen["use_rust"] is True


def test_speedtest_no_rust_flag(csv_file, tmp_path, patched_cache):
    _, seen = patched_cache
    result = CliRunner().invoke(
        cli,
        [
            "speedtest",
            str(csv_file),
            "--cache-dir",
            str(tmp_path / "c"),
            "--no-rust",
            "--batch-size",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["use_rust"] is False


def test_speedtest_prunes_window_incrementally(
    csv_file, tmp_path, patched_cache, monkeypatch
):
    """The CLI uses PrefetchCoordinator, so each step primes one batch, not the window."""
    import planfetch.dataloader as dl_mod

    calls = []
    orig = dl_mod.PrefetchCoordinator.on_batch_yielded

    def spy(self, b):
        n0 = len(self._counts)
        orig(self, b)
        calls.append((b, n0, len(self._counts)))

    monkeypatch.setattr(dl_mod.PrefetchCoordinator, "on_batch_yielded", spy)
    result = CliRunner().invoke(
        cli,
        [
            "speedtest",
            str(csv_file),
            "--cache-dir",
            str(tmp_path / "c"),
            "--batch-size",
            "2",
            "--batches-ahead",
            "2",
            "--batches-back",
            "0",
            "--strategy",
            "sequential",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == list(range(6))
    # the window never holds more than batches_ahead batches worth of uris
    assert all(c[2] <= 4 for c in calls)
    # only the last window [5, 6) is left on disk, everything earlier was dropped
    cache_dir = tmp_path / "c"
    left = [p for p in cache_dir.rglob("*") if p.is_file() and len(p.name) == 64]
    assert len(left) == 2

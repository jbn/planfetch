"""Comprehensive tests for planfetch.dataset module."""
import threading
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

from planfetch.cache import SharedDiskCache
from planfetch.dataset import S3PathDataset


class FakeS3Client:
    """Minimal mock for boto3 S3 client."""

    def __init__(self, objects: Dict[Tuple[str, str], bytes]):
        self.objects = dict(objects)
        self.calls = []
        self.lock = threading.Lock()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        with self.lock:
            self.calls.append((bucket, key))
        if (bucket, key) not in self.objects:
            raise Exception(f"NoSuchKey: {bucket}/{key}")
        data = self.objects[(bucket, key)]
        with open(filename, "wb") as f:
            f.write(data)


def make_cache(tmp_path, fake_s3, *, max_bytes=10_000_000, threads=4):
    return SharedDiskCache(
        tmp_path / "cache",
        max_bytes=max_bytes,
        download_threads=threads,
        s3_client=fake_s3,
        min_age_seconds_before_delete=0.0,
    )


class TestS3PathDataset:
    """Test suite for S3PathDataset."""

    def test_init(self, tmp_path):
        """Test initialization."""
        objects = {("b", "k1"): b"data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="loaded_image")

        s3_uris = ["s3://b/k1", "s3://b/k2"]
        labels = [0, 1]

        dataset = S3PathDataset(
            s3_uris=s3_uris,
            labels=labels,
            cache=cache,
            image_loader=loader,
            transform=None,
        )

        assert dataset.s3_uris == s3_uris
        np.testing.assert_array_equal(dataset.labels, np.array([0, 1], dtype=np.int64))
        assert dataset.cache is cache
        assert dataset.image_loader is loader
        assert dataset.transform is None

    def test_len(self, tmp_path):
        """Test __len__ method."""
        fake = FakeS3Client({})
        cache = make_cache(tmp_path, fake)
        loader = Mock()

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1", "s3://b/k2", "s3://b/k3"],
            labels=[0, 1, 2],
            cache=cache,
            image_loader=loader,
        )

        assert len(dataset) == 3

    def test_len_empty(self, tmp_path):
        """Test __len__ with empty dataset."""
        fake = FakeS3Client({})
        cache = make_cache(tmp_path, fake)
        loader = Mock()

        dataset = S3PathDataset(
            s3_uris=[],
            labels=[],
            cache=cache,
            image_loader=loader,
        )

        assert len(dataset) == 0

    def test_getitem_basic(self, tmp_path):
        """Test __getitem__ returns (image, label) tuple."""
        objects = {("b", "k1"): b"image_data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="loaded_image")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[42],
            cache=cache,
            image_loader=loader,
        )

        img, label = dataset[0]

        assert img == "loaded_image"
        assert label == 42
        loader.assert_called_once()

    def test_getitem_calls_cache(self, tmp_path):
        """Test __getitem__ calls cache.ensure_local with correct URI."""
        objects = {("b", "k1"): b"data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="img")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[0],
            cache=cache,
            image_loader=loader,
        )

        dataset[0]

        # The loader should be called with the local path from cache
        assert loader.call_count == 1
        called_path = loader.call_args[0][0]
        assert Path(called_path).exists()

    def test_getitem_with_transform(self, tmp_path):
        """Test __getitem__ applies transform."""
        objects = {("b", "k1"): b"data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="raw_image")
        transform = Mock(return_value="transformed_image")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[0],
            cache=cache,
            image_loader=loader,
            transform=transform,
        )

        img, label = dataset[0]

        assert img == "transformed_image"
        transform.assert_called_once_with("raw_image")

    def test_getitem_without_transform(self, tmp_path):
        """Test __getitem__ without transform."""
        objects = {("b", "k1"): b"data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="raw_image")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[0],
            cache=cache,
            image_loader=loader,
            transform=None,
        )

        img, label = dataset[0]

        assert img == "raw_image"

    def test_getitem_multiple_indices(self, tmp_path):
        """Test __getitem__ with multiple indices."""
        objects = {
            ("b", "k0"): b"data0",
            ("b", "k1"): b"data1",
            ("b", "k2"): b"data2",
        }
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)

        call_count = 0

        def loader(path):
            nonlocal call_count
            call_count += 1
            return f"img_{call_count}"

        dataset = S3PathDataset(
            s3_uris=["s3://b/k0", "s3://b/k1", "s3://b/k2"],
            labels=[10, 20, 30],
            cache=cache,
            image_loader=loader,
        )

        img0, label0 = dataset[0]
        img1, label1 = dataset[1]
        img2, label2 = dataset[2]

        assert label0 == 10
        assert label1 == 20
        assert label2 == 30

    def test_getitem_with_numpy_index(self, tmp_path):
        """Test __getitem__ with numpy integer index."""
        objects = {("b", "k1"): b"data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="img")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[42],
            cache=cache,
            image_loader=loader,
        )

        img, label = dataset[np.int64(0)]

        assert label == 42

    def test_labels_converted_to_int64(self, tmp_path):
        """Test that labels are stored as int64."""
        fake = FakeS3Client({})
        cache = make_cache(tmp_path, fake)
        loader = Mock()

        # Pass labels as regular Python ints
        dataset = S3PathDataset(
            s3_uris=["s3://b/k1", "s3://b/k2"],
            labels=[1, 2],
            cache=cache,
            image_loader=loader,
        )

        assert dataset.labels.dtype == np.int64

    def test_s3_uris_converted_to_list(self, tmp_path):
        """Test that s3_uris are stored as list."""
        fake = FakeS3Client({})
        cache = make_cache(tmp_path, fake)
        loader = Mock()

        # Pass s3_uris as tuple
        dataset = S3PathDataset(
            s3_uris=("s3://b/k1", "s3://b/k2"),
            labels=[1, 2],
            cache=cache,
            image_loader=loader,
        )

        assert isinstance(dataset.s3_uris, list)

    def test_getitem_returns_int_label(self, tmp_path):
        """Test that __getitem__ returns Python int for label."""
        objects = {("b", "k1"): b"data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="img")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[42],
            cache=cache,
            image_loader=loader,
        )

        img, label = dataset[0]

        # Should be a Python int, not numpy int64
        assert isinstance(label, int)
        assert label == 42

    def test_caching_behavior(self, tmp_path):
        """Test that cache is used for subsequent accesses."""
        objects = {("b", "k1"): b"data"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="img")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[0],
            cache=cache,
            image_loader=loader,
        )

        # Access the same item twice
        dataset[0]
        dataset[0]

        # Cache should have only downloaded once
        assert len(fake.calls) == 1

    def test_different_items_different_downloads(self, tmp_path):
        """Test that different items trigger different downloads."""
        objects = {
            ("b", "k0"): b"data0",
            ("b", "k1"): b"data1",
        }
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)
        loader = Mock(return_value="img")

        dataset = S3PathDataset(
            s3_uris=["s3://b/k0", "s3://b/k1"],
            labels=[0, 1],
            cache=cache,
            image_loader=loader,
        )

        dataset[0]
        dataset[1]

        assert len(fake.calls) == 2

    def test_image_loader_receives_correct_path(self, tmp_path):
        """Test that image_loader receives the correct local path."""
        objects = {("b", "k1"): b"image_content"}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake)

        received_paths = []

        def loader(path):
            received_paths.append(path)
            return "img"

        dataset = S3PathDataset(
            s3_uris=["s3://b/k1"],
            labels=[0],
            cache=cache,
            image_loader=loader,
        )

        dataset[0]

        assert len(received_paths) == 1
        # The path should be a valid file path
        assert Path(received_paths[0]).exists()
        # The file should contain the original content
        assert Path(received_paths[0]).read_bytes() == b"image_content"

    def test_with_numpy_labels(self, tmp_path):
        """Test initialization with numpy array labels."""
        fake = FakeS3Client({})
        cache = make_cache(tmp_path, fake)
        loader = Mock()

        labels = np.array([10, 20, 30], dtype=np.int32)  # Different dtype

        dataset = S3PathDataset(
            s3_uris=["s3://b/k0", "s3://b/k1", "s3://b/k2"],
            labels=labels,
            cache=cache,
            image_loader=loader,
        )

        # Should be converted to int64
        assert dataset.labels.dtype == np.int64
        np.testing.assert_array_equal(dataset.labels, [10, 20, 30])

    def test_concurrent_access(self, tmp_path):
        """Test concurrent access to dataset."""
        objects = {(f"b", f"k{i}"): f"data{i}".encode() for i in range(10)}
        fake = FakeS3Client(objects)
        cache = make_cache(tmp_path, fake, threads=8)
        loader = Mock(return_value="img")

        dataset = S3PathDataset(
            s3_uris=[f"s3://b/k{i}" for i in range(10)],
            labels=list(range(10)),
            cache=cache,
            image_loader=loader,
        )

        results = []
        errors = []

        def worker(idx):
            try:
                img, label = dataset[idx]
                results.append((idx, label))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

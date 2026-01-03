"""Comprehensive tests for planfetch.uri_storage module."""
import numpy as np
import pytest

from planfetch.uri_storage import S3KeyList


class TestS3KeyList:
    """Test suite for S3KeyList class."""

    def test_from_uris_basic(self):
        """Test basic S3KeyList creation from URIs."""
        uris = [
            "s3://mybucket/path/to/file1.jpg",
            "s3://mybucket/path/to/file2.jpg",
            "s3://mybucket/other/file3.png",
        ]
        keys = S3KeyList.from_uris(uris)

        assert keys.bucket == "mybucket"
        assert len(keys) == 3

    def test_getitem_reconstructs_uri(self):
        """Test that __getitem__ reconstructs full URIs correctly."""
        uris = [
            "s3://bucket/a.jpg",
            "s3://bucket/folder/b.png",
            "s3://bucket/deep/nested/path/c.gif",
        ]
        keys = S3KeyList.from_uris(uris)

        assert keys[0] == "s3://bucket/a.jpg"
        assert keys[1] == "s3://bucket/folder/b.png"
        assert keys[2] == "s3://bucket/deep/nested/path/c.gif"

    def test_negative_indexing(self):
        """Test negative indexing works."""
        uris = ["s3://b/a.jpg", "s3://b/b.jpg", "s3://b/c.jpg"]
        keys = S3KeyList.from_uris(uris)

        assert keys[-1] == "s3://b/c.jpg"
        assert keys[-2] == "s3://b/b.jpg"
        assert keys[-3] == "s3://b/a.jpg"

    def test_index_out_of_range(self):
        """Test IndexError for out-of-range access."""
        uris = ["s3://b/a.jpg", "s3://b/b.jpg"]
        keys = S3KeyList.from_uris(uris)

        with pytest.raises(IndexError):
            _ = keys[2]

        with pytest.raises(IndexError):
            _ = keys[-3]

    def test_get_key(self):
        """Test get_key returns just the key without bucket."""
        uris = [
            "s3://mybucket/path/to/file.jpg",
            "s3://mybucket/other.png",
        ]
        keys = S3KeyList.from_uris(uris)

        assert keys.get_key(0) == "path/to/file.jpg"
        assert keys.get_key(1) == "other.png"

    def test_get_bucket_and_key(self):
        """Test get_bucket_and_key returns tuple."""
        uris = ["s3://mybucket/path/file.jpg"]
        keys = S3KeyList.from_uris(uris)

        bucket, key = keys.get_bucket_and_key(0)
        assert bucket == "mybucket"
        assert key == "path/file.jpg"

    def test_iteration(self):
        """Test iterating over S3KeyList yields full URIs."""
        uris = [
            "s3://bucket/a.jpg",
            "s3://bucket/b.jpg",
            "s3://bucket/c.jpg",
        ]
        keys = S3KeyList.from_uris(uris)

        result = list(keys)
        assert result == uris

    def test_to_list(self):
        """Test to_list converts back to full URI list."""
        uris = [
            "s3://bucket/a.jpg",
            "s3://bucket/b.jpg",
        ]
        keys = S3KeyList.from_uris(uris)

        result = keys.to_list()
        assert result == uris
        assert isinstance(result, list)

    def test_len(self):
        """Test __len__ returns correct count."""
        uris = ["s3://b/a", "s3://b/b", "s3://b/c", "s3://b/d"]
        keys = S3KeyList.from_uris(uris)
        assert len(keys) == 4

    def test_empty_raises(self):
        """Test that empty URI list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            S3KeyList.from_uris([])

    def test_mixed_buckets_raises(self):
        """Test that mixed buckets raises ValueError."""
        uris = [
            "s3://bucket1/file.jpg",
            "s3://bucket2/file.jpg",
        ]
        with pytest.raises(ValueError, match="Mixed buckets"):
            S3KeyList.from_uris(uris)

    def test_invalid_uri_no_prefix(self):
        """Test invalid URI without s3:// prefix."""
        uris = ["http://bucket/file.jpg"]
        with pytest.raises(ValueError, match="must start with s3://"):
            S3KeyList.from_uris(uris)

    def test_invalid_uri_no_key(self):
        """Test invalid URI with no key after bucket."""
        uris = ["s3://bucket"]
        with pytest.raises(ValueError, match="no key after bucket"):
            S3KeyList.from_uris(uris)

    def test_second_uri_invalid(self):
        """Test error when second URI is invalid."""
        uris = [
            "s3://bucket/valid.jpg",
            "http://invalid/path",
        ]
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            S3KeyList.from_uris(uris)

    def test_unicode_keys(self):
        """Test keys with unicode characters."""
        uris = [
            "s3://bucket/path/文件.jpg",
            "s3://bucket/émoji/🎉.png",
        ]
        keys = S3KeyList.from_uris(uris)

        assert keys[0] == "s3://bucket/path/文件.jpg"
        assert keys[1] == "s3://bucket/émoji/🎉.png"
        assert keys.get_key(0) == "path/文件.jpg"

    def test_keys_with_special_characters(self):
        """Test keys with spaces and special characters."""
        uris = [
            "s3://bucket/path with spaces/file.jpg",
            "s3://bucket/path-with-dashes/file_underscore.jpg",
            "s3://bucket/path.with.dots/file.jpg",
        ]
        keys = S3KeyList.from_uris(uris)

        assert keys[0] == "s3://bucket/path with spaces/file.jpg"
        assert keys.get_key(1) == "path-with-dashes/file_underscore.jpg"

    def test_single_uri(self):
        """Test with single URI."""
        uris = ["s3://mybucket/single.jpg"]
        keys = S3KeyList.from_uris(uris)

        assert len(keys) == 1
        assert keys[0] == "s3://mybucket/single.jpg"
        assert keys.bucket == "mybucket"

    def test_many_uris(self):
        """Test with many URIs for memory efficiency."""
        n = 10000
        uris = [f"s3://bucket/file{i}.jpg" for i in range(n)]
        keys = S3KeyList.from_uris(uris)

        assert len(keys) == n
        assert keys[0] == "s3://bucket/file0.jpg"
        assert keys[n - 1] == f"s3://bucket/file{n - 1}.jpg"
        assert keys[-1] == f"s3://bucket/file{n - 1}.jpg"

    def test_repr(self):
        """Test string representation."""
        uris = ["s3://mybucket/a.jpg", "s3://mybucket/b.jpg"]
        keys = S3KeyList.from_uris(uris)

        r = repr(keys)
        assert "mybucket" in r
        assert "n=2" in r

    def test_packed_storage_structure(self):
        """Test internal storage structure is efficient."""
        uris = [
            "s3://bucket/a.jpg",
            "s3://bucket/bb.jpg",
            "s3://bucket/ccc.jpg",
        ]
        keys = S3KeyList.from_uris(uris)

        # Verify internal structure
        assert isinstance(keys._data, bytes)
        assert isinstance(keys._offsets, np.ndarray)
        assert keys._offsets.dtype == np.int64
        assert len(keys._offsets) == len(uris) + 1  # n+1 offsets

        # Verify data is packed correctly
        assert keys._data == b"a.jpgbb.jpgccc.jpg"
        np.testing.assert_array_equal(keys._offsets, [0, 5, 11, 18])

    def test_slotted_class(self):
        """Test that S3KeyList uses __slots__ for memory efficiency."""
        assert hasattr(S3KeyList, "__slots__")
        assert "bucket" in S3KeyList.__slots__
        assert "_data" in S3KeyList.__slots__
        assert "_offsets" in S3KeyList.__slots__


class TestS3KeyListWithIO:
    """Integration tests with io module."""

    def test_load_index_csv_compact(self, tmp_path):
        """Test load_index_csv_compact returns S3KeyList."""
        from planfetch.io import load_index_csv_compact

        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://bucket/a.jpg,0\n"
            "s3://bucket/b.jpg,1\n"
            "s3://bucket/c.jpg,2\n"
        )

        keys, labels = load_index_csv_compact(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert isinstance(keys, S3KeyList)
        assert len(keys) == 3
        assert keys.bucket == "bucket"
        assert keys[0] == "s3://bucket/a.jpg"
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_load_index_csv_compact_mixed_buckets_raises(self, tmp_path):
        """Test load_index_csv_compact raises for mixed buckets."""
        from planfetch.io import load_index_csv_compact

        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://bucket1/a.jpg,0\n"
            "s3://bucket2/b.jpg,1\n"
        )

        with pytest.raises(ValueError, match="Mixed buckets"):
            load_index_csv_compact(csv_path, s3_uri_col="s3_path", class_col="class_id")

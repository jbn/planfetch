"""Comprehensive tests for planfetch.io module."""
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from planfetch.io import load_index_csv, load_index_dataframe
from planfetch.uri_storage import S3KeyList


class TestLoadIndexCsv:
    """Test suite for load_index_csv function."""

    def test_basic_functionality(self, tmp_path):
        """Test basic CSV loading."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id,other_col\n"
            "s3://bucket/key1,0,foo\n"
            "s3://bucket/key2,1,bar\n"
            "s3://bucket/key3,2,baz\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == [
            "s3://bucket/key1",
            "s3://bucket/key2",
            "s3://bucket/key3",
        ]
        np.testing.assert_array_equal(labels, np.array([0, 1, 2], dtype=np.int64))

    def test_different_column_names(self, tmp_path):
        """Test with non-standard column names."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "uri,label\n"
            "s3://b/k1,5\n"
            "s3://b/k2,10\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="uri", class_col="label"
        )

        assert s3_uris == ["s3://b/k1", "s3://b/k2"]
        np.testing.assert_array_equal(labels, np.array([5, 10], dtype=np.int64))

    def test_string_path(self, tmp_path):
        """Test with string path instead of Path object."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,1\n")

        s3_uris, labels = load_index_csv(
            str(csv_path), s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://b/k"]
        assert labels[0] == 1

    def test_pathlib_path(self, tmp_path):
        """Test with Path object."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,1\n")

        s3_uris, labels = load_index_csv(
            Path(csv_path), s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://b/k"]
        assert labels[0] == 1

    def test_empty_csv(self, tmp_path):
        """Test with CSV that has headers but no data rows."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\n")

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == []
        assert len(labels) == 0
        assert labels.dtype == np.int64

    def test_single_row(self, tmp_path):
        """Test with single data row."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://bucket/single,42\n")

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://bucket/single"]
        assert labels[0] == 42

    def test_large_class_ids(self, tmp_path):
        """Test with large class IDs."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,999999\n")

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert labels[0] == 999999

    def test_negative_class_ids(self, tmp_path):
        """Test with negative class IDs (edge case)."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,-1\n")

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert labels[0] == -1

    def test_dtype_is_int64(self, tmp_path):
        """Test that labels are returned as int64."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,1\n")

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert labels.dtype == np.int64

    def test_preserves_uri_order(self, tmp_path):
        """Test that URI order is preserved."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://b/z,0\n"
            "s3://b/a,1\n"
            "s3://b/m,2\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://b/z", "s3://b/a", "s3://b/m"]

    def test_extra_columns_ignored(self, tmp_path):
        """Test that extra columns are ignored."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "extra1,s3_path,extra2,class_id,extra3\n"
            "foo,s3://b/k,bar,5,baz\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://b/k"]
        assert labels[0] == 5

    def test_uris_with_special_characters(self, tmp_path):
        """Test URIs containing special characters."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            '"s3://bucket/path with spaces/file.jpg",0\n'
            "s3://bucket/path-with-dashes/file_with_underscore.jpg,1\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert len(s3_uris) == 2
        assert "spaces" in s3_uris[0]

    # Error handling tests

    def test_missing_s3_uri_column(self, tmp_path):
        """Test error when s3_uri_col is missing."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("other_col,class_id\nfoo,1\n")

        with pytest.raises(ValueError, match="missing required columns"):
            load_index_csv(csv_path, s3_uri_col="s3_path", class_col="class_id")

    def test_missing_class_column(self, tmp_path):
        """Test error when class_col is missing."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,other_col\ns3://b/k,foo\n")

        with pytest.raises(ValueError, match="missing required columns"):
            load_index_csv(csv_path, s3_uri_col="s3_path", class_col="class_id")

    def test_missing_both_columns(self, tmp_path):
        """Test error when both required columns are missing."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\nfoo,bar\n")

        with pytest.raises(ValueError, match="missing required columns"):
            load_index_csv(csv_path, s3_uri_col="s3_path", class_col="class_id")

    def test_empty_file(self, tmp_path):
        """Test error with completely empty file."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("")

        with pytest.raises(ValueError, match="no header"):
            load_index_csv(csv_path, s3_uri_col="s3_path", class_col="class_id")

    def test_nonexistent_file(self, tmp_path):
        """Test error when file doesn't exist."""
        csv_path = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            load_index_csv(csv_path, s3_uri_col="s3_path", class_col="class_id")

    def test_invalid_class_id(self, tmp_path):
        """Test error when class_id cannot be converted to int."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,not_an_int\n")

        with pytest.raises(ValueError):
            load_index_csv(csv_path, s3_uri_col="s3_path", class_col="class_id")

    def test_float_class_id(self, tmp_path):
        """Test float class_id gets converted to int."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,5.0\n")

        # This will work because int("5.0") in python raises, but the CSV
        # returns string "5.0" which int() can't parse
        with pytest.raises(ValueError):
            load_index_csv(csv_path, s3_uri_col="s3_path", class_col="class_id")

    def test_many_rows(self, tmp_path):
        """Test with many rows."""
        csv_path = tmp_path / "test.csv"
        n = 10000
        lines = ["s3_path,class_id"]
        for i in range(n):
            lines.append(f"s3://bucket/key{i},{i % 100}")
        csv_path.write_text("\n".join(lines))

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert len(s3_uris) == n
        assert len(labels) == n
        assert s3_uris[0] == "s3://bucket/key0"
        assert s3_uris[-1] == f"s3://bucket/key{n-1}"

    def test_return_types(self, tmp_path):
        """Test that return types are correct."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://b/k,1\n")

        s3_uris, labels = load_index_csv(
            csv_path, s3_uri_col="s3_path", class_col="class_id"
        )

        assert isinstance(s3_uris, list)
        assert isinstance(labels, np.ndarray)
        assert all(isinstance(uri, str) for uri in s3_uris)

    # pack_uris tests

    def test_pack_uris_infers_prefix(self, tmp_path):
        """Test pack_uris=True without compact_prefix infers prefix from URIs."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://bucket/key1,0\n"
            "s3://bucket/key2,1\n"
            "s3://bucket/key3,2\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
        )

        assert isinstance(s3_uris, S3KeyList)
        assert len(s3_uris) == 3
        assert s3_uris[0] == "s3://bucket/key1"
        assert s3_uris[1] == "s3://bucket/key2"
        assert s3_uris[2] == "s3://bucket/key3"
        assert s3_uris.bucket == "bucket"
        np.testing.assert_array_equal(labels, np.array([0, 1, 2], dtype=np.int64))

    def test_pack_uris_mixed_buckets_raises(self, tmp_path):
        """Test pack_uris=True with mixed buckets raises ValueError."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://bucket1/key1,0\n"
            "s3://bucket2/key2,1\n"
        )

        with pytest.raises(ValueError, match="Mixed buckets"):
            load_index_csv(
                csv_path,
                s3_uri_col="s3_path",
                class_col="class_id",
                pack_uris=True,
            )

    def test_pack_uris_false_returns_list(self, tmp_path):
        """Test pack_uris=False (default) returns List[str]."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://bucket/key1,0\n")

        s3_uris, labels = load_index_csv(
            csv_path,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=False,
        )

        assert isinstance(s3_uris, list)
        assert not isinstance(s3_uris, S3KeyList)

    def test_compact_prefix_ignored_when_pack_uris_false(self, tmp_path):
        """Test that compact_prefix is ignored when pack_uris=False."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://bucket/key1,0\n")

        s3_uris, labels = load_index_csv(
            csv_path,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=False,
            compact_prefix="s3://bucket",
        )

        assert isinstance(s3_uris, list)
        assert not isinstance(s3_uris, S3KeyList)

    def test_pack_uris_with_compact_prefix(self, tmp_path):
        """Test pack_uris=True with explicit compact_prefix."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://bucket/key1,0\n"
            "s3://bucket/key2,1\n"
            "s3://bucket/key3,2\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://bucket",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert len(s3_uris) == 3
        assert s3_uris[0] == "s3://bucket/key1"
        assert s3_uris[1] == "s3://bucket/key2"
        assert s3_uris[2] == "s3://bucket/key3"
        np.testing.assert_array_equal(labels, np.array([0, 1, 2], dtype=np.int64))

    def test_pack_uris_compact_prefix_trailing_slash(self, tmp_path):
        """Test pack_uris with compact_prefix with trailing slash."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://bucket/key1,0\n")

        s3_uris, labels = load_index_csv(
            csv_path,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://bucket/",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert s3_uris[0] == "s3://bucket/key1"

    def test_pack_uris_compact_prefix_partial_path(self, tmp_path):
        """Test pack_uris with partial path prefix."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://bucket/data/train/img1.jpg,0\n"
            "s3://bucket/data/train/img2.jpg,1\n"
        )

        s3_uris, labels = load_index_csv(
            csv_path,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://bucket/data/train",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert s3_uris[0] == "s3://bucket/data/train/img1.jpg"
        assert s3_uris[1] == "s3://bucket/data/train/img2.jpg"

    def test_pack_uris_compact_prefix_mismatch_raises(self, tmp_path):
        """Test that prefix mismatch raises ValueError with index."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "s3_path,class_id\n"
            "s3://bucket/key1,0\n"
            "s3://other-bucket/key2,1\n"
        )

        with pytest.raises(ValueError, match="index 1"):
            load_index_csv(
                csv_path,
                s3_uri_col="s3_path",
                class_col="class_id",
                pack_uris=True,
                compact_prefix="s3://bucket",
            )

    def test_pack_uris_bucket_property(self, tmp_path):
        """Test that S3KeyList.bucket property works."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\ns3://mybucket/key,0\n")

        s3_uris, labels = load_index_csv(
            csv_path,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://mybucket",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert s3_uris.bucket == "mybucket"

    def test_pack_uris_empty_csv_raises(self, tmp_path):
        """Test that empty CSV with pack_uris raises."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("s3_path,class_id\n")

        with pytest.raises(ValueError, match="empty"):
            load_index_csv(
                csv_path,
                s3_uri_col="s3_path",
                class_col="class_id",
                pack_uris=True,
            )


class TestLoadIndexDataframe:
    """Test suite for load_index_dataframe function."""

    def test_basic_functionality(self):
        """Test basic DataFrame loading."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1", "s3://bucket/key2", "s3://bucket/key3"],
            "class_id": [0, 1, 2],
            "other_col": ["foo", "bar", "baz"],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == [
            "s3://bucket/key1",
            "s3://bucket/key2",
            "s3://bucket/key3",
        ]
        np.testing.assert_array_equal(labels, np.array([0, 1, 2], dtype=np.int64))

    def test_different_column_names(self):
        """Test with non-standard column names."""
        df = pd.DataFrame({
            "uri": ["s3://b/k1", "s3://b/k2"],
            "label": [5, 10],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="uri", class_col="label"
        )

        assert s3_uris == ["s3://b/k1", "s3://b/k2"]
        np.testing.assert_array_equal(labels, np.array([5, 10], dtype=np.int64))

    def test_empty_dataframe(self):
        """Test with DataFrame that has columns but no data rows."""
        df = pd.DataFrame({"s3_path": [], "class_id": []})

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == []
        assert len(labels) == 0
        assert labels.dtype == np.int64

    def test_single_row(self):
        """Test with single data row."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/single"],
            "class_id": [42],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://bucket/single"]
        assert labels[0] == 42

    def test_large_class_ids(self):
        """Test with large class IDs."""
        df = pd.DataFrame({
            "s3_path": ["s3://b/k"],
            "class_id": [999999],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert labels[0] == 999999

    def test_negative_class_ids(self):
        """Test with negative class IDs (edge case)."""
        df = pd.DataFrame({
            "s3_path": ["s3://b/k"],
            "class_id": [-1],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert labels[0] == -1

    def test_dtype_is_int64(self):
        """Test that labels are returned as int64."""
        df = pd.DataFrame({
            "s3_path": ["s3://b/k"],
            "class_id": [1],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert labels.dtype == np.int64

    def test_preserves_uri_order(self):
        """Test that URI order is preserved."""
        df = pd.DataFrame({
            "s3_path": ["s3://b/z", "s3://b/a", "s3://b/m"],
            "class_id": [0, 1, 2],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://b/z", "s3://b/a", "s3://b/m"]

    def test_extra_columns_ignored(self):
        """Test that extra columns are ignored."""
        df = pd.DataFrame({
            "extra1": ["foo"],
            "s3_path": ["s3://b/k"],
            "extra2": ["bar"],
            "class_id": [5],
            "extra3": ["baz"],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert s3_uris == ["s3://b/k"]
        assert labels[0] == 5

    def test_missing_s3_uri_column(self):
        """Test error when s3_uri_col is missing."""
        df = pd.DataFrame({
            "other_col": ["foo"],
            "class_id": [1],
        })

        with pytest.raises(ValueError, match="missing required columns"):
            load_index_dataframe(df, s3_uri_col="s3_path", class_col="class_id")

    def test_missing_class_column(self):
        """Test error when class_col is missing."""
        df = pd.DataFrame({
            "s3_path": ["s3://b/k"],
            "other_col": ["foo"],
        })

        with pytest.raises(ValueError, match="missing required columns"):
            load_index_dataframe(df, s3_uri_col="s3_path", class_col="class_id")

    def test_missing_both_columns(self):
        """Test error when both required columns are missing."""
        df = pd.DataFrame({
            "col1": ["foo"],
            "col2": ["bar"],
        })

        with pytest.raises(ValueError, match="missing required columns"):
            load_index_dataframe(df, s3_uri_col="s3_path", class_col="class_id")

    def test_float_class_id_converts(self):
        """Test that float class IDs are converted to int."""
        df = pd.DataFrame({
            "s3_path": ["s3://b/k"],
            "class_id": [5.0],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert labels[0] == 5
        assert labels.dtype == np.int64

    def test_many_rows(self):
        """Test with many rows."""
        n = 10000
        df = pd.DataFrame({
            "s3_path": [f"s3://bucket/key{i}" for i in range(n)],
            "class_id": [i % 100 for i in range(n)],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert len(s3_uris) == n
        assert len(labels) == n
        assert s3_uris[0] == "s3://bucket/key0"
        assert s3_uris[-1] == f"s3://bucket/key{n-1}"

    def test_return_types(self):
        """Test that return types are correct."""
        df = pd.DataFrame({
            "s3_path": ["s3://b/k"],
            "class_id": [1],
        })

        s3_uris, labels = load_index_dataframe(
            df, s3_uri_col="s3_path", class_col="class_id"
        )

        assert isinstance(s3_uris, list)
        assert isinstance(labels, np.ndarray)
        assert all(isinstance(uri, str) for uri in s3_uris)

    # pack_uris tests

    def test_pack_uris_infers_prefix(self):
        """Test pack_uris=True without compact_prefix infers prefix from URIs."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1", "s3://bucket/key2", "s3://bucket/key3"],
            "class_id": [0, 1, 2],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
        )

        assert isinstance(s3_uris, S3KeyList)
        assert len(s3_uris) == 3
        assert s3_uris[0] == "s3://bucket/key1"
        assert s3_uris[1] == "s3://bucket/key2"
        assert s3_uris[2] == "s3://bucket/key3"
        assert s3_uris.bucket == "bucket"
        np.testing.assert_array_equal(labels, np.array([0, 1, 2], dtype=np.int64))

    def test_pack_uris_mixed_buckets_raises(self):
        """Test pack_uris=True with mixed buckets raises ValueError."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket1/key1", "s3://bucket2/key2"],
            "class_id": [0, 1],
        })

        with pytest.raises(ValueError, match="Mixed buckets"):
            load_index_dataframe(
                df,
                s3_uri_col="s3_path",
                class_col="class_id",
                pack_uris=True,
            )

    def test_pack_uris_false_returns_list(self):
        """Test pack_uris=False (default) returns List[str]."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1"],
            "class_id": [0],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=False,
        )

        assert isinstance(s3_uris, list)
        assert not isinstance(s3_uris, S3KeyList)

    def test_compact_prefix_ignored_when_pack_uris_false(self):
        """Test that compact_prefix is ignored when pack_uris=False."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1"],
            "class_id": [0],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=False,
            compact_prefix="s3://bucket",
        )

        assert isinstance(s3_uris, list)
        assert not isinstance(s3_uris, S3KeyList)

    def test_pack_uris_with_compact_prefix(self):
        """Test pack_uris=True with explicit compact_prefix."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1", "s3://bucket/key2", "s3://bucket/key3"],
            "class_id": [0, 1, 2],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://bucket",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert len(s3_uris) == 3
        assert s3_uris[0] == "s3://bucket/key1"
        assert s3_uris[1] == "s3://bucket/key2"
        assert s3_uris[2] == "s3://bucket/key3"
        np.testing.assert_array_equal(labels, np.array([0, 1, 2], dtype=np.int64))

    def test_pack_uris_compact_prefix_trailing_slash(self):
        """Test pack_uris with compact_prefix with trailing slash."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1"],
            "class_id": [0],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://bucket/",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert s3_uris[0] == "s3://bucket/key1"

    def test_pack_uris_compact_prefix_partial_path(self):
        """Test pack_uris with partial path prefix."""
        df = pd.DataFrame({
            "s3_path": [
                "s3://bucket/data/train/img1.jpg",
                "s3://bucket/data/train/img2.jpg",
            ],
            "class_id": [0, 1],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://bucket/data/train",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert s3_uris[0] == "s3://bucket/data/train/img1.jpg"
        assert s3_uris[1] == "s3://bucket/data/train/img2.jpg"

    def test_pack_uris_compact_prefix_mismatch_raises(self):
        """Test that prefix mismatch raises ValueError with index."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1", "s3://other-bucket/key2"],
            "class_id": [0, 1],
        })

        with pytest.raises(ValueError, match="index 1"):
            load_index_dataframe(
                df,
                s3_uri_col="s3_path",
                class_col="class_id",
                pack_uris=True,
                compact_prefix="s3://bucket",
            )

    def test_pack_uris_bucket_property(self):
        """Test that S3KeyList.bucket property works."""
        df = pd.DataFrame({
            "s3_path": ["s3://mybucket/key"],
            "class_id": [0],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://mybucket",
        )

        assert isinstance(s3_uris, S3KeyList)
        assert s3_uris.bucket == "mybucket"

    def test_pack_uris_empty_df_raises(self):
        """Test that empty DataFrame with pack_uris raises."""
        df = pd.DataFrame({"s3_path": [], "class_id": []})

        with pytest.raises(ValueError, match="empty"):
            load_index_dataframe(
                df,
                s3_uri_col="s3_path",
                class_col="class_id",
                pack_uris=True,
            )

    def test_pack_uris_iteration(self):
        """Test that S3KeyList supports iteration."""
        df = pd.DataFrame({
            "s3_path": ["s3://bucket/key1", "s3://bucket/key2"],
            "class_id": [0, 1],
        })

        s3_uris, labels = load_index_dataframe(
            df,
            s3_uri_col="s3_path",
            class_col="class_id",
            pack_uris=True,
            compact_prefix="s3://bucket",
        )

        uris_list = list(s3_uris)
        assert uris_list == ["s3://bucket/key1", "s3://bucket/key2"]

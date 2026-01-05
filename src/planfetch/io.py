# io.py
"""CSV and file I/O utilities for loading dataset indices."""
from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from .uri_storage import S3KeyList, S3URISequence

if TYPE_CHECKING:
    import pandas as pd


def load_index_csv(
    csv_path: Union[str, os.PathLike],
    *,
    s3_uri_col: str,
    class_col: str,
    pack_uris: bool = False,
    compact_prefix: Optional[str] = None,
) -> Tuple[S3URISequence, np.ndarray]:
    """
    Load a CSV that contains (at minimum) two columns:
      - s3_uri_col: an S3 URI like "s3://bucket/key"
      - class_col: an integer-like class id

    Args:
        csv_path: Path to CSV file.
        s3_uri_col: Column name containing S3 URIs.
        class_col: Column name containing integer class labels.
        pack_uris: If True, return memory-efficient S3KeyList instead of List[str].
            Provides ~58% memory reduction for large datasets.
        compact_prefix: When pack_uris=True, use this prefix instead of inferring
            from the first URI. All URIs must start with this prefix.
            Ignored when pack_uris=False.

    Returns:
        s3_uris: List[str] if pack_uris=False, else S3KeyList
        labels: np.ndarray[int64] shape [N]

    Raises:
        ValueError: If columns are missing, or if pack_uris=True and URIs have
            mixed buckets (when inferring) or don't match compact_prefix.

    Note:
      This loads the whole CSV into memory. If your CSV is huge, consider:
      - Using pack_uris=True for ~58% memory reduction
      - Parquet + memory-mapped columns
      - sharded CSVs (per class or per prefix)
    """
    # Validate CSV structure first
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        if s3_uri_col not in reader.fieldnames or class_col not in reader.fieldnames:
            raise ValueError(f"CSV missing required columns. Found: {reader.fieldnames}")

    # Read all data
    uris: List[str] = []
    labels: List[int] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uris.append(row[s3_uri_col])
            labels.append(int(row[class_col]))

    labels_array = np.asarray(labels, dtype=np.int64)

    if pack_uris:
        if compact_prefix is not None:
            key_list = S3KeyList.from_prefix_iterable(compact_prefix, iter(uris))
        else:
            key_list = S3KeyList.from_uris(uris)
        return key_list, labels_array
    else:
        return uris, labels_array


def load_index_dataframe(
    df: "pd.DataFrame",
    *,
    s3_uri_col: str,
    class_col: str,
    pack_uris: bool = False,
    compact_prefix: Optional[str] = None,
) -> Tuple[S3URISequence, np.ndarray]:
    """
    Load dataset index from a pandas DataFrame.

    The DataFrame must contain (at minimum) two columns:
      - s3_uri_col: an S3 URI like "s3://bucket/key"
      - class_col: an integer-like class id

    Args:
        df: pandas DataFrame containing the index data.
        s3_uri_col: Column name containing S3 URIs.
        class_col: Column name containing integer class labels.
        pack_uris: If True, return memory-efficient S3KeyList instead of List[str].
            Provides ~58% memory reduction for large datasets.
        compact_prefix: When pack_uris=True, use this prefix instead of inferring
            from the first URI. All URIs must start with this prefix.
            Ignored when pack_uris=False.

    Returns:
        s3_uris: List[str] if pack_uris=False, else S3KeyList
        labels: np.ndarray[int64] shape [N]

    Raises:
        ValueError: If required columns are missing, or if pack_uris=True and URIs
            have mixed buckets (when inferring) or don't match compact_prefix.
    """
    if s3_uri_col not in df.columns or class_col not in df.columns:
        raise ValueError(
            f"DataFrame missing required columns. "
            f"Expected '{s3_uri_col}' and '{class_col}', found: {list(df.columns)}"
        )

    labels = df[class_col].to_numpy(dtype=np.int64)

    if pack_uris:
        if compact_prefix is not None:
            key_list = S3KeyList.from_prefix_iterable(compact_prefix, df[s3_uri_col])
        else:
            key_list = S3KeyList.from_uris(df[s3_uri_col].tolist())
        return key_list, labels
    else:
        s3_uris = df[s3_uri_col].tolist()
        return s3_uris, labels

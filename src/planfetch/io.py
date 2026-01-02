# io.py
"""CSV and file I/O utilities for loading dataset indices."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


def load_index_csv(
    csv_path: Union[str, os.PathLike],
    *,
    s3_uri_col: str,
    class_col: str,
) -> Tuple[List[str], np.ndarray]:
    """
    Load a CSV that contains (at minimum) two columns:
      - s3_uri_col: an S3 URI like "s3://bucket/key"
      - class_col: an integer-like class id

    Returns:
      s3_uris: list[str] length N
      labels:  np.ndarray[int64] shape [N]

    Note:
      This loads the whole CSV into memory. If your CSV is huge, consider:
      - Parquet + memory-mapped columns
      - sharded CSVs (per class or per prefix)
      - a database or key-value index
    """
    s3_uris: List[str] = []
    labels: List[int] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        if s3_uri_col not in reader.fieldnames or class_col not in reader.fieldnames:
            raise ValueError(f"CSV missing required columns. Found: {reader.fieldnames}")

        for row in reader:
            s3_uris.append(row[s3_uri_col])
            labels.append(int(row[class_col]))

    return s3_uris, np.asarray(labels, dtype=np.int64)

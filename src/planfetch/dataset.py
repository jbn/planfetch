# dataset.py
"""PyTorch Dataset wrapper for S3-backed data."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from torch.utils.data import Dataset

from .cache import SharedDiskCache


class S3PathDataset(Dataset):
    """
    Dataset: idx -> (image, label)

    The Dataset owns preprocessing.
    The cache owns S3 fetching and local storage.
    """
    def __init__(
        self,
        s3_uris: Sequence[str],
        labels: Sequence[int],
        *,
        cache: SharedDiskCache,
        image_loader,
        transform=None,
    ):
        self.s3_uris = list(s3_uris)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.cache = cache
        self.image_loader = image_loader
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.s3_uris))

    def __getitem__(self, idx):
        i = int(idx)
        uri = self.s3_uris[i]
        path = self.cache.ensure_local(uri)
        img = self.image_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

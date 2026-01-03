# dataset.py
"""PyTorch Dataset wrapper for S3-backed data."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from .cache import SharedDiskCache
from .uri_storage import S3KeyList, S3URISequence


class S3PathDataset(Dataset):
    """
    Dataset: idx -> (image, label)

    The Dataset owns preprocessing.
    The cache owns S3 fetching and local storage.

    Accepts either List[str] of full URIs or S3KeyList for memory efficiency.
    """

    def __init__(
        self,
        s3_uris: S3URISequence,
        labels: Sequence[int],
        *,
        cache: SharedDiskCache,
        image_loader,
        transform=None,
    ):
        # Store efficiently based on input type
        if isinstance(s3_uris, S3KeyList):
            self._uri_storage: S3URISequence = s3_uris
        else:
            self._uri_storage = list(s3_uris)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.cache = cache
        self.image_loader = image_loader
        self.transform = transform

    @property
    def s3_uris(self) -> S3URISequence:
        """URI storage (S3KeyList or List[str])."""
        return self._uri_storage

    def __len__(self) -> int:
        return len(self._uri_storage)

    def __getitem__(self, idx):
        i = int(idx)
        uri = self._uri_storage[i]
        path = self.cache.ensure_local(uri)
        img = self.image_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

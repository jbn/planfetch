# dataset.py
"""PyTorch Dataset wrapper for S3-backed data."""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from .cache import S3TerminalError, SharedDiskCache
from .uri_storage import S3KeyList, S3URISequence

# How many substitutions on_error may chain before giving up on an index.
_MAX_ERROR_HOPS = 8


class S3PathDataset(Dataset[Tuple[Any, int]]):
    """
    Dataset: idx -> (image, label)

    The Dataset owns preprocessing.
    The cache owns S3 fetching and local storage.

    Accepts either List[str] of full URIs or S3KeyList for memory efficiency.

    miss_wait_seconds: when the file is not in the cache yet, wait up to this
        long for it to appear (the parent's prefetch is fetching it) before
        downloading it here. 0 (default) downloads at once; the prefetching
        DataLoader builder sets this on the dataset because under it a miss is
        almost always a file already in flight in the parent.
    on_error: called as on_error(idx, exc) when S3 gives a terminal answer for
        idx (S3ObjectMissing, S3AccessDenied, S3TerminalError). It returns a
        replacement index to load instead. None (default) re-raises.
    """

    def __init__(
        self,
        s3_uris: S3URISequence,
        labels: Union[Sequence[int], ArrayLike],
        *,
        cache: SharedDiskCache,
        image_loader,
        transform=None,
        miss_wait_seconds: float = 0.0,
        on_error: Optional[Callable[[int, S3TerminalError], int]] = None,
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
        self.miss_wait_seconds = float(miss_wait_seconds)
        self.on_error = on_error

    @property
    def s3_uris(self) -> S3URISequence:
        """URI storage (S3KeyList or List[str])."""
        return self._uri_storage

    def __len__(self) -> int:
        return len(self._uri_storage)

    def _local_path(self, uri: str) -> str:
        if self.miss_wait_seconds > 0 and self.cache.wait_local(
            uri, self.miss_wait_seconds
        ):
            return self.cache.path_for(uri)
        return self.cache.ensure_local(uri)

    def __getitem__(self, idx):  # type: ignore[override]
        i = int(idx)
        for _hop in range(_MAX_ERROR_HOPS):
            uri = self._uri_storage[i]
            try:
                path = self._local_path(uri)
            except S3TerminalError as exc:
                if self.on_error is None:
                    raise
                i = int(self.on_error(i, exc))
                continue
            img = self.image_loader(path)
            if self.transform is not None:
                img = self.transform(img)
            return img, int(self.labels[i])
        raise S3TerminalError(
            self._uri_storage[i],
            "TooManySubstitutions",
            0,
            f"on_error substituted {_MAX_ERROR_HOPS} indices in a row, all terminal",
        )

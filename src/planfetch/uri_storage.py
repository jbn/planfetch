# uri_storage.py
"""Memory-efficient S3 URI storage for datasets with a common prefix."""
from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple, Union

import numpy as np


class S3KeyList:
    """
    Memory-efficient storage for S3 URIs that share a common prefix.

    Instead of storing full URIs like "s3://bucket/path/file.jpg" for each sample,
    this class stores:
    - The common prefix once (e.g., "s3://bucket/" or "s3://bucket/data/train/")
    - All suffixes packed into a single bytes object
    - An offset array to slice out individual suffixes

    Memory savings for 2M samples with common prefix + 40-char avg suffix:
    - List[str] of full URIs: ~230 MB
    - S3KeyList: ~96 MB (58% reduction)

    Usage:
        >>> uris = ["s3://mybucket/a.jpg", "s3://mybucket/b.jpg"]
        >>> keys = S3KeyList.from_uris(uris)
        >>> keys[0]  # "s3://mybucket/a.jpg"
        >>> keys.get_key(0)  # "a.jpg"
        >>> keys.get_bucket_and_key(0)  # ("mybucket", "a.jpg")

        # With explicit prefix for streaming construction:
        >>> keys = S3KeyList.from_prefix_iterable("s3://mybucket", uris)
    """

    __slots__ = ("prefix", "_data", "_offsets")

    def __init__(self, prefix: str, data: bytes, offsets: np.ndarray) -> None:
        """
        Direct constructor. Prefer using from_uris() or from_prefix_iterable().

        Args:
            prefix: The common URI prefix ending with "/" (e.g., "s3://bucket/").
            data: Concatenated UTF-8 encoded suffixes.
            offsets: int64 array of shape [n+1] where offsets[i]:offsets[i+1] is suffix i.
        """
        self.prefix = prefix
        self._data = data
        self._offsets = offsets

    @property
    def bucket(self) -> str:
        """Extract bucket name from prefix for backward compatibility."""
        # prefix is like "s3://bucket/" or "s3://bucket/path/"
        if self.prefix.startswith("s3://"):
            rest = self.prefix[5:]
            slash_idx = rest.find("/")
            if slash_idx != -1:
                return rest[:slash_idx]
            return rest
        return self.prefix

    @classmethod
    def from_uris(cls, uris: Sequence[str]) -> S3KeyList:
        """
        Create from full S3 URIs. All must share the same bucket.

        Infers the bucket from the first URI and validates all others match.

        Args:
            uris: Sequence of S3 URIs like "s3://bucket/key".

        Returns:
            S3KeyList with packed key storage.

        Raises:
            ValueError: If URIs are empty, have different buckets, or invalid format.
        """
        if not uris:
            raise ValueError("Cannot create S3KeyList from empty sequence")

        # Parse first URI to get bucket
        first = uris[0]
        if not first.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI (must start with s3://): {first}")
        rest = first[5:]
        slash_idx = rest.find("/")
        if slash_idx == -1:
            raise ValueError(f"Invalid S3 URI (no key after bucket): {first}")
        bucket = rest[:slash_idx]
        first_key = rest[slash_idx + 1 :]

        # Build packed keys
        expected_prefix = f"s3://{bucket}/"
        prefix_len = len(expected_prefix)

        # Collect keys as bytes
        key_bytes_list: List[bytes] = [first_key.encode("utf-8")]

        for uri in uris[1:]:
            if not uri.startswith(expected_prefix):
                if uri.startswith("s3://"):
                    other_rest = uri[5:]
                    other_slash = other_rest.find("/")
                    other_bucket = (
                        other_rest[:other_slash] if other_slash != -1 else other_rest
                    )
                    raise ValueError(
                        f"Mixed buckets: expected '{bucket}', got '{other_bucket}'"
                    )
                raise ValueError(f"Invalid S3 URI: {uri}")
            key_bytes_list.append(uri[prefix_len:].encode("utf-8"))

        # Build offset array and concatenate data
        n = len(key_bytes_list)
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        for i, kb in enumerate(key_bytes_list):
            offsets[i + 1] = offsets[i] + len(kb)

        data = b"".join(key_bytes_list)

        return cls(prefix=expected_prefix, data=data, offsets=offsets)

    @classmethod
    def from_prefix_iterable(cls, compact_prefix: str, uris: Iterable[str]) -> S3KeyList:
        """
        Create from an iterable of URIs with an explicit prefix.

        This is more memory-efficient than from_uris() as it streams URIs
        without building an intermediate list.

        Args:
            compact_prefix: The common prefix all URIs must start with.
                Can be "s3://bucket" or "s3://bucket/path/to/data".
                Will be normalized to end with "/".
            uris: Iterable of S3 URIs that all start with compact_prefix.

        Returns:
            S3KeyList with packed suffix storage.

        Raises:
            ValueError: If any URI doesn't start with the prefix.
        """
        # Normalize prefix to end with /
        if compact_prefix.endswith("/"):
            prefix = compact_prefix
        else:
            prefix = compact_prefix + "/"

        prefix_len = len(prefix)

        # Stream through URIs, collecting suffix bytes
        suffix_bytes_list: List[bytes] = []
        for idx, uri in enumerate(uris):
            if not uri.startswith(prefix):
                raise ValueError(
                    f"URI at index {idx} does not start with prefix '{compact_prefix}': {uri}"
                )
            suffix = uri[prefix_len:]
            suffix_bytes_list.append(suffix.encode("utf-8"))

        if not suffix_bytes_list:
            raise ValueError("Cannot create S3KeyList from empty iterable")

        # Build offset array and concatenate data
        n = len(suffix_bytes_list)
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        for i, sb in enumerate(suffix_bytes_list):
            offsets[i + 1] = offsets[i] + len(sb)

        data = b"".join(suffix_bytes_list)

        return cls(prefix=prefix, data=data, offsets=offsets)

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, idx: int) -> str:
        """Reconstruct full URI for index."""
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for S3KeyList of length {len(self)}")
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        suffix = self._data[start:end].decode("utf-8")
        return self.prefix + suffix

    def __iter__(self) -> Iterator[str]:
        """Iterate over reconstructed URIs."""
        for i in range(len(self)):
            start = int(self._offsets[i])
            end = int(self._offsets[i + 1])
            suffix = self._data[start:end].decode("utf-8")
            yield self.prefix + suffix

    def get_suffix(self, idx: int) -> str:
        """Get suffix (the part after the prefix) without full URI reconstruction."""
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for S3KeyList of length {len(self)}")
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        return self._data[start:end].decode("utf-8")

    def get_key(self, idx: int) -> str:
        """Get key (suffix) without URI reconstruction. Alias for get_suffix()."""
        return self.get_suffix(idx)

    def get_bucket_and_key(self, idx: int) -> Tuple[str, str]:
        """Get (bucket, key) tuple without URI string allocation."""
        return self.bucket, self.get_suffix(idx)

    def to_list(self) -> List[str]:
        """Convert to full URI list (for backward compatibility)."""
        return list(self)

    def __repr__(self) -> str:
        return f"S3KeyList(prefix={self.prefix!r}, n={len(self)})"


# Type alias for functions that accept either format
S3URISequence = Union[Sequence[str], S3KeyList]

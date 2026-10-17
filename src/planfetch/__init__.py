# planfetch - S3 prefetching for PyTorch DataLoaders
"""
S3 prefetching utilities for efficient data loading in PyTorch.

This package provides:
- SharedDiskCache: Local disk cache for S3 objects with concurrent downloads
- PlanView/PlanProvider: Batch scheduling strategies (sequential, shuffled, stratified)
- PrefetchCoordinator: Sliding-window prefetch and prune logic
- S3PathDataset: PyTorch Dataset for S3-backed images
- S3PrefetchLightningCallback: Optional PyTorch Lightning integration
"""

from .cache import (
    CacheMeta,
    S3AccessDenied,
    S3ObjectMissing,
    S3TerminalError,
    SharedDiskCache,
    assert_rust_backend,
    has_rust_backend,
    parse_s3_uri,
)
from .dataloader import (
    PlanBatchSampler,
    PrefetchController,
    PrefetchCoordinator,
    PrefetchingLoader,
    build_prefetching_dataloader,
)
from .dataset import S3PathDataset
from .io import load_index_csv, load_index_dataframe
from .uri_storage import S3KeyList, S3URISequence
from .plans import (
    BalancedSubsamplingPlanProvider,
    PlanProvider,
    PlanView,
    SequentialPlanProvider,
    ShuffleOncePerEpochPlanProvider,
    StreamingStratifiedBatchPlanProvider,
)

# Optional Lightning callback (only available if pytorch_lightning is installed)
try:
    from .lightning import S3PrefetchLightningCallback
except Exception:
    pass

__all__ = [
    # Cache
    "CacheMeta",
    "S3AccessDenied",
    "S3ObjectMissing",
    "S3TerminalError",
    "SharedDiskCache",
    "assert_rust_backend",
    "has_rust_backend",
    "parse_s3_uri",
    # Plans
    "BalancedSubsamplingPlanProvider",
    "PlanView",
    "PlanProvider",
    "SequentialPlanProvider",
    "ShuffleOncePerEpochPlanProvider",
    "StreamingStratifiedBatchPlanProvider",
    # I/O
    "load_index_csv",
    "load_index_dataframe",
    # URI Storage
    "S3KeyList",
    "S3URISequence",
    # DataLoader
    "PlanBatchSampler",
    "PrefetchCoordinator",
    "PrefetchingLoader",
    "PrefetchController",
    "build_prefetching_dataloader",
    # Dataset
    "S3PathDataset",
    # Lightning (optional)
    "S3PrefetchLightningCallback",
]


def main() -> None:
    from .cli import cli

    cli()

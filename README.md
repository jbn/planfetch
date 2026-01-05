# S3 prefetch + sliding-window NVMe cache for PyTorch DataLoader

This repo-style design keeps things **PyTorch-native** while hiding S3 latency:

- **DataLoader owns ordering** via a `BatchSampler` (so no `shuffle=True`).
- A **PlanProvider** precomputes an epoch’s **batch plan** (shuffle, stratified batches, curriculum, etc.).
- A **PrefetchCoordinator** uses the same plan to download “what’s next” to NVMe and prune a moving window.
- The **Dataset stays responsible for decoding + transforms**; it just asks the cache for a local path.

It’s designed to be minimally intrusive:

- **Plain torch**: one line per epoch (`controller.set_epoch(epoch)`), then `for batch in loader: ...`
- **PyTorch Lightning**: a small callback calls `controller.set_epoch(...)` at the right hook.

---

## Why a PlanView (the “plan view” upgrade)

Instead of a Python list of batches (lots of Python-object overhead), a **PlanView** stores a dense matrix:

- shape: `[num_batches, batch_size]`
- dtype: `np.int64`

Benefits:

- **Memory**: 2,000,000 indices as `int64` is ~16 MB; as Python ints it’s typically hundreds of MB.
- **Speed**: fast iteration and fast window queries for prefetching (`plan.range_flat(lo, hi)`).

---

## Memory-thrifty stratification upgrade

A naive stratifier often does per epoch:

- for each class pool:
  - copy the class’s indices
  - shuffle the copy

If a class pool has millions of items, that copy-and-shuffle becomes expensive.

`StreamingStratifiedBatchPlanProvider` avoids this by:

- building per-class pools once: `np.where(labels == class)`
- generating an epoch stream for each class using a compact **modular permutation** (no pool copies)
- composing batches from those class streams

Tradeoff: it’s not a perfect Fisher–Yates shuffle over each class pool, but it is deterministic, “random-looking”, and very memory friendly.

---

## Core components

### `SharedDiskCache`
- Downloads `s3://bucket/key` objects into a deterministic path under `cache_dir`
- Uses:
  - **tmp → atomic rename** (no partial files)
  - **file locks** to dedupe across DataLoader worker processes
  - a **ThreadPoolExecutor** to run concurrent boto3 downloads
- Pruning modes:
  - `mode="window"`: delete anything not in the current keep-set (best for huge datasets)
  - `mode="pressure"`: evict only if `total_bytes > max_bytes` (LRU-ish by last access)

### `PlanProvider` → `PlanView`
- Produces a batch-aware plan for the epoch
- Examples:
  - `ShuffleOncePerEpochPlanProvider`: default shuffle
  - `StreamingStratifiedBatchPlanProvider`: per-batch class stratification
  - `BalancedSubsamplingPlanProvider`: exact equal samples per class (for imbalanced datasets)
  - `SequentialPlanProvider`: deterministic val/test

### `PlanBatchSampler`
- A BatchSampler backed by PlanView, keeping ordering in DataLoader land.

### `PrefetchCoordinator`
- Keeps a sliding window of batches on NVMe:
  - keep batches: `[i - B_back, i + B_ahead)`
- Prefetches ahead and periodically prunes outside-window files.

### `PrefetchingLoader`
- Wraps a PyTorch DataLoader and calls `prefetch.on_batch_yielded(batch_idx)` automatically to avoid custom loops.

### `PrefetchController`
- One method per epoch: `controller.set_epoch(epoch)`
- Updates sampler plan and primes prefetch.

---

## Plain PyTorch usage

### 1) Load CSV index
Your CSV must contain:
- an S3 path column (e.g. `s3_path`)
- a class label column (e.g. `class_id`)

```python
import numpy as np
from PIL import Image
import planfetch as pf

# Standard loading (full URIs as List[str])
train_uris, train_labels = pf.load_index_csv(
    "train.csv",
    s3_uri_col="s3_path",
    class_col="class_id",
)

# Memory-efficient loading (~58% less memory for large datasets)
# pack_uris=True returns S3KeyList instead of List[str]
train_uris, train_labels = pf.load_index_csv(
    "train.csv",
    s3_uri_col="s3_path",
    class_col="class_id",
    pack_uris=True,  # infers prefix from first URI
)

# With explicit prefix (validates all URIs match)
train_uris, train_labels = pf.load_index_csv(
    "train.csv",
    s3_uri_col="s3_path",
    class_col="class_id",
    pack_uris=True,
    compact_prefix="s3://my-bucket",  # or "s3://my-bucket/data/train"
)

# From pandas DataFrame
train_uris, train_labels = pf.load_index_dataframe(
    df,
    s3_uri_col="s3_path",
    class_col="class_id",
)

# Memory-efficient from DataFrame
train_uris, train_labels = pf.load_index_dataframe(
    df,
    s3_uri_col="s3_path",
    class_col="class_id",
    pack_uris=True,
    compact_prefix="s3://my-bucket",
)
```

### 2) Configure cache on NVMe
```python
nvme_bytes = 1_000_000_000_000  # example: 1 TB
cache = pf.SharedDiskCache(
    "/mnt/nvme/s3cache_train",
    max_bytes=int(0.8 * nvme_bytes),   # leave headroom
    download_threads=64,
    s3_max_pool_connections=256,
    min_age_seconds_before_delete=2.0,
)
```

### 3) Dataset does decode/transforms; cache does fetching
```python
def image_loader(path: str):
    return Image.open(path).convert("RGB")

train_ds = pf.S3PathDataset(
    train_uris,
    train_labels,
    cache=cache,
    image_loader=image_loader,
    transform=None,  # put torchvision transforms here
)
```

### 4) Choose a plan provider (stratified batches)
```python
batch_size = 256

plan_provider = pf.StreamingStratifiedBatchPlanProvider(
    np.asarray(train_labels),
    batch_size=batch_size,
    epoch_samples=2_000_000,  # samples per epoch
    seed=123,
)
```

#### Alternative: Balanced subsampling (for imbalanced datasets)
If you have class imbalance and want exactly equal samples per class in every batch:

```python
# Specify samples per class directly (batch_size = samples_per_class * num_classes)
plan_provider = pf.BalancedSubsamplingPlanProvider(
    np.asarray(train_labels),
    samples_per_class=8,  # 8 samples per class per batch
    epoch_samples=500_000,
    seed=123,
)

# Or use from_batch_size() if you prefer to set batch_size
# (batch_size must be divisible by num_classes)
plan_provider = pf.BalancedSubsamplingPlanProvider.from_batch_size(
    np.asarray(train_labels),
    batch_size=256,
    epoch_samples=500_000,
    seed=123,
)
```

This provider undersamples majority classes and oversamples minority classes to guarantee equal representation.

### 5) Build the prefetching DataLoader
```python
train_loader, train_controller = pf.build_prefetching_dataloader(
    dataset=train_ds,
    s3_uris=train_uris,
    plan_provider=plan_provider,
    cache=cache,
    num_workers=8,
    batches_ahead=200,
    batches_back=4,
    prune_every=10,
    prune_mode="window",
    # With num_workers>0, scan pruning is most robust across processes:
    prune_backend="scan",
    persistent_workers=True,
    pin_memory=True,
)
```

### 6) Training loop (minimal intrusion)
```python
for epoch in range(10):
    train_controller.set_epoch(epoch)

    for images, y in train_loader:
        # training step
        pass
```

---

## Val/test loaders (same pattern)

You can reuse the exact same wrapper pattern. Typically you swap PlanProvider and tune the window smaller.

### Sequential val/test plan
```python
val_uris, val_labels = pf.load_index_csv("val.csv", s3_uri_col="s3_path", class_col="class_id")

val_cache = pf.SharedDiskCache(
    "/mnt/nvme/s3cache_val",
    max_bytes=int(0.2 * nvme_bytes),
    download_threads=32,
)

val_ds = pf.S3PathDataset(val_uris, val_labels, cache=val_cache, image_loader=image_loader)

val_provider = pf.SequentialPlanProvider(n=len(val_ds), batch_size=256)

val_loader, val_controller = pf.build_prefetching_dataloader(
    dataset=val_ds,
    s3_uris=val_uris,
    plan_provider=val_provider,
    cache=val_cache,
    num_workers=4,
    batches_ahead=50,
    batches_back=2,
    prune_every=20,
    prune_mode="window",
    prune_backend="scan",
)

val_controller.set_epoch(0)
for batch in val_loader:
    pass
```

### Keeping val cached across runs
If val/test repeats frequently and you want it warm:
- use a separate cache directory
- use `prune_mode="pressure"` so it only evicts when over budget

---

## PyTorch Lightning usage

Two patterns work well:

1) **DataModule returns the PrefetchingLoader** for each split  
2) A **Callback** calls `controller.set_epoch(...)` at the right hooks

### Example (DataModule + Callback)

```python
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import planfetch as pf

def image_loader(path: str):
    return Image.open(path).convert("RGB")

class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_csv: str, val_csv: str, nvme_bytes: int):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.nvme_bytes = nvme_bytes

    def setup(self, stage=None):
        self.train_uris, self.train_labels = pf.load_index_csv(self.train_csv, s3_uri_col="s3_path", class_col="class_id")
        self.val_uris, self.val_labels = pf.load_index_csv(self.val_csv, s3_uri_col="s3_path", class_col="class_id")

        self.train_cache = pf.SharedDiskCache("/mnt/nvme/s3cache_train", max_bytes=int(0.8 * self.nvme_bytes), download_threads=64)
        self.val_cache   = pf.SharedDiskCache("/mnt/nvme/s3cache_val",   max_bytes=int(0.2 * self.nvme_bytes), download_threads=32)

        self.train_ds = pf.S3PathDataset(self.train_uris, self.train_labels, cache=self.train_cache, image_loader=image_loader)
        self.val_ds   = pf.S3PathDataset(self.val_uris,   self.val_labels,   cache=self.val_cache,   image_loader=image_loader)

        self.train_provider = pf.StreamingStratifiedBatchPlanProvider(
            np.asarray(self.train_labels), batch_size=256, epoch_samples=2_000_000, seed=123
        )
        self.val_provider = pf.SequentialPlanProvider(n=len(self.val_ds), batch_size=256)

        self.train_loader, self.train_controller = pf.build_prefetching_dataloader(
            dataset=self.train_ds,
            s3_uris=self.train_uris,
            plan_provider=self.train_provider,
            cache=self.train_cache,
            num_workers=8,
            batches_ahead=200,
            batches_back=4,
            prune_every=10,
            prune_mode="window",
            prune_backend="scan",
        )
        self.val_loader, self.val_controller = pf.build_prefetching_dataloader(
            dataset=self.val_ds,
            s3_uris=self.val_uris,
            plan_provider=self.val_provider,
            cache=self.val_cache,
            num_workers=4,
            batches_ahead=50,
            batches_back=2,
            prune_every=20,
            prune_mode="window",
            prune_backend="scan",
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

datamodule = MyDataModule("train.csv", "val.csv", nvme_bytes=1_000_000_000_000)

callback = pf.S3PrefetchLightningCallback(
    train_controller=datamodule.train_controller,
    val_controller=datamodule.val_controller,
)

trainer = pl.Trainer(callbacks=[callback], max_epochs=10)
# trainer.fit(model, datamodule=datamodule)
```

---

## Knob guide (your numbers: 200 KB avg, 2M samples/epoch, NVMe)

Average epoch payload:
- 2,000,000 × 200 KB ≈ ~400 GB (decimal) ≈ ~372 GiB

Good starting knobs:
- `batches_ahead`: 200
- `batches_back`: 4
- `download_threads`: 64 (try 128 if the network + S3 can sustain)
- `max_bytes`: 70–85% of NVMe
- `prune_backend`: `"scan"` when `num_workers > 0` for correctness on shared disk

---

## CLI: Speedtest

planfetch includes a CLI for benchmarking S3 prefetching performance. Use it to find optimal configuration before training on GPUs.

```bash
planfetch speedtest <csv_path> [OPTIONS]
```

### Quick examples

```bash
# Quick test with 1000 samples
planfetch speedtest data.csv --limit 1000 --batch-size 32

# Tune prefetch settings
planfetch speedtest data.csv \
  --batch-size 64 \
  --download-threads 128 \
  --batches-ahead 300 \
  --cache-dir /mnt/nvme/cache
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --limit` | all | Process only first N rows |
| `--offset` | 0 | Skip first N rows |
| `--sample-frac` | - | Random sample fraction (0.0-1.0) |
| `--s3-uri-col` | `s3_uri` | Column name for S3 URIs |
| `--class-col` | `label` | Column name for class labels |
| `--batch-size` | 64 | Batch size |
| `--cache-dir` | temp | Local cache directory |
| `--max-bytes` | 10GB | Maximum cache size |
| `--download-threads` | 64 | Number of download threads |
| `--s3-pool-connections` | 256 | Boto3 connection pool size |
| `--batches-ahead` | 200 | Batches to prefetch ahead |
| `--batches-back` | 4 | Batches to keep behind |
| `--prune-every` | 5 | Prune frequency (every N batches) |
| `--prune-mode` | window | `window` or `pressure` |
| `--prune-backend` | meta | `meta` or `scan` |
| `--strategy` | shuffle | `shuffle`, `sequential`, or `stratified` |
| `-v, --verbose` | - | Verbose output |

### Output

The speedtest displays a rich progress bar and reports:
- Total time, batches/sec, samples/sec
- Avg/P50/P95/P99 batch latency
- Total bytes downloaded

---

## Memory-efficient URI storage

When all your S3 files share a common bucket, use `pack_uris=True` for significant memory savings:

```python
# Standard loading (full URIs as Python strings)
train_uris, labels = pf.load_index_csv("train.csv", s3_uri_col="s3_path", class_col="class_id")

# Memory-efficient loading (prefix stored once + packed suffixes)
train_uris, labels = pf.load_index_csv(
    "train.csv",
    s3_uri_col="s3_path",
    class_col="class_id",
    pack_uris=True,  # infers prefix from first URI, validates all share same bucket
)

# With explicit prefix (for additional validation or partial path prefixes)
train_uris, labels = pf.load_index_csv(
    "train.csv",
    s3_uri_col="s3_path",
    class_col="class_id",
    pack_uris=True,
    compact_prefix="s3://my-bucket/data/train",  # all URIs must match this prefix
)
```

**Memory savings for 2M samples:**
- Without `pack_uris`: ~230 MB (full URIs as Python strings)
- With `pack_uris=True`: ~96 MB (prefix stored once + packed suffixes)
- **Reduction: ~58%**

The returned `S3KeyList` works identically to `List[str]`:
- Supports indexing: `train_uris[0]` returns `"s3://bucket/path/file.jpg"`
- Supports iteration: `for uri in train_uris: ...`
- Works with `S3PathDataset` and `build_prefetching_dataloader()` directly

**Note:** When using `pack_uris=True`:
- Without `compact_prefix`: Prefix is inferred from first URI; all URIs must share the same bucket
- With `compact_prefix`: All URIs must start with the specified prefix; a `ValueError` is raised with the index of any mismatched URI

---

## Rust S3 downloader (optional, high-performance)

planfetch includes an optional Rust extension (`planfetch_rs`) that provides significantly faster S3 downloads via async I/O.

### Why Rust?

The Python backend uses `boto3` with a `ThreadPoolExecutor`. The Rust backend uses:
- **tokio**: Async runtime with true concurrent I/O (not thread-per-download)
- **aws-sdk-s3**: Native AWS SDK with connection pooling
- **DashMap**: Lock-free concurrent set for inflight deduplication

This can improve download throughput, especially with high `download_threads` settings.

### Usage

The Rust backend is **enabled by default** when:
1. The extension is compiled and available (`planfetch_rs.so`)
2. No custom `s3_client` is passed to `SharedDiskCache`

```python
# Rust backend enabled automatically
cache = pf.SharedDiskCache("/mnt/nvme/cache", download_threads=128)

# Explicitly disable Rust backend
cache = pf.SharedDiskCache("/mnt/nvme/cache", use_rust=False)

# Custom s3_client forces Python backend
cache = pf.SharedDiskCache("/mnt/nvme/cache", s3_client=my_boto_client)
```

### Requiring the Rust backend

For production workloads where Python thread contention is too slow, use `assert_rust_backend()` to fail fast at startup:

```python
import planfetch as pf

# Fail immediately if Rust backend is not available
pf.assert_rust_backend()

# Now safe to proceed - Rust will be used
cache = pf.SharedDiskCache("/mnt/nvme/cache", download_threads=128)
```

You can also check availability without raising:

```python
if pf.has_rust_backend():
    print("Using high-performance Rust backend")
else:
    print("Falling back to Python backend")
```

### Features

- **Async downloads**: Uses tokio multi-threaded runtime
- **Semaphore-limited concurrency**: Configurable via `download_threads`
- **Inflight deduplication**: Won't re-download the same file if already in progress
- **Atomic writes**: Downloads to temp file, then renames
- **Exponential backoff retries**: Up to 6 attempts per file

### Building from source

The Rust extension is built automatically with `uv build` (via maturin). To build manually:

```bash
cd rust
cargo build --release
```

Requirements: Rust toolchain, maturin

---

## Notes on future DDP

When you move to DDP later:
- make cache directories rank-specific:
  - `/mnt/nvme/s3cache/run_{id}/rank_{rank}/...`
- each rank has its own DataLoader, plan, and prefetch controller
- dedupe across ranks is possible but usually not worth the extra coordination

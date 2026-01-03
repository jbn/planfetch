# Rust S3 Downloader Extension for planfetch

## Overview

Replace the Python `SharedDiskCache` download logic with a Rust extension using `pyo3` + `aws-sdk-s3` + `tokio` for high-performance parallel S3 downloads.

## Project Structure

```
planfetch/
├── src/
│   └── planfetch/
│       ├── cache.py          # Modified to use Rust extension
│       └── ...
├── rust/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
├── pyproject.toml            # Add maturin build config
└── ...
```

## Cargo.toml

```toml
[package]
name = "planfetch_rs"
version = "0.1.0"
edition = "2021"

[lib]
name = "planfetch_rs"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
tokio = { version = "1", features = ["rt-multi-thread", "fs", "sync"] }
aws-config = { version = "1.5", features = ["behavior-version-latest"] }
aws-sdk-s3 = "1.65"
sha2 = "0.10"
hex = "0.4"
dashmap = "6"
thiserror = "2"
```

## src/lib.rs

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use aws_sdk_s3::Client;
use dashmap::DashSet;
use sha2::{Sha256, Digest};

/// Shared state for the downloader
struct Downloader {
    client: Client,
    cache_dir: PathBuf,
    runtime: Runtime,
    semaphore: Arc<Semaphore>,
    inflight: Arc<DashSet<String>>,
}

impl Downloader {
    fn new(cache_dir: String, max_concurrent: usize) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let client = runtime.block_on(async {
            let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
            Client::new(&config)
        });

        Ok(Self {
            client,
            cache_dir: PathBuf::from(cache_dir),
            runtime,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            inflight: Arc::new(DashSet::new()),
        })
    }

    fn sha256_hex(s: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        hex::encode(hasher.finalize())
    }

    fn path_for(&self, s3_uri: &str) -> PathBuf {
        let h = Self::sha256_hex(s3_uri);
        self.cache_dir.join(&h[..2]).join(&h[2..4]).join(&h)
    }

    fn parse_s3_uri(uri: &str) -> PyResult<(String, String)> {
        let rest = uri.strip_prefix("s3://")
            .ok_or_else(|| PyRuntimeError::new_err(format!("Not an S3 URI: {}", uri)))?;
        let (bucket, key) = rest.split_once('/')
            .ok_or_else(|| PyRuntimeError::new_err(format!("Invalid S3 URI: {}", uri)))?;
        Ok((bucket.to_string(), key.to_string()))
    }

    /// Download a single file if not already cached
    async fn download_one(
        client: &Client,
        s3_uri: String,
        dst: PathBuf,
        semaphore: Arc<Semaphore>,
        inflight: Arc<DashSet<String>>,
    ) -> Result<String, String> {
        // Check if already cached
        if dst.exists() {
            return Ok(dst.to_string_lossy().to_string());
        }

        // Check if already inflight
        let hash = Self::sha256_hex(&s3_uri);
        if !inflight.insert(hash.clone()) {
            // Already being downloaded, wait and return
            while inflight.contains(&hash) {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
            return Ok(dst.to_string_lossy().to_string());
        }

        // Acquire semaphore permit
        let _permit = semaphore.acquire().await.map_err(|e| e.to_string())?;

        // Double-check after acquiring permit
        if dst.exists() {
            inflight.remove(&hash);
            return Ok(dst.to_string_lossy().to_string());
        }

        // Parse URI
        let (bucket, key) = Self::parse_s3_uri(&s3_uri).map_err(|e| e.to_string())?;

        // Create parent directories
        if let Some(parent) = dst.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
        }

        // Download with retries
        let mut backoff = 250u64;
        for attempt in 0..6 {
            match client
                .get_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
            {
                Ok(resp) => {
                    let body = resp.body.collect().await.map_err(|e| e.to_string())?;
                    let bytes = body.into_bytes();

                    // Write to temp file then rename (atomic)
                    let tmp = dst.with_extension(format!("tmp.{}", std::process::id()));
                    tokio::fs::write(&tmp, &bytes).await.map_err(|e| e.to_string())?;
                    tokio::fs::rename(&tmp, &dst).await.map_err(|e| e.to_string())?;

                    inflight.remove(&hash);
                    return Ok(dst.to_string_lossy().to_string());
                }
                Err(e) => {
                    if attempt == 5 {
                        inflight.remove(&hash);
                        return Err(format!("Failed after 6 attempts: {}: {}", s3_uri, e));
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff)).await;
                    backoff *= 2;
                }
            }
        }

        inflight.remove(&hash);
        Err(format!("Failed to download: {}", s3_uri))
    }

    /// Download multiple URIs in parallel, return local paths
    fn download_batch(&self, uris: Vec<String>) -> PyResult<Vec<String>> {
        let client = self.client.clone();
        let semaphore = self.semaphore.clone();
        let inflight = self.inflight.clone();

        let futures: Vec<_> = uris.iter().map(|uri| {
            let client = client.clone();
            let dst = self.path_for(uri);
            let uri = uri.clone();
            let sem = semaphore.clone();
            let inf = inflight.clone();

            async move {
                Self::download_one(&client, uri, dst, sem, inf).await
            }
        }).collect();

        self.runtime.block_on(async {
            let results = futures::future::join_all(futures).await;
            results.into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PyRuntimeError::new_err(e))
        })
    }

    /// Schedule downloads without blocking (for prefetching)
    fn prime_prefetch(&self, uris: Vec<String>) {
        let client = self.client.clone();
        let semaphore = self.semaphore.clone();
        let inflight = self.inflight.clone();

        for uri in uris {
            let client = client.clone();
            let dst = self.path_for(&uri);
            let sem = semaphore.clone();
            let inf = inflight.clone();

            // Skip if already cached or inflight
            if dst.exists() || inf.contains(&Self::sha256_hex(&uri)) {
                continue;
            }

            self.runtime.spawn(async move {
                let _ = Self::download_one(&client, uri, dst, sem, inf).await;
            });
        }
    }

    /// Ensure a single URI is local (blocking)
    fn ensure_local(&self, uri: String) -> PyResult<String> {
        let paths = self.download_batch(vec![uri])?;
        Ok(paths.into_iter().next().unwrap())
    }

    /// Wait for all inflight downloads to complete
    fn wait_for_inflight(&self) {
        while !self.inflight.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

#[pyclass]
struct RustS3Cache {
    inner: Downloader,
}

#[pymethods]
impl RustS3Cache {
    #[new]
    #[pyo3(signature = (cache_dir, max_concurrent=1024))]
    fn new(cache_dir: String, max_concurrent: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Downloader::new(cache_dir, max_concurrent)?,
        })
    }

    /// Download a batch of S3 URIs, returning local paths
    fn download_batch(&self, uris: Vec<String>) -> PyResult<Vec<String>> {
        self.inner.download_batch(uris)
    }

    /// Schedule downloads for prefetching (non-blocking)
    fn prime_prefetch(&self, uris: Vec<String>) {
        self.inner.prime_prefetch(uris)
    }

    /// Ensure a single URI is downloaded and return local path
    fn ensure_local(&self, uri: String) -> PyResult<String> {
        self.inner.ensure_local(uri)
    }

    /// Wait for all inflight downloads
    fn wait_for_inflight(&self) {
        self.inner.wait_for_inflight()
    }

    /// Get the local path for a URI (doesn't download)
    fn path_for(&self, uri: String) -> String {
        self.inner.path_for(&uri).to_string_lossy().to_string()
    }
}

#[pymodule]
fn planfetch_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustS3Cache>()?;
    Ok(())
}
```

## pyproject.toml additions

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "src"
module-name = "planfetch.planfetch_rs"
manifest-path = "rust/Cargo.toml"
```

## Python Integration (cache.py)

```python
# At the top of cache.py
try:
    from .planfetch_rs import RustS3Cache
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

class SharedDiskCache:
    def __init__(
        self,
        cache_dir: Union[str, os.PathLike],
        *,
        max_bytes: int,
        download_threads: int = 64,
        s3_max_pool_connections: int = 256,
        min_age_seconds_before_delete: float = 2.0,
        s3_client=None,
        verbose: bool = False,
        use_rust: bool = True,  # New parameter
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use Rust backend if available and requested
        self._use_rust = use_rust and HAS_RUST
        if self._use_rust:
            self._rust_cache = RustS3Cache(str(cache_dir), download_threads)
        else:
            # ... existing Python implementation
            ...

    def ensure_local(self, s3_uri: str) -> str:
        if self._use_rust:
            return self._rust_cache.ensure_local(s3_uri)
        # ... existing implementation

    def prime_prefetch(self, s3_uris: Sequence[str]) -> None:
        if self._use_rust:
            self._rust_cache.prime_prefetch(list(s3_uris))
            return
        # ... existing implementation

    def wait_for_inflight(self, *, raise_errors: bool = False) -> None:
        if self._use_rust:
            self._rust_cache.wait_for_inflight()
            return
        # ... existing implementation
```

## Building

```bash
# Install maturin
pip install maturin

# Development build (fast, unoptimized)
maturin develop

# Release build (optimized)
maturin develop --release

# Build wheel for distribution
maturin build --release
```

## Expected Performance

| Implementation | Batches/sec | Notes |
|---------------|-------------|-------|
| Python download_file() | ~1.5 | Original |
| Python get_object() | ~2.7 | Current |
| Rust (1024 concurrent) | ~8-15 | Expected |
| Rust (theoretical max) | ~20+ | S3 latency limited |

The Rust implementation should be ~3-5x faster than the optimized Python version because:
1. No GIL - true parallel execution
2. tokio async runtime is extremely efficient for I/O
3. No Python object overhead in the hot path
4. Single process, shared connection pool

## Testing

```bash
# Quick test
rm -rf /data/quan_02/johnny_tmp_cache
uv run planfetch speedtest ../ai_gen_art_train_v51.csv \
  --batch-size 64 \
  --download-threads 1024 \
  --batches-ahead 64 \
  --cache-dir /data/quan_02/johnny_tmp_cache \
  --s3-uri-col s3_path \
  -n 10000
```

## Notes

1. **AWS Credentials**: The Rust SDK uses the same credential chain as boto3 (env vars, ~/.aws/credentials, IAM role, etc.)

2. **Pruning**: The Rust extension only handles downloads. Pruning still happens in Python (it's not performance-critical).

3. **Error Handling**: The Rust code propagates errors back to Python as `RuntimeError`.

4. **Memory**: Downloads are buffered in memory before writing to disk. For very large files (>100MB), you'd want to stream to disk instead.

5. **Graceful Fallback**: If the Rust extension isn't built, the code falls back to the Python implementation automatically.

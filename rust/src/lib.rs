use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use pyo3::types::PyDict;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use aws_sdk_s3::Client;
use dashmap::DashSet;
use sha2::{Sha256, Digest};

/// Shared state for the downloader
/// Counters for benchmarking: every GET attempt, every retry, throttles, failures, bytes.
#[derive(Default)]
struct Stats {
    attempts: AtomicU64,
    retries: AtomicU64,
    throttled: AtomicU64,
    failures: AtomicU64,
    downloaded: AtomicU64,
    bytes: AtomicU64,
}

/// Removes the hash from the in-flight set when dropped, on every exit path.
struct InflightGuard {
    set: Arc<DashSet<String>>,
    hash: String,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.set.remove(&self.hash);
    }
}

struct Downloader {
    client: Client,
    cache_dir: PathBuf,
    runtime: Runtime,
    semaphore: Arc<Semaphore>,
    inflight: Arc<DashSet<String>>,
    stats: Arc<Stats>,
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
            stats: Arc::new(Stats::default()),
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
        stats: Arc<Stats>,
    ) -> Result<String, String> {
        // Check if already cached
        if dst.exists() {
            return Ok(dst.to_string_lossy().to_string());
        }

        // Check if already inflight
        let hash = Self::sha256_hex(&s3_uri);
        if !inflight.insert(hash.clone()) {
            // Already being downloaded by this process: wait for it, but never forever.
            let started = std::time::Instant::now();
            while inflight.contains(&hash) {
                if dst.exists() {
                    return Ok(dst.to_string_lossy().to_string());
                }
                if started.elapsed() > std::time::Duration::from_secs(120) {
                    return Err(format!("Timed out waiting for in-flight download: {}", s3_uri));
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            }
            if dst.exists() {
                return Ok(dst.to_string_lossy().to_string());
            }
            return Err(format!("In-flight download did not produce the file: {}", s3_uri));
        }
        let _guard = InflightGuard { set: inflight.clone(), hash: hash.clone() };

        // Acquire semaphore permit
        let _permit = semaphore.acquire().await.map_err(|e| e.to_string())?;

        // Double-check after acquiring permit
        if dst.exists() {
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
            stats.attempts.fetch_add(1, Ordering::Relaxed);
            match client
                .get_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
            {
                Ok(resp) => {
                    let body = match resp.body.collect().await {
                        Ok(b) => b,
                        Err(e) => {
                            stats.retries.fetch_add(1, Ordering::Relaxed);
                            if attempt == 5 {
                                stats.failures.fetch_add(1, Ordering::Relaxed);
                                return Err(format!("Body read failed after 6 attempts: {}: {}", s3_uri, e));
                            }
                            tokio::time::sleep(tokio::time::Duration::from_millis(backoff)).await;
                            backoff *= 2;
                            continue;
                        }
                    };
                    let bytes = body.into_bytes();

                    // Write to temp file then rename (atomic)
                    let tmp = dst.with_extension(format!("tmp.{}", std::process::id()));
                    let written = match tokio::fs::write(&tmp, &bytes).await {
                        Ok(()) => tokio::fs::rename(&tmp, &dst).await,
                        Err(e) => Err(e),
                    };
                    if let Err(e) = written {
                        let _ = tokio::fs::remove_file(&tmp).await;
                        stats.failures.fetch_add(1, Ordering::Relaxed);
                        return Err(format!("Failed to write {}: {}", dst.display(), e));
                    }
                    stats.downloaded.fetch_add(1, Ordering::Relaxed);
                    stats.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                    return Ok(dst.to_string_lossy().to_string());
                }
                Err(e) => {
                    let msg = format!("{:?}", e);
                    if msg.contains("SlowDown") || msg.contains("503") || msg.contains("Throttl") {
                        stats.throttled.fetch_add(1, Ordering::Relaxed);
                    }
                    if attempt == 5 {
                        stats.failures.fetch_add(1, Ordering::Relaxed);
                        return Err(format!("Failed after 6 attempts: {}: {}", s3_uri, e));
                    }
                    stats.retries.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff)).await;
                    backoff *= 2;
                }
            }
        }

        Err(format!("Failed to download: {}", s3_uri))
    }

    /// Download multiple URIs in parallel, return local paths
    fn download_batch(&self, uris: Vec<String>) -> PyResult<Vec<String>> {
        let client = self.client.clone();
        let semaphore = self.semaphore.clone();
        let inflight = self.inflight.clone();
        let stats = self.stats.clone();

        let futures: Vec<_> = uris.iter().map(|uri| {
            let client = client.clone();
            let dst = self.path_for(uri);
            let uri = uri.clone();
            let sem = semaphore.clone();
            let inf = inflight.clone();
            let st = stats.clone();

            async move {
                Self::download_one(&client, uri, dst, sem, inf, st).await
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
        let stats = self.stats.clone();

        for uri in uris {
            let client = client.clone();
            let dst = self.path_for(&uri);
            let sem = semaphore.clone();
            let inf = inflight.clone();
            let st = stats.clone();

            // Skip if already cached or inflight
            if dst.exists() || inf.contains(&Self::sha256_hex(&uri)) {
                continue;
            }

            self.runtime.spawn(async move {
                let _ = Self::download_one(&client, uri, dst, sem, inf, st).await;
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

    fn stats_snapshot(&self) -> [(&'static str, u64); 7] {
        let s = &self.stats;
        [
            ("attempts", s.attempts.load(Ordering::Relaxed)),
            ("retries", s.retries.load(Ordering::Relaxed)),
            ("throttled", s.throttled.load(Ordering::Relaxed)),
            ("failures", s.failures.load(Ordering::Relaxed)),
            ("downloaded", s.downloaded.load(Ordering::Relaxed)),
            ("bytes", s.bytes.load(Ordering::Relaxed)),
            ("inflight", self.inflight.len() as u64),
        ]
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

    /// Download a batch of S3 URIs, returning local paths.
    /// Releases the GIL while blocked on S3.
    fn download_batch(&self, py: Python<'_>, uris: Vec<String>) -> PyResult<Vec<String>> {
        py.allow_threads(|| self.inner.download_batch(uris))
    }

    /// Schedule downloads for prefetching (non-blocking).
    /// Releases the GIL: this hashes and stats every URI.
    fn prime_prefetch(&self, py: Python<'_>, uris: Vec<String>) {
        py.allow_threads(|| self.inner.prime_prefetch(uris))
    }

    /// Ensure a single URI is downloaded and return local path.
    /// Releases the GIL while blocked on S3.
    fn ensure_local(&self, py: Python<'_>, uri: String) -> PyResult<String> {
        py.allow_threads(|| self.inner.ensure_local(uri))
    }

    /// Wait for all inflight downloads. Releases the GIL while waiting.
    fn wait_for_inflight(&self, py: Python<'_>) {
        py.allow_threads(|| self.inner.wait_for_inflight())
    }

    /// Get the local path for a URI (doesn't download)
    fn path_for(&self, uri: String) -> String {
        self.inner.path_for(&uri).to_string_lossy().to_string()
    }

    /// Download counters: attempts, retries, throttled, failures, downloaded, bytes, inflight
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in self.inner.stats_snapshot() {
            d.set_item(k, v)?;
        }
        Ok(d)
    }
}

#[pymodule]
fn planfetch_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustS3Cache>()?;
    Ok(())
}

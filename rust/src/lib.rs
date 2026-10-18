use aws_sdk_s3::error::{ProvideErrorMetadata, SdkError};
use aws_sdk_s3::operation::get_object::GetObjectError;
use aws_sdk_s3::Client;
use aws_smithy_runtime_api::client::orchestrator::HttpResponse;
use dashmap::DashSet;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sha2::{Digest, Sha256};
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

/// Counters for benchmarking: every GET attempt, every retry, throttles, failures, bytes.
///
/// `not_found` counts objects S3 said do not exist (NoSuchKey / NoSuchBucket / 404);
/// those are also counted in `failures` and are never retried.
#[derive(Default)]
struct Stats {
    attempts: AtomicU64,
    retries: AtomicU64,
    throttled: AtomicU64,
    failures: AtomicU64,
    not_found: AtomicU64,
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

/// How a failed GET should be handled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ErrClass {
    /// Transient: retry with backoff (5xx, timeouts, connection errors).
    Retry,
    /// S3 is pushing back (503 SlowDown, 429): retry with backoff and count it.
    Throttled,
    /// The object or bucket does not exist: terminal, no retry.
    NotFound,
    /// Not allowed to read it: terminal, no retry.
    Forbidden,
    /// Any other 4xx: terminal, no retry.
    Client,
}

impl ErrClass {
    fn is_terminal(self) -> bool {
        matches!(
            self,
            ErrClass::NotFound | ErrClass::Forbidden | ErrClass::Client
        )
    }

    /// Stable prefix the Python side maps to an exception type.
    fn tag(self) -> &'static str {
        match self {
            ErrClass::NotFound => "NotFound",
            ErrClass::Forbidden => "Forbidden",
            ErrClass::Client => "ClientError",
            ErrClass::Throttled => "Throttled",
            ErrClass::Retry => "Transient",
        }
    }
}

/// Classify a GetObject failure: (class, error code, HTTP status or 0).
fn classify(e: &SdkError<GetObjectError, HttpResponse>) -> (ErrClass, String, u16) {
    match e {
        SdkError::ServiceError(ctx) => {
            let err = ctx.err();
            let status = ctx.raw().status().as_u16();
            let code = err.code().unwrap_or("").to_string();
            let class = if matches!(err, GetObjectError::NoSuchKey(_))
                || code == "NoSuchKey"
                || code == "NoSuchBucket"
                || code == "NotFound"
                || status == 404
            {
                ErrClass::NotFound
            } else if code == "AccessDenied"
                || code == "InvalidAccessKeyId"
                || code == "SignatureDoesNotMatch"
                || code == "ExpiredToken"
                || status == 403
            {
                ErrClass::Forbidden
            } else if code == "SlowDown"
                || code == "Throttling"
                || code == "RequestLimitExceeded"
                || status == 503
                || status == 429
            {
                ErrClass::Throttled
            } else if (400..500).contains(&status) {
                ErrClass::Client
            } else {
                ErrClass::Retry
            };
            let code = if code.is_empty() {
                format!("HTTP{}", status)
            } else {
                code
            };
            (class, code, status)
        }
        SdkError::TimeoutError(_) => (ErrClass::Retry, "Timeout".to_string(), 0),
        SdkError::DispatchFailure(_) => (ErrClass::Retry, "DispatchFailure".to_string(), 0),
        SdkError::ResponseError(_) => (ErrClass::Retry, "ResponseError".to_string(), 0),
        SdkError::ConstructionFailure(_) => (ErrClass::Retry, "ConstructionFailure".to_string(), 0),
        _ => (ErrClass::Retry, "SdkError".to_string(), 0),
    }
}

struct Downloader {
    client: Client,
    cache_dir: PathBuf,
    /// Dropped only by the process that created it; see `impl Drop`.
    runtime: ManuallyDrop<Runtime>,
    semaphore: Arc<Semaphore>,
    inflight: Arc<DashSet<String>>,
    /// Background unlink batches still running (see `drop_paths`).
    pending_drops: Arc<AtomicU64>,
    stats: Arc<Stats>,
    /// The process that built the runtime (its worker threads live only there).
    owner_pid: u32,
    /// Runtime worker threads asked for (None = tokio's default, one per core).
    worker_threads: Option<usize>,
}

impl Drop for Downloader {
    fn drop(&mut self) {
        // A forked child (a DataLoader worker with the default start method)
        // inherits this struct but none of the runtime's threads. Dropping the
        // runtime there blocks forever: tokio's blocking-pool shutdown waits for
        // worker threads that do not exist in this process. So only the owning
        // process drops the runtime; a child leaks the inherited one (it holds
        // no file descriptors the child uses) and builds its own.
        if std::process::id() == self.owner_pid {
            // SAFETY: dropped exactly once, here, and never used afterwards.
            unsafe { ManuallyDrop::drop(&mut self.runtime) };
        }
    }
}

impl Downloader {
    fn new(
        cache_dir: String,
        max_concurrent: usize,
        worker_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder.enable_all();
        if let Some(n) = worker_threads {
            builder.worker_threads(n.max(1));
        }
        let runtime = builder
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let client = runtime.block_on(async {
            let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
            Client::new(&config)
        });

        Ok(Self {
            client,
            cache_dir: PathBuf::from(cache_dir),
            runtime: ManuallyDrop::new(runtime),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            inflight: Arc::new(DashSet::new()),
            pending_drops: Arc::new(AtomicU64::new(0)),
            stats: Arc::new(Stats::default()),
            owner_pid: std::process::id(),
            worker_threads,
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
        let rest = uri
            .strip_prefix("s3://")
            .ok_or_else(|| PyRuntimeError::new_err(format!("Not an S3 URI: {}", uri)))?;
        let (bucket, key) = rest
            .split_once('/')
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
            let started = Instant::now();
            while inflight.contains(&hash) {
                if dst.exists() {
                    return Ok(dst.to_string_lossy().to_string());
                }
                if started.elapsed() > Duration::from_secs(120) {
                    return Err(format!(
                        "Timed out waiting for in-flight download: {}",
                        s3_uri
                    ));
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            if dst.exists() {
                return Ok(dst.to_string_lossy().to_string());
            }
            return Err(format!(
                "In-flight download did not produce the file: {}",
                s3_uri
            ));
        }
        let _guard = InflightGuard {
            set: inflight.clone(),
            hash: hash.clone(),
        };

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
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| e.to_string())?;
        }

        // Download with retries. Transient and throttled errors back off and
        // retry (6 attempts); NotFound / Forbidden / other 4xx return at once.
        let mut backoff = 250u64;
        let mut last = String::new();
        for attempt in 0..6 {
            stats.attempts.fetch_add(1, Ordering::Relaxed);
            match client.get_object().bucket(&bucket).key(&key).send().await {
                Ok(resp) => {
                    let body = match resp.body.collect().await {
                        Ok(b) => b,
                        Err(e) => {
                            last = format!("Transient: BodyRead: {}", e);
                            if attempt == 5 {
                                stats.failures.fetch_add(1, Ordering::Relaxed);
                                return Err(format!(
                                    "Failed after 6 attempts: {}: {}",
                                    s3_uri, last
                                ));
                            }
                            stats.retries.fetch_add(1, Ordering::Relaxed);
                            tokio::time::sleep(Duration::from_millis(backoff)).await;
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
                    let (class, code, status) = classify(&e);
                    last = format!("{}: {} ({})", class.tag(), code, status);
                    if class == ErrClass::Throttled {
                        stats.throttled.fetch_add(1, Ordering::Relaxed);
                    }
                    if class.is_terminal() {
                        if class == ErrClass::NotFound {
                            stats.not_found.fetch_add(1, Ordering::Relaxed);
                        }
                        stats.failures.fetch_add(1, Ordering::Relaxed);
                        return Err(format!("{}: {}", last, s3_uri));
                    }
                    if attempt == 5 {
                        stats.failures.fetch_add(1, Ordering::Relaxed);
                        return Err(format!("Failed after 6 attempts: {}: {}", s3_uri, last));
                    }
                    stats.retries.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                    backoff *= 2;
                }
            }
        }

        Err(format!("Failed to download: {}: {}", s3_uri, last))
    }

    /// Download multiple URIs in parallel, return local paths
    fn download_batch(&self, uris: Vec<String>) -> PyResult<Vec<String>> {
        let client = self.client.clone();
        let semaphore = self.semaphore.clone();
        let inflight = self.inflight.clone();
        let stats = self.stats.clone();

        let futures: Vec<_> = uris
            .iter()
            .map(|uri| {
                let client = client.clone();
                let dst = self.path_for(uri);
                let uri = uri.clone();
                let sem = semaphore.clone();
                let inf = inflight.clone();
                let st = stats.clone();

                async move { Self::download_one(&client, uri, dst, sem, inf, st).await }
            })
            .collect();

        self.runtime.block_on(async {
            let results = futures::future::join_all(futures).await;
            results
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(PyRuntimeError::new_err)
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

    /// Block until the file for `uri` exists, or `timeout` passes. Never downloads.
    fn wait_for_path(&self, uri: &str, timeout: Duration) -> bool {
        let dst = self.path_for(uri);
        let deadline = Instant::now() + timeout;
        loop {
            if dst.exists() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Unlink the cached files for `uris` on a blocking thread; returns at once.
    /// Missing files are ignored. `wait_for_inflight` waits for these too.
    fn drop_paths(&self, uris: Vec<String>) {
        let paths: Vec<PathBuf> = uris.iter().map(|u| self.path_for(u)).collect();
        let pending = self.pending_drops.clone();
        pending.fetch_add(1, Ordering::SeqCst);
        self.runtime.spawn_blocking(move || {
            for p in paths {
                let _ = std::fs::remove_file(p);
            }
            pending.fetch_sub(1, Ordering::SeqCst);
        });
    }

    /// Wait for all inflight downloads and pending unlinks to complete
    fn wait_for_inflight(&self) {
        while !self.inflight.is_empty() || self.pending_drops.load(Ordering::SeqCst) > 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn stats_snapshot(&self) -> [(&'static str, u64); 8] {
        let s = &self.stats;
        [
            ("attempts", s.attempts.load(Ordering::Relaxed)),
            ("retries", s.retries.load(Ordering::Relaxed)),
            ("throttled", s.throttled.load(Ordering::Relaxed)),
            ("failures", s.failures.load(Ordering::Relaxed)),
            ("not_found", s.not_found.load(Ordering::Relaxed)),
            ("downloaded", s.downloaded.load(Ordering::Relaxed)),
            ("bytes", s.bytes.load(Ordering::Relaxed)),
            ("inflight", self.inflight.len() as u64),
        ]
    }

    /// Write the counters plus this process id as JSON, atomically (tmp + rename).
    fn stats_dump(&self, path: &str) -> std::io::Result<()> {
        let mut json = String::from("{");
        for (k, v) in self.stats_snapshot() {
            json.push_str(&format!("\"{}\":{},", k, v));
        }
        json.push_str(&format!("\"pid\":{}}}", std::process::id()));
        let tmp = format!("{}.{}.tmp", path, std::process::id());
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }
}

#[pyclass]
struct RustS3Cache {
    inner: Downloader,
}

#[pymethods]
impl RustS3Cache {
    #[new]
    #[pyo3(signature = (cache_dir, max_concurrent=1024, worker_threads=None))]
    fn new(
        cache_dir: String,
        max_concurrent: usize,
        worker_threads: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Downloader::new(cache_dir, max_concurrent, worker_threads)?,
        })
    }

    /// Runtime worker threads asked for at construction (None = tokio default).
    #[getter]
    fn worker_threads(&self) -> Option<usize> {
        self.inner.worker_threads
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

    /// Block until the cached file for `uri` exists or `timeout_ms` passes.
    /// Returns True when the file is there. Never downloads. Releases the GIL.
    fn wait_for_path(&self, py: Python<'_>, uri: String, timeout_ms: u64) -> bool {
        py.allow_threads(|| {
            self.inner
                .wait_for_path(&uri, Duration::from_millis(timeout_ms))
        })
    }

    /// Unlink the cached files for `uris` in the background; returns at once.
    fn drop_paths(&self, py: Python<'_>, uris: Vec<String>) {
        py.allow_threads(|| self.inner.drop_paths(uris))
    }

    /// Wait for all inflight downloads and pending unlinks. Releases the GIL while waiting.
    fn wait_for_inflight(&self, py: Python<'_>) {
        py.allow_threads(|| self.inner.wait_for_inflight())
    }

    /// Get the local path for a URI (doesn't download)
    fn path_for(&self, uri: String) -> String {
        self.inner.path_for(&uri).to_string_lossy().to_string()
    }

    /// Download counters: attempts, retries, throttled, failures, not_found, downloaded, bytes, inflight
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in self.inner.stats_snapshot() {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Write the counters and this pid to `path` as JSON (atomic tmp + rename).
    fn stats_dump(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.inner.stats_dump(&path))
            .map_err(|e| PyRuntimeError::new_err(format!("stats_dump {}: {}", path, e)))
    }
}

#[pymodule]
fn planfetch_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustS3Cache>()?;
    Ok(())
}

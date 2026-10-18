# cli.py
"""Command-line interface for planfetch."""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import click
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .cache import SharedDiskCache
from .dataloader import PrefetchCoordinator
from .plans import (
    PlanProvider,
    PlanView,
    SequentialPlanProvider,
    ShuffleOncePerEpochPlanProvider,
    StreamingStratifiedBatchPlanProvider,
)

console = Console()

# Above this the training thread is paying for prefetch bookkeeping it should not see.
COORDINATOR_WARN_MS = 5.0


def load_csv_subset(
    csv_path: str,
    *,
    s3_uri_col: str,
    class_col: str,
    limit: Optional[int] = None,
    offset: int = 0,
    sample_frac: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Load a CSV with optional subsetting.

    Args:
        csv_path: Path to CSV file
        s3_uri_col: Column name for S3 URIs
        class_col: Column name for class labels
        limit: Maximum number of rows to load (after offset)
        offset: Number of rows to skip from start
        sample_frac: Random sample fraction (0.0-1.0), applied after limit/offset
        seed: Random seed for sampling

    Returns:
        Tuple of (s3_uris list, labels array)
    """
    s3_uris: List[str] = []
    labels: List[int] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise click.ClickException("CSV has no header row.")
        if s3_uri_col not in reader.fieldnames:
            raise click.ClickException(
                f"Column '{s3_uri_col}' not found. Available: {reader.fieldnames}"
            )
        if class_col not in reader.fieldnames:
            raise click.ClickException(
                f"Column '{class_col}' not found. Available: {reader.fieldnames}"
            )

        for i, row in enumerate(reader):
            if i < offset:
                continue
            if limit is not None and (i - offset) >= limit:
                break
            s3_uris.append(row[s3_uri_col])
            labels.append(int(row[class_col]))

    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        n = len(s3_uris)
        k = max(1, int(n * sample_frac))
        random.seed(seed)
        indices = sorted(random.sample(range(n), k))
        s3_uris = [s3_uris[i] for i in indices]
        labels = [labels[i] for i in indices]

    return s3_uris, np.asarray(labels, dtype=np.int64)


def create_plan_provider(
    strategy: str,
    n: int,
    batch_size: int,
    labels: np.ndarray,
    seed: int = 42,
) -> PlanProvider:
    """Create a plan provider based on strategy name."""
    if strategy == "sequential":
        return SequentialPlanProvider(n=n, batch_size=batch_size, drop_last=True)
    elif strategy == "shuffle":
        return ShuffleOncePerEpochPlanProvider(
            n=n, batch_size=batch_size, seed=seed, epoch_samples=n, drop_last=True
        )
    elif strategy == "stratified":
        return StreamingStratifiedBatchPlanProvider(
            labels=labels,
            batch_size=batch_size,
            seed=seed,
            epoch_samples=n,
            drop_last=True,
        )
    else:
        raise click.ClickException(f"Unknown strategy: {strategy}")


@click.group()
@click.version_option()
def cli() -> None:
    """planfetch - S3 prefetching utilities for PyTorch."""
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
# CSV selection options
@click.option("-n", "--limit", type=int, default=None, help="Process only first N rows")
@click.option("--offset", type=int, default=0, help="Skip first N rows")
@click.option(
    "--sample-frac",
    type=float,
    default=None,
    help="Random sample fraction (0.0-1.0)",
)
@click.option(
    "--s3-uri-col", type=str, default="s3_uri", help="Column name for S3 URIs"
)
@click.option(
    "--class-col", type=str, default="label", help="Column name for class labels"
)
# Batching
@click.option("--batch-size", type=int, default=64, help="Batch size")
# Cache configuration
@click.option(
    "--cache-dir",
    type=click.Path(),
    required=True,
    help="Local cache directory on the disk you will train from (e.g. NVMe). "
    "Required: a temp dir on the root volume would measure the wrong disk.",
)
@click.option(
    "--max-bytes",
    type=int,
    default=10_000_000_000,
    help="Maximum cache size in bytes (default: 10GB)",
)
@click.option(
    "--download-threads", type=int, default=64, help="Number of download threads"
)
@click.option(
    "--s3-pool-connections",
    type=int,
    default=256,
    help="Boto3 connection pool size",
)
# Prefetch tuning
@click.option(
    "--batches-ahead", type=int, default=200, help="Batches to prefetch ahead"
)
@click.option(
    "--batches-back", type=int, default=4, help="Batches to keep behind current"
)
@click.option(
    "--prune-every",
    type=int,
    default=0,
    help="Full prune sweep every N batches (0 = off; per-step eviction is always on)",
)
@click.option(
    "--prune-mode",
    type=click.Choice(["window", "pressure"]),
    default="window",
    help="Prune mode",
)
@click.option(
    "--prune-backend",
    type=click.Choice(["meta", "scan"]),
    default="scan",
    help="Prune backend for the sweep",
)
@click.option(
    "--rust/--no-rust",
    "use_rust",
    default=True,
    help="Use the Rust downloader when built (default). --no-rust forces the boto3 "
    "backend, a diagnostic fallback: its download threads hold the GIL and starve "
    "the training thread, so its numbers are not what a training loop would see",
)
@click.option(
    "--read-files/--no-read-files",
    default=True,
    help="Read each file's bytes after fetch (what training does) instead of only stat()",
)
# Plan strategy
@click.option(
    "--strategy",
    type=click.Choice(["shuffle", "sequential", "stratified"]),
    default="shuffle",
    help="Batch planning strategy",
)
# Output
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--seed", type=int, default=42, help="Random seed")
def speedtest(
    csv_path: str,
    limit: Optional[int],
    offset: int,
    sample_frac: Optional[float],
    s3_uri_col: str,
    class_col: str,
    batch_size: int,
    cache_dir: str,
    max_bytes: int,
    download_threads: int,
    s3_pool_connections: int,
    batches_ahead: int,
    batches_back: int,
    prune_every: int,
    prune_mode: str,
    prune_backend: str,
    use_rust: bool,
    read_files: bool,
    strategy: str,
    verbose: bool,
    seed: int,
) -> None:
    """
    Benchmark S3 prefetching performance.

    Loads a CSV of S3 URIs and measures batched fetch performance with
    configurable prefetching parameters. Use this to find optimal settings
    before training on GPUs. The loop is the same PrefetchCoordinator a
    training job uses, so coordinator overhead is reported separately from
    fetch wait.
    """
    # Load CSV
    console.print("[bold blue]Loading CSV...[/bold blue]")
    s3_uris, labels = load_csv_subset(
        csv_path,
        s3_uri_col=s3_uri_col,
        class_col=class_col,
        limit=limit,
        offset=offset,
        sample_frac=sample_frac,
        seed=seed,
    )
    n = len(s3_uris)

    if n == 0:
        raise click.ClickException("No samples loaded from CSV")

    console.print(f"[green]Loaded {n:,} samples[/green]")

    # Setup cache
    console.print(f"[dim]Using cache: {cache_dir}[/dim]")
    cache = SharedDiskCache(
        cache_dir,
        max_bytes=max_bytes,
        download_threads=download_threads,
        s3_max_pool_connections=s3_pool_connections,
        verbose=verbose,
        use_rust=use_rust,
    )
    backend = "rust" if getattr(cache, "_use_rust", False) else "python (boto3)"

    # Create plan
    console.print(f"[bold blue]Creating {strategy} plan...[/bold blue]")
    provider = create_plan_provider(strategy, n, batch_size, labels, seed=seed)
    plan: PlanView = provider.make_epoch_plan(epoch=0)
    num_batches = len(plan)

    console.print(f"[green]Plan: {num_batches:,} batches of {batch_size}[/green]")

    # Display config
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Samples", f"{n:,}")
    config_table.add_row("Batch size", str(batch_size))
    config_table.add_row("Num batches", f"{num_batches:,}")
    config_table.add_row("Strategy", strategy)
    config_table.add_row("Backend", backend)
    config_table.add_row("Download threads", str(download_threads))
    config_table.add_row("Batches ahead", str(batches_ahead))
    config_table.add_row("Batches back", str(batches_back))
    config_table.add_row("Prune every", str(prune_every))
    config_table.add_row("Prune mode", prune_mode)
    config_table.add_row("Max cache", f"{max_bytes / 1e9:.1f} GB")
    console.print(config_table)
    console.print()

    # Run benchmark
    coordinator = PrefetchCoordinator(
        cache,
        s3_uris,
        batches_ahead=batches_ahead,
        batches_back=batches_back,
        prune_every=prune_every,
        prune_mode=prune_mode,
        prune_backend=prune_backend,
    )
    coordinator.set_plan(plan)

    batch_times: List[float] = []
    file_times: List[float] = []
    coord_times: List[float] = []
    total_bytes = 0
    wall_start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Fetching batches...", total=num_batches)

        # Prime initial prefetch window
        t0 = time.perf_counter()
        coordinator.prime()
        coord_times.append(time.perf_counter() - t0)

        for batch_idx in range(num_batches):
            batch_start = time.perf_counter()

            # Ensure all files are local (blocking), timing each file
            for i in plan.batch(batch_idx):
                uri = s3_uris[int(i)]
                f0 = time.perf_counter()
                path = cache.ensure_local(uri)
                if read_files:
                    with open(path, "rb") as fh:
                        total_bytes += len(fh.read())
                else:
                    try:
                        total_bytes += Path(path).stat().st_size
                    except OSError:
                        pass
                file_times.append(time.perf_counter() - f0)

            batch_end = time.perf_counter()
            batch_times.append(batch_end - batch_start)

            # Advance the window: prime the entering batch, drop the leaving one
            coordinator.on_batch_yielded(batch_idx)
            coord_times.append(time.perf_counter() - batch_end)

            progress.update(task, advance=1)

    # Wait for any remaining downloads
    cache.wait_for_inflight()
    wall_end = time.perf_counter()
    wall_time = wall_end - wall_start

    # Compute statistics using wall-clock time (actual throughput)
    batch_arr = np.array(batch_times) * 1000.0
    file_arr = np.array(file_times) * 1000.0
    coord_arr = np.array(coord_times) * 1000.0
    batches_per_sec = num_batches / wall_time if wall_time > 0 else 0
    samples_per_sec = (num_batches * batch_size) / wall_time if wall_time > 0 else 0
    coord_per_batch = float(coord_arr[1:].mean()) if len(coord_arr) > 1 else 0.0
    stats = cache.stats()

    # Display results
    console.print()
    results_table = Table(title="Results", show_header=False)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_row("Total time", f"{wall_time:.2f} s")
    results_table.add_row("Batches/sec", f"{batches_per_sec:.2f}")
    results_table.add_row("Samples/sec", f"{samples_per_sec:.2f}")
    results_table.add_row("Throughput", f"{total_bytes / wall_time / 1e6:.1f} MB/s")
    results_table.add_row("Avg batch wait", f"{batch_arr.mean():.2f} ms")
    results_table.add_row("P50 batch wait", f"{np.percentile(batch_arr, 50):.2f} ms")
    results_table.add_row("P95 batch wait", f"{np.percentile(batch_arr, 95):.2f} ms")
    results_table.add_row("P99 batch wait", f"{np.percentile(batch_arr, 99):.2f} ms")
    results_table.add_row("P50 file wait", f"{np.percentile(file_arr, 50):.3f} ms")
    results_table.add_row("P95 file wait", f"{np.percentile(file_arr, 95):.3f} ms")
    results_table.add_row("P99 file wait", f"{np.percentile(file_arr, 99):.3f} ms")
    results_table.add_row("Coordinator / batch", f"{coord_per_batch:.2f} ms")
    results_table.add_row("Total downloaded", _fmt_bytes(total_bytes))
    results_table.add_row("S3 GET attempts", f"{stats.get('attempts', 0):,}")
    results_table.add_row("Retries", f"{stats.get('retries', 0):,}")
    results_table.add_row("Throttled (503)", f"{stats.get('throttled', 0):,}")
    results_table.add_row("Failures", f"{stats.get('failures', 0):,}")
    results_table.add_row("Not found", f"{stats.get('not_found', 0):,}")
    console.print(results_table)
    if stats.get("retries", 0) or stats.get("throttled", 0):
        console.print(
            "[yellow]S3 retried or throttled requests: slow batches above may be "
            "backoff, not S3 latency. Spread keys over more prefixes or lower "
            "--download-threads.[/yellow]"
        )
    if stats.get("not_found", 0):
        console.print(
            "[yellow]Some keys do not exist in S3: fix the index, or give "
            "S3PathDataset an on_error handler so training skips them.[/yellow]"
        )
    if coord_per_batch > COORDINATOR_WARN_MS:
        console.print(
            f"[yellow]Coordinator time is {coord_per_batch:.1f} ms per batch "
            f"(expected under {COORDINATOR_WARN_MS:g} ms). On the boto3 backend this is "
            "GIL starvation by the download threads; on the Rust backend check the disk "
            "the cache is on.[/yellow]"
        )


def _fmt_bytes(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    return f"{n / 1e3:.2f} KB"


if __name__ == "__main__":
    cli()

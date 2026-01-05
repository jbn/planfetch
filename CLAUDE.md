- **This is a `uv` based python project.** 
    - Use `uv run` instead of `python` 
    - Use `uv add` instead of `pip install`. 
    - Use `pytest` for testing.
- Follow TDD. 
    - Write the test first; then implement it; then make sure tests pass.
    - After tests pass use `uv run ty check` to fix any type issues and `uv run black` to format code

## Project Overview

planfetch is an S3 prefetching system for PyTorch DataLoaders. It downloads batches ahead of time to a local NVMe cache, minimizing S3 latency in ML training workloads.

## Key Directories

- `src/planfetch/` - Main package
  - `cache.py` - SharedDiskCache (S3 downloads, file locking, pruning)
  - `plans.py` - PlanView/PlanProvider (batch scheduling strategies)
  - `dataloader.py` - PrefetchCoordinator, PrefetchingLoader
  - `dataset.py` - S3PathDataset (PyTorch Dataset wrapper)
  - `io.py` - CSV/DataFrame index loading with pack_uris optimization
  - `uri_storage.py` - Memory-efficient S3 URI storage
  - `lightning.py` - PyTorch Lightning callback
  - `cli.py` - Speedtest CLI
- `rust/` - Optional Rust extension for high-perf async S3 downloads
- `tests/` - Test files (mirror src structure)

## Common Commands

```bash
# Testing
uv run pytest                      # Run all tests
uv run pytest tests/test_cache.py  # Run specific module
uv run pytest -k "test_name"       # Run specific test

# Type checking
uv run ty                          # Check/fix type issues

# Package management
uv add <package>                   # Add dependency
uv sync                            # Install from lock file

# CLI
uv run planfetch speedtest <csv> --batch-size 64 --batches-ahead 200

# Build (includes Rust extension)
uv build
```
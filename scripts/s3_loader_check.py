#!/usr/bin/env python3
"""Real-S3 check of the DataLoader path: forked (or spawned) workers, no worker_init_fn.

The unit suite runs against a fake S3 client and the Python backend, so it
cannot see what the Rust backend does inside a forked worker (the 2026-10-17
runtime-drop hang, for one). Run this against a real bucket before a release:

    AWS_REGION=us-east-1 uv run python scripts/s3_loader_check.py \
        --index index.csv --cache-dir /nvme/planfetch-cache --workers 8 --out w8.json
    uv run python scripts/s3_loader_check.py ... --workers 8 --mp-context spawn
    uv run python scripts/s3_loader_check.py ... --workers 8 --epoch1-batches 200

index.csv has columns s3_uri,label (at least --rows rows). It builds
SharedDiskCache + S3PathDataset + ShuffleOncePerEpochPlanProvider +
build_prefetching_dataloader (or a hand-built DataLoader when --mp-context is
given), iterates --batches batches with a simulated --step-sleep training step,
times every next() and the coordinator separately, and writes a JSON report
with cache.stats(all_processes=True). A watchdog treats --hang-seconds without
progress as a hang: it dumps every worker with py-spy (if installed; needs
sudo for other processes), kills them and exits 3.

Pass criteria used so far: no hang, no exception, retries 0, next() minus the
coordinator p95 under 7.5 ms at a 150 ms step, cache never above
(batches_ahead + batches_back) * batch_size files.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import planfetch as pf


Image.MAX_IMAGE_PIXELS = None  # the training set has a few >89 Mpixel images


def load_rgb(path: str):
    """PIL open + convert("RGB"); module-level so a spawn context can pickle it."""
    return Image.open(path).convert("RGB")


def child_pids(parent: int) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                fields = f.read().split()
            if int(fields[3]) == parent:
                out.append(int(d))
        except Exception:
            pass
    return out


def dump_workers(report_path: str) -> None:
    pids = child_pids(os.getpid())
    with open(report_path, "a") as f:
        f.write(f"\n=== HANG at {time.strftime('%H:%M:%S')} ; children={pids}\n")
        f.write("--- main process (faulthandler) ---\n")
        f.flush()
        faulthandler.dump_traceback(file=f, all_threads=True)
        for p in pids:
            f.write(f"\n--- worker pid {p} (py-spy) ---\n")
            f.flush()
            try:
                subprocess.run(
                    [
                        "sudo",
                        "-n",
                        shutil.which("py-spy") or "py-spy",
                        "dump",
                        "--pid",
                        str(p),
                    ],
                    stdout=f,
                    stderr=f,
                    timeout=60,
                )
            except Exception as e:
                f.write(f"py-spy failed: {e!r}\n")
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except Exception:
            pass


def all_process_stats(cache):
    """cache.stats(all_processes=True) where the installed planfetch has it."""
    try:
        return cache.stats(all_processes=True)
    except TypeError:
        return None


def pct(xs, q):
    return float(np.percentile(np.asarray(xs) * 1000.0, q)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--rows", type=int, default=64000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--batches-ahead", type=int, default=200)
    ap.add_argument("--batches-back", type=int, default=4)
    ap.add_argument("--download-threads", type=int, default=128)
    ap.add_argument("--step-sleep", type=float, default=0.150)
    ap.add_argument("--steady-from", type=int, default=50)
    ap.add_argument("--epoch1-batches", type=int, default=0)
    ap.add_argument(
        "--mp-context", default=None, help="e.g. spawn: build the DataLoader by hand"
    )
    ap.add_argument("--no-rust", action="store_true")
    ap.add_argument("--hang-seconds", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    report: dict = {
        "args": vars(a),
        "pid": os.getpid(),
        "exception": None,
        "hang": False,
    }
    hang_log = a.out + ".hang.txt"

    def save():
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=str)

    t_start = time.perf_counter()
    uris, labels = pf.load_index_csv(a.index, s3_uri_col="s3_uri", class_col="label")
    n_rows = min(a.rows, len(uris))
    uris, labels = [uris[i] for i in range(n_rows)], labels[:n_rows]
    n = len(uris)
    report["n"] = n

    cache = pf.SharedDiskCache(
        a.cache_dir,
        max_bytes=int(200e9),
        download_threads=a.download_threads,
        use_rust=not a.no_rust,
    )
    report["backend"] = "rust" if cache._use_rust else "python"
    ds = pf.S3PathDataset(
        uris,
        labels,
        cache=cache,
        image_loader=load_rgb,
        transform=transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.ToTensor()]
        ),
    )
    provider = pf.ShuffleOncePerEpochPlanProvider(
        n, batch_size=a.batch_size, seed=1, epoch_samples=n
    )

    t_build = time.perf_counter()
    try:
        if a.mp_context:
            plan0 = provider.make_epoch_plan(0)
            sampler = pf.PlanBatchSampler(plan0)
            coord = pf.PrefetchCoordinator(
                cache, uris, batches_ahead=a.batches_ahead, batches_back=a.batches_back
            )
            coord.check_loader_depth(a.workers, 2)
            controller = pf.PrefetchController(provider, sampler, coord)
            dl = DataLoader(
                ds,
                batch_sampler=sampler,  # type: ignore[arg-type]
                num_workers=a.workers,
                pin_memory=False,
                persistent_workers=False,
                multiprocessing_context=a.mp_context,
            )
            loader = pf.PrefetchingLoader(dl, coord)
        else:
            loader, controller = pf.build_prefetching_dataloader(
                dataset=ds,
                s3_uris=uris,
                plan_provider=provider,
                cache=cache,
                num_workers=a.workers,
                batches_ahead=a.batches_ahead,
                batches_back=a.batches_back,
                pin_memory=False,
                persistent_workers=False,
            )
    except Exception as e:
        report["exception"] = {
            "where": "build",
            "type": type(e).__name__,
            "message": str(e),
        }
        report["build_seconds"] = time.perf_counter() - t_build
        save()
        print(f"BUILD RAISED {type(e).__name__}: {e}")
        return 2

    # Watchdog: progress counter bumped per batch; no bump for hang_seconds => dump + kill.
    progress = {"n": 0, "t": time.perf_counter(), "stop": False}

    def watchdog():
        last = -1
        while not progress["stop"]:
            time.sleep(1.0)
            if progress["n"] != last:
                last = progress["n"]
                progress["t"] = time.perf_counter()
            elif time.perf_counter() - progress["t"] > a.hang_seconds:
                report["hang"] = True
                report["hang_at_batch"] = progress["n"]
                save()
                dump_workers(hang_log)
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()

    # Time the coordinator separately: PrefetchingLoader calls on_batch_yielded inside next().
    coord_times: list[float] = []
    coord = loader.prefetch
    _orig_oby = coord.on_batch_yielded

    def _timed_oby(bi):
        c0 = time.perf_counter()
        _orig_oby(bi)
        coord_times.append(time.perf_counter() - c0)

    setattr(coord, "on_batch_yielded", _timed_oby)

    def run_epoch(epoch: int, nb: int, key: str):
        coord_times.clear()
        t0 = time.perf_counter()
        controller.set_epoch(epoch)
        t_set = time.perf_counter() - t0
        stalls: list[float] = []
        marks: dict = {}
        it = iter(loader)
        t_first = None
        try:
            for i in range(nb):
                s0 = time.perf_counter()
                batch = next(it)
                s1 = time.perf_counter()
                if t_first is None:
                    t_first = s1 - t0
                stalls.append(s1 - s0)
                progress["n"] += 1
                if i == a.steady_from:
                    marks["t_steady0"] = s1
                if a.step_sleep > 0:
                    time.sleep(a.step_sleep)
        finally:
            t_end = time.perf_counter()
            marks["t_end"] = t_end
        del it  # stop workers
        steady_n = nb - a.steady_from - 1
        steady_s = marks["t_end"] - marks.get("t_steady0", t0)
        return {
            "batches": nb,
            "set_epoch_seconds": t_set,
            "first_batch_seconds": t_first,
            "wall_seconds": t_end - t0,
            "steady_samples_per_s": (
                steady_n * a.batch_size / steady_s if steady_n > 0 else None
            ),
            "steady_batches_per_s": steady_n / steady_s if steady_n > 0 else None,
            "stall_ms": {
                "p50": pct(stalls, 50),
                "p95": pct(stalls, 95),
                "p99": pct(stalls, 99),
                "max": pct(stalls, 100),
                "total_s": float(sum(stalls)),
            },
            "stall_steady_ms": {
                "p50": pct(stalls[a.steady_from :], 50),
                "p95": pct(stalls[a.steady_from :], 95),
                "p99": pct(stalls[a.steady_from :], 99),
            },
            "coord_ms": {
                "p50": pct(coord_times, 50),
                "p95": pct(coord_times, 95),
                "p99": pct(coord_times, 99),
                "mean": float(np.mean(coord_times) * 1000) if coord_times else None,
                "total_s": float(sum(coord_times)),
            },
            "stall_minus_coord_ms": {
                "p50": pct([s - c for s, c in zip(stalls, coord_times)], 50),
                "p95": pct([s - c for s, c in zip(stalls, coord_times)], 95),
                "p99": pct([s - c for s, c in zip(stalls, coord_times)], 99),
            },
            "cache_stats_parent": cache.stats(),
            "cache_stats_all_processes": all_process_stats(cache),
        }

    try:
        report["epoch0"] = run_epoch(0, a.batches, "epoch0")
        report["startup_seconds_total"] = report["epoch0"]["first_batch_seconds"] + (
            t_build - t_start
        )
        save()
        if a.epoch1_batches > 0:
            report["epoch1"] = run_epoch(1, a.epoch1_batches, "epoch1")
            cache.wait_for_inflight()
            report["epoch1"]["cache_stats_after_wait"] = cache.stats()
    except Exception as e:
        report["exception"] = {
            "where": "iterate",
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(traceback.format_exc())
    progress["stop"] = True
    time.sleep(1.5)  # let workers exit
    cnt = 0
    for root, _dirs, files in os.walk(a.cache_dir):
        cnt += sum(1 for f in files if len(f) == 64)
    report["cache_files_at_end"] = cnt
    report["cache_stats_end"] = cache.stats()
    report["cache_stats_all_processes_end"] = all_process_stats(cache)
    save()
    print(
        json.dumps(
            {
                k: report[k]
                for k in report
                if k
                in (
                    "backend",
                    "epoch0",
                    "epoch1",
                    "exception",
                    "cache_files_at_end",
                    "hang",
                )
            },
            indent=1,
            default=str,
        )
    )
    return 0 if report["exception"] is None else 1


if __name__ == "__main__":
    sys.exit(main())

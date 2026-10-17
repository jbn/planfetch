# dataloader.py
"""DataLoader integration with prefetching and batch sampling."""
from __future__ import annotations

from collections import Counter
from typing import Iterator, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .cache import SharedDiskCache
from .plans import PlanProvider, PlanView
from .uri_storage import S3KeyList, S3URISequence


class PlanBatchSampler(BatchSampler):
    """
    BatchSampler that yields batches from a PlanView.
    """

    def __init__(self, plan: PlanView):
        self.plan = plan

    def set_plan(self, plan: PlanView) -> None:
        self.plan = plan

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        for i in range(len(self.plan)):
            yield self.plan.batch(i)

    def __len__(self) -> int:
        return len(self.plan)


class PrefetchCache(Protocol):
    """What PrefetchCoordinator needs from a cache (SharedDiskCache satisfies it).

    `drop_async` is optional: when the cache has it, per-step eviction goes
    through it and leaves the training thread; otherwise `drop` is used."""

    def prime_prefetch(self, s3_uris: Sequence[str]) -> None: ...

    def drop(self, s3_uris: Sequence[str]) -> int: ...

    def prune(
        self, *, keep_uris: Sequence[str], mode: str = ..., backend: str = ...
    ) -> None: ...


class PrefetchCoordinator:
    """
    Sliding-window prefetch + prune, incremental.

    Keeps batches:
      [batch_idx - B_back, batch_idx + B_ahead)

    Each step primes only the batches entering the window and unlinks only the
    batches leaving it (O(batch size) per step). The old behaviour of re-priming
    the whole window every step and walking the cache directory every few
    batches cost ~163 ms per batch on the training thread at 200 batches ahead
    (FSx benchmark, 2026-10-16); this costs ~6 ms.

    A URI that appears in several batches of the window is kept until the last
    of them leaves. When the plan changes (set_plan, or a jump backwards), files
    of the old window that are not in the new one are dropped.

    `prune_every` (0 = off) additionally runs a full `cache.prune` sweep with the
    current window as keep-set every N steps. That sweep walks the cache
    directory when `prune_backend="scan"`, so keep it rare; it only catches
    orphans (files from a crashed run, or files a worker downloaded on a miss
    outside the window).

    This is designed to keep training loops unobtrusive:
      - PrefetchingLoader calls on_batch_yielded automatically

    Accepts either List[str] of full URIs or S3KeyList for memory efficiency.
    """

    def __init__(
        self,
        cache: PrefetchCache,
        s3_uris: S3URISequence,
        *,
        batches_ahead: int,
        batches_back: int,
        prune_every: int = 0,
        prune_mode: str = "window",
        prune_backend: str = "scan",
    ):
        self.cache = cache
        # Store efficiently based on input type
        if isinstance(s3_uris, S3KeyList):
            self._uri_storage: S3URISequence = s3_uris
        else:
            self._uri_storage = list(s3_uris)
        self.Ba = int(batches_ahead)
        self.Bb = int(batches_back)
        if self.Ba < 1:
            raise ValueError("batches_ahead must be >= 1")
        if self.Bb < 0:
            raise ValueError("batches_back must be >= 0")
        self.prune_every = int(prune_every)
        self.prune_mode = str(prune_mode)
        self.prune_backend = str(prune_backend)

        self.plan: Optional[PlanView] = None
        self._step = 0
        self._drop = getattr(cache, "drop_async", None) or cache.drop
        # Incremental window state (see _advance).
        self._counts: Counter[str] = Counter()  # uri -> batches in window using it
        self._primed_hi = 0  # batches [0, _primed_hi) have been primed
        self._dropped_lo = 0  # batches [0, _dropped_lo) have been dropped
        self._last_idx = -1
        self._stale: Set[str] = set()  # old-window uris to drop after next prime

    @property
    def s3_uris(self) -> S3URISequence:
        """URI storage (S3KeyList or List[str])."""
        return self._uri_storage

    def set_plan(self, plan: PlanView) -> None:
        self.plan = plan
        self._step = 0
        self._reset_window()

    def check_loader_depth(
        self, num_workers: int, prefetch_factor: Optional[int] = 2
    ) -> None:
        """
        Raise if the DataLoader can have more batches in flight than the window
        reaches ahead. Workers run up to num_workers * prefetch_factor batches
        ahead of the training loop; the coordinator only keeps [idx - back,
        idx + ahead) on disk, so anything beyond that would be dropped while a
        worker reads it (that was the Phase 0 hang in the FSx benchmark).
        """
        if num_workers <= 0:
            return
        depth = int(num_workers) * int(prefetch_factor or 2)
        if self.Ba <= depth:
            raise ValueError(
                f"batches_ahead={self.Ba} must exceed num_workers * prefetch_factor "
                f"= {num_workers} * {prefetch_factor or 2} = {depth}, or the prefetch "
                "window will drop files that DataLoader workers are still reading."
            )

    # ------------------------------------------------------------------ #
    # Window arithmetic
    # ------------------------------------------------------------------ #

    def _window_bounds(self, batch_idx: int) -> Tuple[int, int]:
        assert self.plan is not None
        lo = max(0, int(batch_idx) - self.Bb)
        hi = min(len(self.plan), int(batch_idx) + self.Ba)
        return lo, hi

    def _uris_for(self, lo: int, hi: int) -> List[str]:
        assert self.plan is not None
        idxs = self.plan.range_flat(lo, hi)
        return [self._uri_storage[int(i)] for i in idxs]

    def _reset_window(self) -> None:
        """Forget the current window; its files are dropped after the next prime."""
        self._stale |= set(self._counts)
        self._counts.clear()
        self._primed_hi = 0
        self._dropped_lo = 0
        self._last_idx = -1

    def _flush_stale(self) -> None:
        if not self._stale:
            return
        gone = sorted(u for u in self._stale if u not in self._counts)
        self._stale.clear()
        if gone:
            self._drop(gone)

    def _advance(self, batch_idx: int) -> None:
        """Bring the on-disk window to [batch_idx - Bb, batch_idx + Ba)."""
        assert self.plan is not None
        if batch_idx < self._last_idx:
            # Went backwards (restart / resume): rebuild the window from scratch.
            self._reset_window()
        lo, hi = self._window_bounds(batch_idx)

        if self._primed_hi < lo:
            # Jumped past the primed region: nothing in [primed_hi, lo) is needed.
            self._primed_hi = lo
        if hi > self._primed_hi:
            entering = self._uris_for(self._primed_hi, hi)
            self._counts.update(entering)
            self.cache.prime_prefetch(entering)
            self._primed_hi = hi

        if lo > self._dropped_lo:
            leaving = self._uris_for(self._dropped_lo, lo)
            self._dropped_lo = lo
            gone: List[str] = []
            for u in leaving:
                c = self._counts[u] - 1
                if c <= 0:
                    del self._counts[u]
                    gone.append(u)
                else:
                    self._counts[u] = c
            if gone:
                self._drop(gone)

        self._flush_stale()
        self._last_idx = batch_idx

    # ------------------------------------------------------------------ #
    # Public stepping API
    # ------------------------------------------------------------------ #

    def prime(self) -> None:
        """Prime the first window of the current plan."""
        if self.plan is None or len(self.plan) == 0:
            return
        self._advance(0)
        self._last_idx = -1  # batch 0 has not been yielded yet

    def on_batch_yielded(self, batch_idx: int) -> None:
        if self.plan is None:
            return
        self._advance(batch_idx)

        self._step += 1
        if self.prune_every > 0 and self._step % self.prune_every == 0:
            self.cache.prune(
                keep_uris=list(self._counts),
                mode=self.prune_mode,
                backend=self.prune_backend,
            )


class PrefetchingLoader:
    """
    Wrap a DataLoader and automatically advance prefetching per yielded batch.
    """

    def __init__(self, dataloader: DataLoader, prefetch: PrefetchCoordinator):
        self.dataloader = dataloader
        self.prefetch = prefetch

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        for bi, batch in enumerate(self.dataloader):
            self.prefetch.on_batch_yielded(bi)
            yield batch


class PrefetchController:
    """
    One method per epoch: set_epoch(epoch).
    """

    def __init__(
        self,
        plan_provider: PlanProvider,
        batch_sampler: PlanBatchSampler,
        prefetch: PrefetchCoordinator,
    ):
        self.plan_provider = plan_provider
        self.batch_sampler = batch_sampler
        self.prefetch = prefetch

    def set_epoch(self, epoch: int) -> None:
        plan = self.plan_provider.make_epoch_plan(epoch)
        self.batch_sampler.set_plan(plan)
        self.prefetch.set_plan(plan)
        self.prefetch.prime()


def build_prefetching_dataloader(
    *,
    dataset: Dataset,
    s3_uris: S3URISequence,
    plan_provider: PlanProvider,
    cache: SharedDiskCache,
    num_workers: int,
    batches_ahead: int,
    batches_back: int,
    prune_every: int = 0,
    prune_mode: str = "window",
    prune_backend: str = "scan",
    pin_memory: bool = True,
    persistent_workers: bool = True,
    worker_init_fn=None,
    prefetch_factor: Optional[int] = None,
    miss_wait_seconds: float = 30.0,
) -> Tuple[PrefetchingLoader, PrefetchController]:
    """
    Create:
      - DataLoader configured with a PlanBatchSampler
      - PrefetchingLoader wrapper
      - PrefetchController to advance epoch plans

    You call controller.set_epoch(epoch) once per epoch.

    `batches_ahead` must exceed `num_workers * prefetch_factor` (DataLoader's
    default prefetch_factor is 2), otherwise workers read batches the window
    has already dropped; a ValueError is raised in that case.

    The cache is shared with the workers as-is: SharedDiskCache rebuilds its
    download backend per process, so no worker_init_fn is needed for it.

    Under this loader a cache miss in a worker is almost always a file the
    parent is already downloading, so if `dataset` has a `miss_wait_seconds`
    attribute left at 0 it is set to `miss_wait_seconds`: the worker waits for
    the file instead of fetching it a second time. A non-zero value already on
    the dataset is kept.
    """
    if (
        miss_wait_seconds > 0
        and hasattr(dataset, "miss_wait_seconds")
        and not getattr(dataset, "miss_wait_seconds")
    ):
        dataset.miss_wait_seconds = float(miss_wait_seconds)  # type: ignore[attr-defined]

    plan0 = plan_provider.make_epoch_plan(0)
    sampler = PlanBatchSampler(plan0)

    prefetch = PrefetchCoordinator(
        cache=cache,
        s3_uris=s3_uris,
        batches_ahead=batches_ahead,
        batches_back=batches_back,
        prune_every=prune_every,
        prune_mode=prune_mode,
        prune_backend=prune_backend,
    )

    prefetch.check_loader_depth(int(num_workers), prefetch_factor or 2)

    controller = PrefetchController(plan_provider, sampler, prefetch)

    dl_kwargs = dict(
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init_fn,
    )
    if int(num_workers) > 0:
        # DataLoader rejects these when num_workers == 0.
        dl_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = int(prefetch_factor)
    dl = DataLoader(dataset, **dl_kwargs)  # type: ignore[arg-type]

    return PrefetchingLoader(dl, prefetch), controller

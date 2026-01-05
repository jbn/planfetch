# dataloader.py
"""DataLoader integration with prefetching and batch sampling."""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

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


class PrefetchCoordinator:
    """
    Sliding-window prefetch + prune.

    Keeps batches:
      [batch_idx - B_back, batch_idx + B_ahead)

    This is designed to keep training loops unobtrusive:
      - PrefetchingLoader calls on_batch_yielded automatically

    Accepts either List[str] of full URIs or S3KeyList for memory efficiency.
    """

    def __init__(
        self,
        cache: SharedDiskCache,
        s3_uris: S3URISequence,
        *,
        batches_ahead: int,
        batches_back: int,
        prune_every: int = 5,
        prune_mode: str = "window",
        prune_backend: str = "meta",
    ):
        self.cache = cache
        # Store efficiently based on input type
        if isinstance(s3_uris, S3KeyList):
            self._uri_storage: S3URISequence = s3_uris
        else:
            self._uri_storage = list(s3_uris)
        self.Ba = int(batches_ahead)
        self.Bb = int(batches_back)
        self.prune_every = int(prune_every)
        self.prune_mode = str(prune_mode)
        self.prune_backend = str(prune_backend)

        self.plan: Optional[PlanView] = None
        self._step = 0

    @property
    def s3_uris(self) -> S3URISequence:
        """URI storage (S3KeyList or List[str])."""
        return self._uri_storage

    def set_plan(self, plan: PlanView) -> None:
        self.plan = plan
        self._step = 0

    def _window_bounds(self, batch_idx: int) -> Tuple[int, int]:
        assert self.plan is not None
        lo = max(0, int(batch_idx) - self.Bb)
        hi = min(len(self.plan), int(batch_idx) + self.Ba)
        return lo, hi

    def _window_uris(self, batch_idx: int) -> List[str]:
        assert self.plan is not None
        lo, hi = self._window_bounds(batch_idx)
        idxs = self.plan.range_flat(lo, hi)
        return [self._uri_storage[int(i)] for i in idxs]

    def prime(self) -> None:
        if self.plan is None:
            return
        keep = self._window_uris(0)
        self.cache.prime_prefetch(keep)

    def on_batch_yielded(self, batch_idx: int) -> None:
        if self.plan is None:
            return
        keep = self._window_uris(batch_idx)
        self.cache.prime_prefetch(keep)

        self._step += 1
        if self._step % self.prune_every == 0:
            self.cache.prune(keep_uris=keep, mode=self.prune_mode, backend=self.prune_backend)


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
    def __init__(self, plan_provider: PlanProvider, batch_sampler: PlanBatchSampler, prefetch: PrefetchCoordinator):
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
    prune_every: int = 5,
    prune_mode: str = "window",
    prune_backend: str = "meta",
    pin_memory: bool = True,
    persistent_workers: bool = True,
    worker_init_fn=None,
) -> Tuple[PrefetchingLoader, PrefetchController]:
    """
    Create:
      - DataLoader configured with a PlanBatchSampler
      - PrefetchingLoader wrapper
      - PrefetchController to advance epoch plans

    You call controller.set_epoch(epoch) once per epoch.
    """
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

    controller = PrefetchController(plan_provider, sampler, prefetch)

    dl = DataLoader(
        dataset,
        batch_sampler=sampler,  # type: ignore[arg-type]
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        worker_init_fn=worker_init_fn,
    )

    return PrefetchingLoader(dl, prefetch), controller

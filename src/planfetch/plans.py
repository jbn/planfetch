# plans.py
"""PlanView and plan providers for batch scheduling."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class PlanView:
    """
    A dense, batch-aware epoch plan.

    Stores a matrix of indices: [num_batches, batch_size] as np.int64.

    Benefits:
    - ~16MB for 2M indices (instead of hundreds of MB as Python ints).
    - Fast iteration, fast window queries for prefetching.
    """

    def __init__(self, batches: np.ndarray):
        if batches.ndim != 2 or batches.dtype != np.int64:
            raise ValueError(
                "PlanView expects a 2D np.int64 array [num_batches, batch_size]."
            )
        self._batches = batches

    def __len__(self) -> int:
        return int(self._batches.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self._batches.shape[1])

    def batch(self, i: int) -> np.ndarray:
        return self._batches[int(i)]  # view

    def range_flat(self, lo: int, hi: int) -> np.ndarray:
        lo = max(0, int(lo))
        hi = min(len(self), int(hi))
        if hi <= lo:
            return np.empty((0,), dtype=np.int64)
        return self._batches[lo:hi].reshape(-1)


class PlanProvider(ABC):
    @abstractmethod
    def make_epoch_plan(self, epoch: int) -> PlanView:
        raise NotImplementedError


class SequentialPlanProvider(PlanProvider):
    """
    Deterministic sequential plan, useful for val/test.
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        *,
        epoch_samples: Optional[int] = None,
        drop_last: bool = True,
    ):
        self.n = int(n)
        self.bs = int(batch_size)
        self.epoch_samples = int(epoch_samples) if epoch_samples is not None else self.n
        self.drop_last = bool(drop_last)

    def make_epoch_plan(self, epoch: int) -> PlanView:
        total = min(self.epoch_samples, self.n)
        if self.drop_last:
            nb = total // self.bs
            total2 = nb * self.bs
            idxs = np.arange(total2, dtype=np.int64)
        else:
            nb = math.ceil(total / self.bs)
            total2 = nb * self.bs
            idxs = np.arange(total2, dtype=np.int64)
            idxs[idxs >= total] = idxs[idxs >= total] % total
        batches = idxs.reshape(nb, self.bs)
        return PlanView(batches)


class ShuffleOncePerEpochPlanProvider(PlanProvider):
    """
    Default "shuffle once per epoch" plan.

    For large datasets, it avoids permuting the full [0..n) by using
    choice-without-replacement for epoch_samples.
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        *,
        seed: int = 123,
        epoch_samples: int = 2_000_000,
        drop_last: bool = True,
        full_permutation_threshold: int = 10_000_000,
    ):
        self.n = int(n)
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.epoch_samples = int(epoch_samples)
        self.drop_last = bool(drop_last)
        self.full_perm_thresh = int(full_permutation_threshold)

    def make_epoch_plan(self, epoch: int) -> PlanView:
        rng = np.random.default_rng(self.seed + int(epoch))

        total = min(self.epoch_samples, self.n)
        if self.drop_last:
            nb = total // self.bs
            total2 = nb * self.bs
        else:
            nb = math.ceil(total / self.bs)
            total2 = nb * self.bs

        if self.n <= self.full_perm_thresh:
            perm = rng.permutation(self.n).astype(np.int64, copy=False)
            seq = perm[:total2]
        else:
            seq = rng.choice(self.n, size=total2, replace=False).astype(
                np.int64, copy=False
            )

        if not self.drop_last and total2 > total:
            seq[total:] = seq[total:] % total

        batches = seq.reshape(nb, self.bs)
        return PlanView(batches)


def _choose_coprime_step(n: int, rng: np.random.Generator) -> int:
    if n <= 1:
        return 1
    step = int(rng.integers(1, n))
    for _ in range(256):
        if math.gcd(step, n) == 1:
            return step
        step += 1
        if step >= n:
            step = 1
    return 1


class _CyclicShuffler:
    """
    Memory-thrifty permutation-like stream over a fixed pool array.

    It never copies the pool. Each cycle uses:
      idx(pos) = (start + step*pos) mod n
    with gcd(step, n) == 1, yielding a full cycle.

    When the cycle finishes, it picks a new start/step.
    """

    def __init__(self, pool: np.ndarray, *, seed: int, epoch: int, class_id: int):
        self.pool = pool
        self.n = int(pool.shape[0])
        if self.n <= 0:
            raise ValueError("Empty class pool.")
        ss = np.random.SeedSequence([int(seed), int(epoch), int(class_id)])
        self.rng = np.random.default_rng(ss)
        self.start = int(self.rng.integers(0, self.n))
        self.step = _choose_coprime_step(self.n, self.rng)
        self.pos = 0

    def _next_cycle(self) -> None:
        self.start = int(self.rng.integers(0, self.n))
        self.step = _choose_coprime_step(self.n, self.rng)
        self.pos = 0

    def take(self, k: int) -> np.ndarray:
        k = int(k)
        if k <= 0:
            return np.empty((0,), dtype=np.int64)

        out = np.empty((k,), dtype=np.int64)
        written = 0
        while written < k:
            rem = self.n - self.pos
            take_now = min(k - written, rem)

            positions = self.pos + np.arange(take_now, dtype=np.int64)
            idxs = (self.start + (self.step * positions)) % self.n
            out[written : written + take_now] = self.pool[idxs]

            self.pos += take_now
            written += take_now

            if self.pos >= self.n and written < k:
                self._next_cycle()

        return out


def _allocate_fixed_counts(
    probs: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Allocate stable per-class counts for each batch, summing to batch_size.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    expected = probs * float(batch_size)
    base = np.floor(expected).astype(np.int64)
    rem = int(batch_size - int(base.sum()))
    if rem <= 0:
        return base

    frac = expected - base
    order = np.arange(len(probs))
    rng.shuffle(order)
    ranked = sorted(order.tolist(), key=lambda i: (frac[i],), reverse=True)
    base[ranked[:rem]] += 1
    return base


class StreamingStratifiedBatchPlanProvider(PlanProvider):
    """
    Memory-thrifty stratified batch plan provider.

    Key upgrade:
      - Avoids pool.copy() and per-epoch full shuffles of large class pools.
      - Uses a small per-class modular permutation stream instead.

    This keeps per-epoch allocations small even when class pools are enormous.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        *,
        seed: int = 123,
        epoch_samples: int = 2_000_000,
        drop_last: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        shuffle_within_batch: bool = True,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.epoch_samples = int(epoch_samples)
        self.drop_last = bool(drop_last)
        self.shuffle_within_batch = bool(shuffle_within_batch)

        classes = np.unique(self.labels)
        self.classes = np.asarray(classes, dtype=np.int64)

        # Pools built once
        self.class_to_pool: Dict[int, np.ndarray] = {}
        for c in self.classes:
            ci = int(c)
            self.class_to_pool[ci] = np.where(self.labels == c)[0].astype(np.int64)

        # Class probabilities (uniform over classes unless weighted)
        if class_weights is None:
            weights = {int(c): 1.0 for c in self.classes}
        else:
            weights = {int(k): float(v) for k, v in class_weights.items()}
            for c in self.classes:
                weights.setdefault(int(c), 0.0)

        w = np.array([weights[int(c)] for c in self.classes], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("class_weights sum to 0; provide positive weights.")
        self.class_probs = w / w.sum()

    def make_epoch_plan(self, epoch: int) -> PlanView:
        total = int(self.epoch_samples)  # may exceed dataset size; repeats are allowed
        if self.drop_last:
            nb = total // self.bs
            total2 = nb * self.bs
        else:
            nb = math.ceil(total / self.bs)
            total2 = nb * self.bs

        rng = np.random.default_rng(self.seed + int(epoch))

        counts = _allocate_fixed_counts(self.class_probs, self.bs, rng)

        active: List[Tuple[int, int]] = []
        for i, c in enumerate(self.classes):
            k = int(counts[i])
            if k > 0:
                active.append((int(c), k))
        if sum(k for _, k in active) != self.bs:
            raise RuntimeError("Internal error: counts do not sum to batch_size.")

        shufflers: Dict[int, _CyclicShuffler] = {}
        for c, _k in active:
            shufflers[c] = _CyclicShuffler(
                self.class_to_pool[c], seed=self.seed, epoch=int(epoch), class_id=int(c)
            )

        out = np.empty((nb, self.bs), dtype=np.int64)

        for b in range(nb):
            pos = 0
            for c, k in active:
                idxs = shufflers[c].take(k)
                out[b, pos : pos + k] = idxs
                pos += k

            if self.shuffle_within_batch:
                perm = rng.permutation(self.bs)
                out[b, :] = out[b, perm]

        if not self.drop_last and total2 > total:
            out_flat = out.reshape(-1)
            out_flat[total:] = out_flat[total:] % int(len(self.labels))
            out = out_flat.reshape(nb, self.bs)

        return PlanView(out)


class BalancedSubsamplingPlanProvider(PlanProvider):
    """
    Balanced subsampling plan provider for imbalanced datasets.

    Creates batches with exactly equal samples per class by:
    - Undersampling majority classes (random subset each epoch)
    - Oversampling minority classes if needed (via cyclic shuffling)

    Uses _CyclicShuffler for memory-efficient streaming without pool copies.
    """

    def __init__(
        self,
        labels: np.ndarray,
        samples_per_class: int,
        *,
        seed: int = 123,
        epoch_samples: Optional[int] = None,
        drop_last: bool = True,
        shuffle_within_batch: bool = True,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle_within_batch = bool(shuffle_within_batch)

        if self.samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")

        # Discover classes and build pools (once)
        classes = np.unique(self.labels)
        self.classes = np.asarray(classes, dtype=np.int64)

        self.class_to_pool: Dict[int, np.ndarray] = {}
        for c in self.classes:
            ci = int(c)
            self.class_to_pool[ci] = np.where(self.labels == c)[0].astype(np.int64)

        # Validate: all classes must have at least 1 sample
        for c, pool in self.class_to_pool.items():
            if len(pool) == 0:
                raise ValueError(f"Class {c} has no samples")

        # Compute batch_size
        self._batch_size = self.samples_per_class * len(self.classes)

        # Default epoch_samples: one pass through minority class
        self._min_class_size = min(len(pool) for pool in self.class_to_pool.values())
        if epoch_samples is None:
            self.epoch_samples = self._min_class_size * len(self.classes)
        else:
            self.epoch_samples = int(epoch_samples)

    @classmethod
    def from_batch_size(
        cls,
        labels: np.ndarray,
        batch_size: int,
        *,
        seed: int = 123,
        epoch_samples: Optional[int] = None,
        drop_last: bool = True,
        shuffle_within_batch: bool = True,
    ) -> "BalancedSubsamplingPlanProvider":
        """Create from batch_size, raising if not divisible by num_classes."""
        labels = np.asarray(labels, dtype=np.int64)
        num_classes = len(np.unique(labels))
        if batch_size % num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_classes ({num_classes}) "
                f"for balanced batching. Consider samples_per_class={batch_size // num_classes} "
                f"which gives batch_size={batch_size // num_classes * num_classes}."
            )
        samples_per_class = batch_size // num_classes
        return cls(
            labels,
            samples_per_class,
            seed=seed,
            epoch_samples=epoch_samples,
            drop_last=drop_last,
            shuffle_within_batch=shuffle_within_batch,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def min_class_size(self) -> int:
        return self._min_class_size

    def make_epoch_plan(self, epoch: int) -> PlanView:
        total = self.epoch_samples
        if self.drop_last:
            nb = total // self._batch_size
            total2 = nb * self._batch_size
        else:
            nb = math.ceil(total / self._batch_size)
            total2 = nb * self._batch_size

        if nb == 0:
            raise ValueError(
                f"epoch_samples ({self.epoch_samples}) is less than batch_size ({self._batch_size}). "
                f"Cannot create any batches."
            )

        rng = np.random.default_rng(self.seed + int(epoch))

        # Create one CyclicShuffler per class
        shufflers: Dict[int, _CyclicShuffler] = {}
        for c in self.classes:
            ci = int(c)
            shufflers[ci] = _CyclicShuffler(
                self.class_to_pool[ci],
                seed=self.seed,
                epoch=int(epoch),
                class_id=ci,
            )

        # Allocate output
        out = np.empty((nb, self._batch_size), dtype=np.int64)

        # Fill batches
        for b in range(nb):
            pos = 0
            for c in self.classes:
                ci = int(c)
                idxs = shufflers[ci].take(self.samples_per_class)
                out[b, pos : pos + self.samples_per_class] = idxs
                pos += self.samples_per_class

            if self.shuffle_within_batch:
                perm = rng.permutation(self._batch_size)
                out[b, :] = out[b, perm]

        # Handle drop_last=False padding (edge case)
        if not self.drop_last and total2 > total:
            out_flat = out.reshape(-1)
            out_flat[total:] = out_flat[total:] % int(len(self.labels))
            out = out_flat.reshape(nb, self._batch_size)

        return PlanView(out)

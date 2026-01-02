"""Tests for planfetch.plans module."""
import numpy as np
import pytest

from planfetch.plans import (
    PlanView,
    StreamingStratifiedBatchPlanProvider,
)


def test_planview_range_flat():
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    pv = PlanView(arr)
    assert len(pv) == 3
    assert pv.batch_size == 4
    assert np.all(pv.batch(1) == np.array([4, 5, 6, 7], dtype=np.int64))

    flat = pv.range_flat(1, 3)
    assert flat.shape == (8,)
    assert np.all(flat == np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64))


def test_streaming_stratified_counts_and_determinism():
    # Two balanced classes, batch_size divisible by num_classes -> exact per-batch stratification.
    labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    prov = StreamingStratifiedBatchPlanProvider(labels, batch_size=10, epoch_samples=100, seed=123)

    plan0_a = prov.make_epoch_plan(0)
    plan0_b = prov.make_epoch_plan(0)
    assert np.array_equal(
        plan0_a.range_flat(0, len(plan0_a)),
        plan0_b.range_flat(0, len(plan0_b)),
    )

    plan1 = prov.make_epoch_plan(1)
    assert not np.array_equal(plan0_a.range_flat(0, 3), plan1.range_flat(0, 3))

    # Check each batch has 5 of each class
    for bi in range(len(plan0_a)):
        idxs = plan0_a.batch(bi)
        batch_labels = labels[idxs]
        c0 = int(np.sum(batch_labels == 0))
        c1 = int(np.sum(batch_labels == 1))
        assert c0 == 5
        assert c1 == 5

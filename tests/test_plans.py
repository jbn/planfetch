"""Tests for planfetch.plans module."""
import numpy as np
import pytest

from planfetch.plans import (
    BalancedSubsamplingPlanProvider,
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


# ============================================================================
# BalancedSubsamplingPlanProvider Tests
# ============================================================================


def test_balanced_subsampling_basic():
    """Test balanced batches with equally-sized classes."""
    # 4 classes, 100 samples each
    labels = np.concatenate([np.full(100, i, dtype=np.int64) for i in range(4)])
    prov = BalancedSubsamplingPlanProvider(
        labels, samples_per_class=8, seed=42, epoch_samples=320
    )

    assert prov.batch_size == 32  # 8 * 4 classes
    assert prov.num_classes == 4

    plan = prov.make_epoch_plan(0)
    assert len(plan) == 10  # 320 / 32

    # Each batch should have exactly 8 samples per class
    for bi in range(len(plan)):
        batch = plan.batch(bi)
        batch_labels = labels[batch]
        for c in range(4):
            assert np.sum(batch_labels == c) == 8


def test_balanced_subsampling_imbalanced():
    """Test that majority classes are undersampled."""
    # Class 0: 1000 samples (majority)
    # Class 1: 100 samples (minority)
    labels = np.concatenate([
        np.zeros(1000, dtype=np.int64),
        np.ones(100, dtype=np.int64),
    ])

    prov = BalancedSubsamplingPlanProvider(
        labels, samples_per_class=10, seed=42
    )

    assert prov.batch_size == 20
    assert prov.min_class_size == 100
    # Default epoch_samples = 100 * 2 = 200
    assert prov.epoch_samples == 200

    plan = prov.make_epoch_plan(0)
    assert len(plan) == 10  # 200 / 20

    # Check balance in each batch
    for bi in range(len(plan)):
        batch = plan.batch(bi)
        batch_labels = labels[batch]
        assert np.sum(batch_labels == 0) == 10
        assert np.sum(batch_labels == 1) == 10


def test_balanced_subsampling_oversampling():
    """Test that minority classes are oversampled when needed."""
    # Class 0: 50 samples (minority)
    # Class 1: 500 samples (majority)
    labels = np.concatenate([
        np.zeros(50, dtype=np.int64),
        np.ones(500, dtype=np.int64),
    ])

    # Request more samples than minority class has
    prov = BalancedSubsamplingPlanProvider(
        labels, samples_per_class=10, seed=42, epoch_samples=400
    )

    plan = prov.make_epoch_plan(0)
    assert len(plan) == 20  # 400 / 20

    # Each batch still balanced
    for bi in range(len(plan)):
        batch = plan.batch(bi)
        batch_labels = labels[batch]
        assert np.sum(batch_labels == 0) == 10
        assert np.sum(batch_labels == 1) == 10

    # Minority class samples should repeat (oversampling)
    all_class0_idxs = []
    for bi in range(len(plan)):
        batch = plan.batch(bi)
        all_class0_idxs.extend(batch[labels[batch] == 0].tolist())

    # 200 samples needed from class 0, but only 50 exist -> repeats
    assert len(all_class0_idxs) == 200
    assert len(set(all_class0_idxs)) <= 50  # Some must repeat


def test_balanced_subsampling_determinism():
    """Test that same seed + epoch produces identical plans."""
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100, dtype=np.int64)

    prov = BalancedSubsamplingPlanProvider(labels, samples_per_class=5, seed=123)

    plan_a = prov.make_epoch_plan(0)
    plan_b = prov.make_epoch_plan(0)

    assert np.array_equal(
        plan_a.range_flat(0, len(plan_a)),
        plan_b.range_flat(0, len(plan_b)),
    )

    # Different epochs should differ
    plan_c = prov.make_epoch_plan(1)
    assert not np.array_equal(
        plan_a.range_flat(0, len(plan_a)),
        plan_c.range_flat(0, len(plan_c)),
    )


def test_balanced_subsampling_different_subsets_per_epoch():
    """Test that majority classes get different random subsets each epoch."""
    # Large majority class
    labels = np.concatenate([
        np.zeros(10000, dtype=np.int64),  # majority
        np.ones(100, dtype=np.int64),      # minority
    ])

    prov = BalancedSubsamplingPlanProvider(
        labels, samples_per_class=10, seed=42
    )

    plan0 = prov.make_epoch_plan(0)
    plan1 = prov.make_epoch_plan(1)

    # Get class 0 indices from each epoch
    def get_class0_idxs(plan):
        idxs = []
        for bi in range(len(plan)):
            batch = plan.batch(bi)
            idxs.extend(batch[labels[batch] == 0].tolist())
        return set(idxs)

    class0_epoch0 = get_class0_idxs(plan0)
    class0_epoch1 = get_class0_idxs(plan1)

    # Different subsets should be selected
    assert class0_epoch0 != class0_epoch1


def test_balanced_subsampling_from_batch_size():
    """Test the from_batch_size class method."""
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50, dtype=np.int64)

    # Valid: 32 / 4 classes = 8 per class
    prov = BalancedSubsamplingPlanProvider.from_batch_size(
        labels, batch_size=32, seed=42
    )
    assert prov.samples_per_class == 8
    assert prov.batch_size == 32

    # Invalid: 30 not divisible by 4
    with pytest.raises(ValueError, match="divisible"):
        BalancedSubsamplingPlanProvider.from_batch_size(
            labels, batch_size=30, seed=42
        )


def test_balanced_subsampling_single_sample_minority():
    """Test with a class that has only 1 sample."""
    labels = np.array([0] * 100 + [1], dtype=np.int64)  # class 1 has 1 sample

    prov = BalancedSubsamplingPlanProvider(
        labels, samples_per_class=5, seed=42, epoch_samples=100
    )

    plan = prov.make_epoch_plan(0)

    # Should still work, with class 1 heavily oversampled
    for bi in range(len(plan)):
        batch = plan.batch(bi)
        batch_labels = labels[batch]
        assert np.sum(batch_labels == 0) == 5
        assert np.sum(batch_labels == 1) == 5  # All 5 are index 100


def test_balanced_subsampling_many_classes():
    """Test with many classes (e.g., 100-way classification)."""
    num_classes = 100
    samples_per_class_data = 50
    labels = np.concatenate([
        np.full(samples_per_class_data, i, dtype=np.int64)
        for i in range(num_classes)
    ])

    prov = BalancedSubsamplingPlanProvider(
        labels, samples_per_class=2, seed=42
    )

    assert prov.batch_size == 200  # 2 * 100
    assert prov.num_classes == 100

    plan = prov.make_epoch_plan(0)

    # Verify balance
    batch = plan.batch(0)
    batch_labels = labels[batch]
    for c in range(num_classes):
        assert np.sum(batch_labels == c) == 2

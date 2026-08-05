"""Batching and label helpers.

The shuffling assertions matter more than they look: a `get_batches` that quietly
returns the same order every epoch turns minibatch SGD into deterministic gradient
descent over a fixed sequence, which trains, converges to something, and never fails a
test that only checks shapes.
"""

import numpy as np
import pytest

from pynn.utils.data import get_batches, one_hot


def dataset(n: int = 10, features: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Features whose first column is the example index, so order is observable."""
    X = np.zeros((n, features))
    X[:, 0] = np.arange(n)
    return X, np.arange(n).reshape(n, 1)


# --------------------------------------------------------------------------- #
# one_hot
# --------------------------------------------------------------------------- #


def test_one_hot_encodes_each_label():
    encoded = one_hot(np.array([0, 2, 1]), k=3)

    assert encoded.shape == (3, 3)
    assert np.array_equal(encoded, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_one_hot_accepts_a_column_vector():
    assert np.array_equal(one_hot(np.array([[1], [0]]), k=2), [[0, 1], [1, 0]])


def test_one_hot_defaults_to_ten_classes():
    assert one_hot(np.array([3])).shape == (1, 10)


# --------------------------------------------------------------------------- #
# get_batches
# --------------------------------------------------------------------------- #


def test_batches_partition_the_dataset():
    X, y = dataset(10)

    batches = list(get_batches(X, y, batch_size=4, shuffle=False))

    assert [len(Xb) for Xb, _ in batches] == [4, 4, 2]
    assert np.array_equal(np.concatenate([Xb for Xb, _ in batches]), X)
    assert np.array_equal(np.concatenate([yb for _, yb in batches]), y)


def test_batches_keep_examples_and_targets_aligned():
    X, y = dataset(9)

    for Xb, yb in get_batches(X, y, batch_size=4, rng=0):
        assert np.array_equal(Xb[:, 0].reshape(-1, 1), yb)


def test_an_exact_multiple_produces_no_trailing_batch():
    X, y = dataset(8)
    assert [len(Xb) for Xb, _ in get_batches(X, y, batch_size=4, shuffle=False)] == [
        4,
        4,
    ]


def test_a_dataset_smaller_than_one_batch_yields_a_single_batch():
    X, y = dataset(3)
    assert [len(Xb) for Xb, _ in get_batches(X, y, batch_size=8, shuffle=False)] == [3]


def test_an_empty_dataset_yields_nothing():
    assert list(get_batches(np.zeros((0, 3)), np.zeros((0, 1)), batch_size=4)) == []


def test_shuffle_permutes_without_dropping_or_duplicating():
    X, y = dataset(20)

    order = np.concatenate([Xb[:, 0] for Xb, _ in get_batches(X, y, 5, rng=0)])

    assert sorted(order.tolist()) == list(range(20))
    assert order.tolist() != list(range(20))


def test_shuffle_is_on_by_default():
    """The default has to be the stochastic one; opting in is what gets forgotten."""
    X, y = dataset(50)

    order = np.concatenate([Xb[:, 0] for Xb, _ in get_batches(X, y, 10)])

    assert order.tolist() != list(range(50))


def test_consecutive_epochs_use_different_orders():
    X, y = dataset(50)
    rng = np.random.default_rng(0)

    first = np.concatenate([Xb[:, 0] for Xb, _ in get_batches(X, y, 10, rng=rng)])
    second = np.concatenate([Xb[:, 0] for Xb, _ in get_batches(X, y, 10, rng=rng)])

    assert first.tolist() != second.tolist()


def test_a_seed_makes_the_order_reproducible():
    X, y = dataset(20)

    first = np.concatenate([Xb[:, 0] for Xb, _ in get_batches(X, y, 5, rng=7)])
    second = np.concatenate([Xb[:, 0] for Xb, _ in get_batches(X, y, 5, rng=7)])

    assert first.tolist() == second.tolist()


def test_shuffle_false_preserves_the_original_order():
    X, y = dataset(20)

    order = np.concatenate([Xb[:, 0] for Xb, _ in get_batches(X, y, 5, shuffle=False)])

    assert order.tolist() == list(range(20))


def test_batch_size_defaults_to_thirty_two():
    X, y = dataset(70)
    assert [len(Xb) for Xb, _ in get_batches(X, y, shuffle=False)] == [32, 32, 6]


def test_m_is_accepted_as_an_alias_for_batch_size():
    X, y = dataset(10)
    assert [len(Xb) for Xb, _ in get_batches(X, y, m=5, shuffle=False)] == [5, 5]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_non_positive_batch_size_is_rejected(batch_size):
    X, y = dataset(4)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        get_batches(X, y, batch_size=batch_size)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number of examples"):
        get_batches(np.zeros((4, 3)), np.zeros((5, 1)))


def test_argument_errors_are_raised_before_the_first_batch():
    """A bare generator would defer these until the caller starts iterating."""
    with pytest.raises(ValueError):
        get_batches(np.zeros((4, 3)), np.zeros((5, 1)), batch_size=2)

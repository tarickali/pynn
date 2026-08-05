"""The class-based Loss wrappers.

The numerics live in `pynn.functional.losses` and are gradient-checked by the verify
sweep. What is checked here is the wiring: that each class forwards its constructor
options to the function it wraps, and that the `logits` / `reduction` switches actually
select a different computation rather than being stored and ignored.
"""

import numpy as np
import pytest

import pynn.functional as F
from pynn.core import Tensor
from pynn.functional.losses import (
    binary_crossentropy,
    categorical_crossentropy,
    mean_absolute_error,
    mean_squared_error,
)
from pynn.nn import (
    BCELoss,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    CrossEntropyLoss,
    L1Loss,
    MeanAbsoluteError,
    MeanSquaredError,
    MSELoss,
)


@pytest.fixture
def regression(rng):
    return Tensor(rng.standard_normal((5, 3))), Tensor(rng.standard_normal((5, 3)))


@pytest.fixture
def binary(rng):
    true = Tensor(rng.integers(0, 2, (6, 1)).astype(np.float64))
    return true, Tensor(rng.standard_normal((6, 1)))


@pytest.fixture
def categorical(rng):
    true = Tensor(np.eye(4)[rng.integers(0, 4, 6)])
    return true, Tensor(rng.standard_normal((6, 4)))


def test_mean_squared_error_matches_the_function(regression):
    true, pred = regression
    assert np.allclose(
        MeanSquaredError()(true, pred).data, mean_squared_error(true, pred).data
    )


def test_mean_squared_error_forwards_the_reduction(regression):
    true, pred = regression

    total = MeanSquaredError(reduction="sum")(true, pred)

    assert np.allclose(total.data, mean_squared_error(true, pred, "sum").data)
    assert not np.allclose(total.data, MeanSquaredError()(true, pred).data)


def test_mean_absolute_error_matches_the_function(regression):
    true, pred = regression
    assert np.allclose(
        MeanAbsoluteError()(true, pred).data, mean_absolute_error(true, pred).data
    )


@pytest.mark.parametrize("logits", [True, False])
def test_binary_crossentropy_forwards_the_logits_flag(binary, logits):
    true, raw = binary
    pred = raw if logits else F.sigmoid(raw)

    assert np.allclose(
        BinaryCrossentropy(logits=logits)(true, pred).data,
        binary_crossentropy(true, pred, logits).data,
    )


@pytest.mark.parametrize("logits", [True, False])
def test_categorical_crossentropy_forwards_the_logits_flag(categorical, logits):
    true, raw = categorical
    pred = raw if logits else F.softmax(raw)

    assert np.allclose(
        CategoricalCrossentropy(logits=logits)(true, pred).data,
        categorical_crossentropy(true, pred, logits).data,
    )


def test_the_logits_flag_selects_a_different_computation(categorical):
    """Storing `logits` and then always taking the fused branch would pass otherwise."""
    true, logits = categorical

    fused = CategoricalCrossentropy(logits=True)(true, logits)
    probabilities = CategoricalCrossentropy(logits=False)(true, F.softmax(logits))

    assert np.allclose(fused.data, probabilities.data)
    assert not np.allclose(
        fused.data, CategoricalCrossentropy(logits=False)(true, logits).data
    )


def test_losses_default_to_the_logits_form(binary, categorical):
    true, pred = binary
    assert np.allclose(
        BinaryCrossentropy()(true, pred).data,
        BinaryCrossentropy(logits=True)(true, pred).data,
    )

    true, pred = categorical
    assert np.allclose(
        CategoricalCrossentropy()(true, pred).data,
        CategoricalCrossentropy(logits=True)(true, pred).data,
    )


def test_losses_are_differentiable_through_the_class(regression):
    true, pred = regression

    MeanSquaredError()(true, pred).backward()

    assert np.any(pred.grad != 0.0)


@pytest.mark.parametrize(
    "alias,target",
    [
        (BCELoss, BinaryCrossentropy),
        (CrossEntropyLoss, CategoricalCrossentropy),
        (MSELoss, MeanSquaredError),
        (L1Loss, MeanAbsoluteError),
    ],
)
def test_pytorch_style_aliases_point_at_the_same_class(alias, target):
    assert alias is target


# --------------------------------------------------------------------------- #
# Contract
#
# Shape agreement was enforced with a bare `assert`, which vanishes under `python -O`
# and would then let a broadcast turn a shape bug into a silently different loss.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "loss",
    [
        MeanSquaredError(),
        MeanAbsoluteError(),
        BinaryCrossentropy(),
        CategoricalCrossentropy(),
    ],
    ids=lambda loss: type(loss).__name__,
)
def test_mismatched_shapes_raise(loss):
    with pytest.raises(ValueError, match="same shape"):
        loss(Tensor(np.zeros((4, 3))), Tensor(np.zeros((4, 2))))


def test_an_unknown_reduction_raises():
    with pytest.raises(ValueError, match="reduction must be"):
        mean_squared_error(
            Tensor(np.zeros((2, 2))),
            Tensor(np.zeros((2, 2))),
            reduction="median",  # type: ignore[arg-type]
        )

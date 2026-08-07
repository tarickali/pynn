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
    BCEWithLogitsLoss,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    CrossEntropyLoss,
    HuberLoss,
    L1Loss,
    MeanAbsoluteError,
    MeanSquaredError,
    MSELoss,
    SmoothL1Loss,
    SparseCategoricalCrossentropy,
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
        (CrossEntropyLoss, CategoricalCrossentropy),
        (MSELoss, MeanSquaredError),
        (L1Loss, MeanAbsoluteError),
        (SmoothL1Loss, HuberLoss),
    ],
)
def test_pytorch_style_aliases_point_at_the_same_class(alias, target):
    assert alias is target


def test_the_bce_names_match_pytorch_not_each_other():
    """`BCELoss` used to be an alias for the *logits* form, which is backwards.

    PyTorch's `BCELoss` takes probabilities and `BCEWithLogitsLoss` takes logits, so
    aliasing `BCELoss` to a class defaulting to `logits=True` meant anyone reaching for
    the familiar name got the other function — and got it silently, since both accept
    the same shapes and return a plausible number.
    """
    assert BCELoss().logits is False
    assert BCEWithLogitsLoss().logits is True
    assert issubclass(BCELoss, BinaryCrossentropy)
    assert issubclass(BCEWithLogitsLoss, BinaryCrossentropy)


def test_the_two_bce_paths_agree_when_composed(binary):
    true, logits = binary

    fused = BCEWithLogitsLoss()(true, logits)
    composed = BCELoss()(true, F.sigmoid(logits))

    assert np.allclose(fused.data, composed.data)


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


# --------------------------------------------------------------------------- #
# Reduction
#
# Every loss takes the same three modes. They are only worth having if they mean the
# same thing in each: sum/mean must differ by the item count, and "none" must return
# the per-item losses those two reduce.
# --------------------------------------------------------------------------- #


def losses_under(reduction):
    return [
        MeanSquaredError(reduction=reduction),
        MeanAbsoluteError(reduction=reduction),
        HuberLoss(reduction=reduction),
        BCEWithLogitsLoss(reduction=reduction),
        CategoricalCrossentropy(reduction=reduction),
    ]


def inputs_for(loss, rng):
    if isinstance(loss, BinaryCrossentropy):
        return Tensor(rng.integers(0, 2, (6, 1)).astype(float)), Tensor(
            rng.standard_normal((6, 1))
        )
    if isinstance(loss, CategoricalCrossentropy):
        return Tensor(np.eye(4)[rng.integers(0, 4, 6)]), Tensor(
            rng.standard_normal((6, 4))
        )
    return Tensor(rng.standard_normal((6, 3))), Tensor(rng.standard_normal((6, 3)))


@pytest.mark.parametrize(
    "index", range(5), ids=["mse", "mae", "huber", "bce", "categorical"]
)
def test_sum_is_the_item_count_times_the_mean(index, rng):
    mean_loss = losses_under("mean")[index]
    sum_loss = losses_under("sum")[index]
    none_loss = losses_under("none")[index]
    true, pred = inputs_for(mean_loss, rng)

    per_item = none_loss(true, pred).data
    assert np.allclose(mean_loss(true, pred).data, per_item.mean())
    assert np.allclose(sum_loss(true, pred).data, per_item.sum())


@pytest.mark.parametrize(
    "index", range(5), ids=["mse", "mae", "huber", "bce", "categorical"]
)
def test_an_unreduced_loss_keeps_its_per_item_shape(index, rng):
    loss = losses_under("none")[index]
    true, pred = inputs_for(loss, rng)

    output = loss(true, pred)

    # The cross-entropies sum over classes first, so an "item" there is one example.
    expected = (6,) if isinstance(loss, CategoricalCrossentropy) else pred.shape
    assert output.shape == expected
    assert output.size > 1


def test_an_unreduced_loss_needs_an_explicit_backward_seed(rng):
    true, pred = inputs_for(MeanSquaredError(), rng)

    output = MeanSquaredError(reduction="none")(true, pred)

    with pytest.raises(ValueError, match="more than one element"):
        output.backward()

    output.backward(gradient=np.ones(output.shape))
    assert np.any(pred.grad != 0.0)


@pytest.mark.parametrize(
    "loss",
    [MeanSquaredError, MeanAbsoluteError, HuberLoss, BinaryCrossentropy],
    ids=lambda c: c.__name__,
)
def test_every_loss_rejects_an_unknown_reduction(loss, rng):
    true, pred = inputs_for(loss(), rng)
    with pytest.raises(ValueError, match="'mean', 'sum', or 'none'"):
        loss(reduction="median")(true, pred)


# --------------------------------------------------------------------------- #
# Huber
# --------------------------------------------------------------------------- #


def test_huber_is_quadratic_near_zero_and_linear_beyond_delta():
    true = Tensor(np.zeros((1, 4)))
    pred = Tensor(np.array([[0.5, 1.0, 2.0, 10.0]]))

    values = HuberLoss(delta=1.0, reduction="none")(true, pred).data

    # 0.5 r^2 inside, delta(|r| - delta/2) outside.
    assert np.allclose(values, [[0.125, 0.5, 1.5, 9.5]])


def test_huber_is_continuous_in_value_and_slope_at_the_join():
    """Both pieces meet at |r| = delta, which is the point of choosing delta/2."""
    delta = 2.0
    true = Tensor(np.zeros((1, 3)))
    step = 1e-7

    at_join = HuberLoss(delta=delta, reduction="none")(
        true, Tensor([[delta, 0.0, 0.0]])
    ).data[0, 0]
    assert at_join == pytest.approx(0.5 * delta**2)

    # Value: the gap either side of the join shrinks with the step, so it is the
    # function that is continuous rather than the tolerance that is generous.
    pred = Tensor([[delta - step, delta + step, 0.0]])
    values = HuberLoss(delta=delta, reduction="none")(true, pred).data
    assert abs(values[0, 0] - values[0, 1]) < 10 * step * delta

    # Slope: r on the quadratic side and delta*sign(r) on the linear side are the same
    # number at r = delta, which is what makes the derivative continuous too.
    HuberLoss(delta=delta, reduction="sum")(true, pred).backward()
    assert pred.grad[0, 0] == pytest.approx(delta, abs=1e-6)
    assert pred.grad[0, 1] == pytest.approx(delta)


def test_huber_resists_an_outlier_that_dominates_squared_error():
    """The reason to reach for it: one bad label should not own the gradient."""
    true = Tensor(np.zeros((1, 5)))
    pred = Tensor(np.array([[0.1, 0.1, 0.1, 0.1, 50.0]]))

    HuberLoss(reduction="sum")(true, pred).backward()
    huber_grad = pred.grad.copy()

    pred.zero_grad()
    MeanSquaredError(reduction="sum")(true, pred).backward()
    squared_grad = pred.grad

    assert abs(huber_grad[0, -1]) == pytest.approx(1.0)
    assert abs(squared_grad[0, -1]) > 50


def test_huber_rejects_a_non_positive_delta(regression):
    true, pred = regression
    with pytest.raises(ValueError, match="delta must be positive"):
        HuberLoss(delta=0.0)(true, pred)


# --------------------------------------------------------------------------- #
# Sparse categorical cross-entropy
# --------------------------------------------------------------------------- #


def test_sparse_matches_the_one_hot_form(rng):
    labels = rng.integers(0, 4, 6)
    logits = rng.standard_normal((6, 4))

    sparse_pred = Tensor(logits.copy())
    sparse = SparseCategoricalCrossentropy()(Tensor(labels.astype(float)), sparse_pred)
    sparse.backward()

    dense_pred = Tensor(logits.copy())
    dense = CategoricalCrossentropy()(Tensor(np.eye(4)[labels]), dense_pred)
    dense.backward()

    assert np.allclose(sparse.data, dense.data)
    assert np.allclose(sparse_pred.grad, dense_pred.grad)


def test_sparse_accepts_a_plain_array_of_labels(rng):
    labels = rng.integers(0, 3, 5)
    logits = Tensor(rng.standard_normal((5, 3)))

    assert np.isfinite(SparseCategoricalCrossentropy()(labels, logits).data).all()


@pytest.mark.parametrize("logits", [True, False])
def test_sparse_matches_the_one_hot_form_for_probabilities(rng, logits):
    labels = rng.integers(0, 4, 6)
    raw = rng.standard_normal((6, 4))
    pred_data = raw if logits else F.softmax(Tensor(raw)).data

    sparse = SparseCategoricalCrossentropy(logits=logits)(
        Tensor(labels.astype(float)), Tensor(pred_data)
    )
    dense = CategoricalCrossentropy(logits=logits)(
        Tensor(np.eye(4)[labels]), Tensor(pred_data)
    )

    assert np.allclose(sparse.data, dense.data)


def test_sparse_rejects_a_label_count_mismatch(rng):
    with pytest.raises(ValueError, match="one label per example"):
        SparseCategoricalCrossentropy()(
            np.array([0, 1]), Tensor(rng.standard_normal((5, 3)))
        )


def test_sparse_rejects_an_out_of_range_label(rng):
    with pytest.raises(ValueError, match=r"labels must be in \[0, 3\)"):
        SparseCategoricalCrossentropy()(
            np.array([0, 1, 7]), Tensor(rng.standard_normal((3, 3)))
        )


def test_sparse_rejects_a_non_matrix_prediction(rng):
    with pytest.raises(ValueError, match=r"\(batch, classes\)"):
        SparseCategoricalCrossentropy()(
            np.array([0, 1]), Tensor(rng.standard_normal((2, 3, 4)))
        )

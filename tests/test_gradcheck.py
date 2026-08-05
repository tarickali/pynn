"""Numerical gradient checks for every differentiable operation.

Each test compares reverse-mode gradients against central differences. The functions
under test are wrapped so that they reduce to a scalar, and they are contracted against
a second random tensor rather than merely summed, so that a wrong Jacobian cannot hide
behind a gradient of all ones.
"""

import numpy as np
import pytest

import pynn.core.math as pmath
import pynn.functional as F
from pynn.core import Tensor
from pynn.functional.losses import (
    binary_crossentropy,
    categorical_crossentropy,
    mean_absolute_error,
    mean_squared_error,
)
from pynn.functional.modules import conv2d, flatten, linear
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.verify import check_gradients


def assert_gradients(fn, inputs, **kwargs):
    result = check_gradients(fn, inputs, **kwargs)
    assert result.passed, f"\n{result}"


# --------------------------------------------------------------------------- #
# Tensor operators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda a, b: a + b, id="add"),
        pytest.param(lambda a, b: a - b, id="sub"),
        pytest.param(lambda a, b: a * b, id="mul"),
        pytest.param(lambda a, b: a + b * a - b, id="mixed"),
    ],
)
@pytest.mark.parametrize(
    "shapes",
    [
        ((3, 4), (3, 4)),
        ((3, 4), (4,)),
        ((3, 4), (1, 4)),
        ((3, 1), (1, 4)),
        ((2, 3, 4), (3, 4)),
        ((2, 3, 4), (1, 1, 4)),
        ((5,), ()),
    ],
    ids=str,
)
def test_broadcasting_binary_ops(rng, op, shapes):
    """Broadcast operands must have their gradients summed over the broadcast axes."""
    inputs = [Tensor(rng.standard_normal(shape)) for shape in shapes]
    assert_gradients(lambda ts: pmath.sum(op(ts[0], ts[1])), inputs)


def test_division(rng):
    inputs = [Tensor(rng.standard_normal((3, 4))), Tensor(rng.uniform(1.0, 2.0, (4,)))]
    assert_gradients(lambda ts: pmath.sum(ts[0] / ts[1]), inputs)


def test_reverse_division(rng):
    inputs = [Tensor(rng.uniform(1.0, 2.0, (3, 4)))]
    assert_gradients(lambda ts: pmath.sum(2.0 / ts[0]), inputs)


@pytest.mark.parametrize("exponent", [2, 3, -1, -2])
def test_power(rng, exponent):
    inputs = [Tensor(rng.uniform(1.0, 2.0, (3, 4)))]
    assert_gradients(lambda ts: pmath.sum(ts[0] ** exponent), inputs)


def test_negation(rng):
    inputs = [Tensor(rng.standard_normal((3, 4)))]
    assert_gradients(lambda ts: pmath.sum(-ts[0]), inputs)


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((3, 4), (4, 2)),
        ((1, 4), (4, 1)),
        ((4,), (4, 2)),
        ((3, 4), (4,)),
        ((4,), (4,)),
        ((2, 3, 4), (4, 5)),
        ((2, 3, 4), (2, 4, 5)),
        ((2, 3, 4), (4,)),
    ],
    ids=str,
)
def test_matmul(rng, left_shape, right_shape):
    """Covers matmul's vector promotion and batch broadcasting rules."""
    inputs = [
        Tensor(rng.standard_normal(left_shape)),
        Tensor(rng.standard_normal(right_shape)),
    ]
    assert_gradients(lambda ts: pmath.sum(ts[0] @ ts[1]), inputs)


def test_transpose(rng):
    inputs = [Tensor(rng.standard_normal((3, 4))), Tensor(rng.standard_normal((3, 2)))]
    assert_gradients(lambda ts: pmath.sum(ts[0].T @ ts[1]), inputs)


def test_transpose_with_axes(rng):
    inputs = [
        Tensor(rng.standard_normal((2, 3, 4))),
        Tensor(rng.standard_normal((4, 2, 3))),
    ]
    assert_gradients(lambda ts: pmath.sum(ts[0].transpose((2, 0, 1)) * ts[1]), inputs)


# --------------------------------------------------------------------------- #
# Math functions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), -1])
def test_sum_over_axes(rng, axis):
    inputs = [Tensor(rng.standard_normal((3, 4)))]
    assert_gradients(lambda ts: pmath.sum(pmath.sum(ts[0], axis=axis)), inputs)


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), -1])
def test_mean_over_axes(rng, axis):
    inputs = [Tensor(rng.standard_normal((3, 4)))]
    assert_gradients(lambda ts: pmath.sum(pmath.mean(ts[0], axis=axis)), inputs)


def test_exp(rng):
    inputs = [Tensor(rng.standard_normal((3, 4))), Tensor(rng.standard_normal((3, 4)))]
    assert_gradients(lambda ts: pmath.sum(pmath.exp(ts[0]) * ts[1]), inputs)


def test_log(rng):
    inputs = [
        Tensor(rng.uniform(0.5, 3.0, (3, 4))),
        Tensor(rng.standard_normal((3, 4))),
    ]
    assert_gradients(lambda ts: pmath.sum(pmath.log(ts[0]) * ts[1]), inputs)


def test_abs(rng):
    # Values kept away from zero, where |x| is not differentiable.
    data = rng.uniform(0.5, 2.0, (3, 4)) * rng.choice([-1.0, 1.0], (3, 4))
    inputs = [Tensor(data), Tensor(rng.standard_normal((3, 4)))]
    assert_gradients(lambda ts: pmath.sum(pmath.abs(ts[0]) * ts[1]), inputs)


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #

ACTIVATIONS = [
    pytest.param(F.identity, id="identity"),
    pytest.param(lambda x: F.affine(x, 2.0, 3.0), id="affine"),
    pytest.param(F.relu, id="relu"),
    pytest.param(lambda x: F.relu(x, 0.2), id="leaky_relu"),
    pytest.param(F.sigmoid, id="sigmoid"),
    pytest.param(F.tanh, id="tanh"),
    pytest.param(F.elu, id="elu_default"),
    pytest.param(lambda x: F.elu(x, 0.5), id="elu_alpha"),
    pytest.param(F.selu, id="selu"),
    pytest.param(F.softplus, id="softplus"),
    pytest.param(F.softmax, id="softmax"),
    pytest.param(lambda x: F.softmax(x, axis=0), id="softmax_axis0"),
]


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_activation_gradients(rng, activation):
    # Offset away from zero so that relu's kink is never straddled by the +/- eps probe.
    data = rng.standard_normal((4, 5))
    data[np.abs(data) < 1e-3] = 0.5
    inputs = [Tensor(data), Tensor(rng.standard_normal((4, 5)))]
    assert_gradients(lambda ts: pmath.sum(activation(ts[0]) * ts[1]), inputs)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_activation_gradients_at_scale(rng, activation):
    """Large-magnitude inputs: catches overflow in sigmoid, softplus, and softmax."""
    data = rng.uniform(20.0, 60.0, (4, 5)) * rng.choice([-1.0, 1.0], (4, 5))
    inputs = [Tensor(data), Tensor(rng.standard_normal((4, 5)))]
    with np.errstate(over="raise", invalid="raise"):
        assert_gradients(
            lambda ts: pmath.sum(activation(ts[0]) * ts[1]), inputs, eps=1e-4
        )


def test_softplus_does_not_overflow():
    assert np.isfinite(F.softplus(Tensor(np.array([800.0]))).data).all()
    assert F.softplus(Tensor(np.array([800.0]))).data[0] == pytest.approx(800.0)


def test_sigmoid_does_not_overflow():
    with np.errstate(over="raise"):
        assert F.sigmoid(Tensor(np.array([-800.0, 800.0]))).data.tolist() == [0.0, 1.0]


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_mean_squared_error(rng, reduction):
    inputs = [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((4, 3)))]
    assert_gradients(
        lambda ts: mean_squared_error(ts[0], ts[1], reduction=reduction), inputs
    )


def test_mean_absolute_error(rng):
    true = rng.standard_normal((4, 3))
    pred = true + rng.uniform(0.5, 1.0, (4, 3)) * rng.choice([-1.0, 1.0], (4, 3))
    inputs = [Tensor(true), Tensor(pred)]
    assert_gradients(lambda ts: mean_absolute_error(ts[0], ts[1]), inputs)


def test_binary_crossentropy_from_logits(rng):
    targets = Tensor(rng.integers(0, 2, (6, 1)).astype(np.float64))
    inputs = [Tensor(rng.standard_normal((6, 1)))]
    assert_gradients(
        lambda ts: binary_crossentropy(targets, ts[0], logits=True), inputs
    )


def test_binary_crossentropy_from_probabilities(rng):
    targets = Tensor(rng.integers(0, 2, (6, 1)).astype(np.float64))
    inputs = [Tensor(rng.uniform(0.15, 0.85, (6, 1)))]
    assert_gradients(
        lambda ts: binary_crossentropy(targets, ts[0], logits=False), inputs
    )


def test_binary_crossentropy_through_sigmoid(rng):
    """Composing sigmoid with the probability form must match the fused logit form."""
    targets = Tensor(rng.integers(0, 2, (6, 1)).astype(np.float64))
    inputs = [Tensor(rng.standard_normal((6, 1)))]
    assert_gradients(
        lambda ts: binary_crossentropy(targets, F.sigmoid(ts[0]), logits=False), inputs
    )


def test_categorical_crossentropy_from_logits(rng):
    targets = Tensor(np.eye(4)[rng.integers(0, 4, 6)])
    inputs = [Tensor(rng.standard_normal((6, 4)))]
    assert_gradients(
        lambda ts: categorical_crossentropy(targets, ts[0], logits=True), inputs
    )


def test_categorical_crossentropy_from_probabilities(rng):
    targets = Tensor(np.eye(4)[rng.integers(0, 4, 6)])
    inputs = [Tensor(rng.uniform(0.1, 0.9, (6, 4)))]
    assert_gradients(
        lambda ts: categorical_crossentropy(targets, ts[0], logits=False), inputs
    )


def test_categorical_crossentropy_through_softmax(rng):
    """softmax + probability form must agree with the fused logits form."""
    targets = Tensor(np.eye(4)[rng.integers(0, 4, 6)])
    logits = rng.standard_normal((6, 4))

    fused = Tensor(logits.copy())
    categorical_crossentropy(targets, fused, logits=True).backward()

    composed = Tensor(logits.copy())
    categorical_crossentropy(targets, F.softmax(composed), logits=False).backward()

    assert np.allclose(fused.grad, composed.grad)


def test_categorical_crossentropy_matches_closed_form(rng):
    """dL/dlogits for softmax cross-entropy is exactly (softmax(z) - y) / batch."""
    targets = np.eye(4)[rng.integers(0, 4, 6)]
    logits = rng.standard_normal((6, 4))

    tensor = Tensor(logits.copy())
    categorical_crossentropy(Tensor(targets), tensor, logits=True).backward()

    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=-1, keepdims=True)
    expected = (probabilities - targets) / logits.shape[0]

    assert np.allclose(tensor.grad, expected)


# --------------------------------------------------------------------------- #
# Module functions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("include_bias", [True, False])
def test_linear(rng, include_bias):
    inputs = [
        Tensor(rng.standard_normal((5, 3))),
        Tensor(rng.standard_normal((3, 2))),
        Tensor(rng.standard_normal((2,))),
    ]
    bias_index = 2 if include_bias else None

    def fn(ts):
        bias = ts[bias_index] if bias_index is not None else None
        return pmath.sum(linear(ts[0], ts[1], bias))

    assert_gradients(fn, inputs if include_bias else inputs[:2])


def test_flatten(rng):
    inputs = [
        Tensor(rng.standard_normal((2, 3, 4))),
        Tensor(rng.standard_normal((2, 12))),
    ]
    assert_gradients(lambda ts: pmath.sum(flatten(ts[0]) * ts[1]), inputs)


@pytest.mark.parametrize(
    "stride,padding",
    [((1, 1), (0, 0)), ((2, 2), (0, 0)), ((1, 1), (1, 1)), ((2, 2), (1, 2))],
    ids=str,
)
def test_conv2d(rng, stride, padding):
    x = Tensor(rng.standard_normal((2, 2, 5, 5)))
    kernel = Tensor(rng.standard_normal((3, 2, 3, 3)))
    probe = Tensor(rng.standard_normal(conv2d(x, kernel, None, stride, padding).shape))
    assert_gradients(
        lambda ts: pmath.sum(conv2d(ts[0], ts[1], None, stride, padding) * probe),
        [x, kernel],
    )


def test_conv2d_with_bias(rng):
    x = Tensor(rng.standard_normal((2, 2, 4, 4)))
    kernel = Tensor(rng.standard_normal((3, 2, 3, 3)))
    bias = Tensor(rng.standard_normal((3, 2, 2)))
    assert_gradients(
        lambda ts: pmath.sum(conv2d(ts[0], ts[1], ts[2])), [x, kernel, bias]
    )


# --------------------------------------------------------------------------- #
# Graph topology
#
# A tensor consumed by more than one operation must have the contributions from
# every consumer summed. These are the cases a plain feedforward chain never
# exercises, and they are where accumulation bugs surface.
# --------------------------------------------------------------------------- #


def test_input_used_twice(rng):
    inputs = [Tensor(rng.standard_normal((3, 4)))]
    assert_gradients(lambda ts: pmath.sum(ts[0]) + pmath.sum(ts[0] * 2.0), inputs)


def test_branch_that_rejoins(rng):
    data = rng.standard_normal((3, 4))
    data[np.abs(data) < 1e-3] = 0.5
    inputs = [Tensor(data)]
    assert_gradients(lambda ts: pmath.sum(F.relu(ts[0])) + pmath.sum(ts[0]), inputs)


def test_diamond(rng):
    inputs = [Tensor(rng.uniform(-0.5, 0.5, (3, 4)))]
    assert_gradients(
        lambda ts: pmath.sum(pmath.exp(ts[0]) * pmath.log(ts[0] + 3.0)), inputs
    )


def test_self_product(rng):
    inputs = [Tensor(rng.standard_normal((3, 4)))]
    assert_gradients(lambda ts: pmath.sum(ts[0] * ts[0]), inputs)


def test_tied_weights(rng):
    """The same weight matrix applied at two depths, as in a weight-tied network."""
    inputs = [Tensor(rng.standard_normal((3, 4))), Tensor(rng.standard_normal((4, 4)))]
    assert_gradients(
        lambda ts: pmath.sum(F.tanh(F.tanh(ts[0] @ ts[1]) @ ts[1])), inputs
    )


def test_two_heads_sharing_a_trunk(rng):
    inputs = [
        Tensor(rng.standard_normal((4, 3))),
        Tensor(rng.standard_normal((3, 3))),
        Tensor(rng.standard_normal((3, 2))),
        Tensor(rng.standard_normal((3, 2))),
    ]

    def fn(ts):
        trunk = F.tanh(ts[0] @ ts[1])
        return pmath.sum(trunk @ ts[2]) + pmath.sum(F.sigmoid(trunk @ ts[3]))

    assert_gradients(fn, inputs)


def test_residual_connection(rng):
    inputs = [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((3, 3)))]
    assert_gradients(lambda ts: pmath.sum(ts[0] + F.tanh(ts[0] @ ts[1])), inputs)


def test_auxiliary_loss(rng):
    """Two losses on the same graph, as in deep supervision."""
    targets = Tensor(rng.standard_normal((4, 2)))
    inputs = [
        Tensor(rng.standard_normal((4, 3))),
        Tensor(rng.standard_normal((3, 2))),
    ]

    def fn(ts):
        hidden = F.tanh(ts[0] @ ts[1])
        main = mean_squared_error(targets, hidden)
        auxiliary = mean_absolute_error(targets, hidden)
        return main + 0.5 * auxiliary

    assert_gradients(fn, inputs)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_mlp_parameter_gradients(activation):
    """Every parameter of a trained-shape MLP against numerical gradients."""
    rng = np.random.default_rng(7)
    X = Tensor(rng.standard_normal((6, 3)))
    y = Tensor(rng.standard_normal((6, 2)))

    model = Sequential([Linear(3, 5, activation=activation), Linear(5, 2)])
    model(X)  # lazily builds the parameters

    parameters = [p for module in model.modules for p in module.parameters.values()]
    loss_fn = MeanSquaredError()

    def fn(_):
        return loss_fn(y, model(X))

    result = check_gradients(fn, parameters)
    assert result.passed, f"\n{result}"


def test_deep_chain_does_not_overflow_the_stack():
    """A long chain must not hit Python's recursion limit during backward()."""
    tensor = Tensor(np.array([1.0]))
    accumulator = tensor
    for _ in range(5000):
        accumulator = accumulator + 1.0
    accumulator.backward()
    assert tensor.grad.tolist() == [1.0]


def test_backward_accumulates_across_calls():
    """Two backward passes without zero_grad must double the gradient, not scale it."""
    rng = np.random.default_rng(11)
    X = Tensor(rng.standard_normal((4, 3)))
    W = Tensor(rng.standard_normal((3, 2)))
    b = Tensor(rng.standard_normal((2,)))

    pmath.sum(X @ W + b).backward()
    first = b.grad.copy()

    pmath.sum(X @ W + b).backward()

    assert np.allclose(b.grad, 2 * first)


def test_backward_requires_scalar_or_explicit_seed():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="more than one element"):
        (x * 2.0).backward()

    y = x * 2.0
    y.backward(gradient=np.ones((2, 2)))
    assert np.allclose(x.grad, 2.0)


def test_backward_rejects_mismatched_seed():
    x = Tensor(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="does not match"):
        (x * 2.0).backward(gradient=np.ones((3,)))

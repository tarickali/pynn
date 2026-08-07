"""The class-based Activation wrappers.

Each class exists only to bind hyperparameters to the matching function in
`pynn.functional.activations`, which the verify sweep gradient-checks. What is asserted
here is that the binding is real: that `ELU(alpha=0.5)` reaches `elu` with 0.5 rather
than silently taking the default.
"""

import numpy as np
import pytest

import pynn.functional as F
from pynn.core import Module, Tensor
from pynn.nn.activations import (
    ELU,
    GELU,
    SELU,
    Affine,
    Identity,
    LogSoftmax,
    PReLU,
    ReLU,
    Sigmoid,
    SiLU,
    Softmax,
    SoftPlus,
    Tanh,
)


@pytest.fixture
def x(rng) -> Tensor:
    return Tensor(rng.standard_normal((4, 5)))


@pytest.mark.parametrize(
    "activation,function",
    [
        (Identity(), F.identity),
        (Affine(), F.affine),
        (ReLU(), F.relu),
        (Sigmoid(), F.sigmoid),
        (Tanh(), F.tanh),
        (ELU(), F.elu),
        (SELU(), F.selu),
        (SoftPlus(), F.softplus),
        (Softmax(), F.softmax),
        (GELU(), F.gelu),
        (SiLU(), F.silu),
        (LogSoftmax(), F.log_softmax),
    ],
    ids=lambda value: getattr(value, "__name__", type(value).__name__),
)
def test_each_activation_matches_its_function(activation, function, x):
    assert np.allclose(activation(x).data, function(x).data)


@pytest.mark.parametrize(
    "activation,expected",
    [
        (Affine(slope=2.0, intercept=1.0), lambda a: 2.0 * a + 1.0),
        (ReLU(alpha=0.1), lambda a: np.where(a > 0, a, 0.1 * a)),
        (ELU(alpha=0.5), lambda a: np.where(a >= 0, a, 0.5 * np.expm1(a))),
    ],
)
def test_hyperparameters_reach_the_function(activation, expected, x):
    assert np.allclose(activation(x).data, expected(x.data))


def test_softmax_axis_reaches_the_function(x):
    over_rows = Softmax(axis=0)(x).data

    assert np.allclose(over_rows.sum(axis=0), 1.0)
    assert not np.allclose(over_rows, Softmax()(x).data)


def test_activations_stay_on_the_tape(x):
    """A wrapper that unwrapped `.data` would detach the graph without erroring."""
    import pynn.core.math as pmath

    pmath.sum(Tanh()(x)).backward()

    assert np.allclose(x.grad, 1.0 - np.tanh(x.data) ** 2)


# --------------------------------------------------------------------------- #
# GELU, SiLU, LogSoftmax
#
# The verify sweep gradient-checks all of these. What is asserted here is that the
# forward pass is the function it claims to be — the properties that pin down *which*
# activation this is, rather than that its derivative is self-consistent.
# --------------------------------------------------------------------------- #


def test_gelu_matches_its_definition(x):
    """x * Phi(x), with Phi the standard normal CDF."""
    from math import erf as scalar_erf

    expected = np.array(
        [v * 0.5 * (1.0 + scalar_erf(v / np.sqrt(2.0))) for v in x.data.ravel()]
    ).reshape(x.shape)

    assert np.allclose(F.gelu(x, approximate="none").data, expected)


def test_the_gelu_approximation_tracks_the_exact_form(x):
    """The tanh form is an approximation, so close but deliberately not equal."""
    approximate = F.gelu(x, approximate="tanh").data
    exact = F.gelu(x, approximate="none").data

    assert np.allclose(approximate, exact, atol=1e-3)
    assert not np.array_equal(approximate, exact)


@pytest.mark.parametrize("approximate", ["tanh", "none"])
def test_gelu_is_bounded_by_zero_and_the_identity(rng, approximate):
    values = Tensor(rng.standard_normal((200,)) * 3.0)

    output = F.gelu(values, approximate=approximate).data

    # x * Phi(x) with Phi in [0, 1]: never past x, never past 0 by much.
    assert np.all(output <= np.maximum(values.data, 0.0) + 1e-9)
    assert output.min() > -0.2


def test_gelu_rejects_an_unknown_approximation(x):
    with pytest.raises(ValueError, match="'tanh' or 'none'"):
        F.gelu(x, approximate="erf")


def test_silu_matches_its_definition(x):
    assert np.allclose(F.silu(x).data, x.data / (1.0 + np.exp(-x.data)))


def test_silu_is_non_monotonic():
    """The property that distinguishes it from ReLU: it dips below zero."""
    values = Tensor(np.linspace(-4.0, 0.0, 200))
    output = F.silu(values).data

    assert output.min() < -0.2
    assert np.argmin(output) not in (0, len(output) - 1), "the dip is interior"


def test_log_softmax_is_the_log_of_softmax(x):
    assert np.allclose(F.log_softmax(x).data, np.log(F.softmax(x).data))


def test_log_softmax_exponentiates_to_a_distribution(rng):
    values = Tensor(rng.standard_normal((5, 7)))

    probabilities = np.exp(F.log_softmax(values).data)

    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_log_softmax_stays_finite_where_the_naive_form_does_not():
    """The whole reason it exists: log(softmax(z)) is -inf for a saturated class."""
    values = Tensor(np.array([[1000.0, 0.0, -1000.0]]))

    stable = F.log_softmax(values).data
    with np.errstate(divide="ignore"):  # the naive form is meant to blow up here
        naive = np.log(F.softmax(values).data)

    assert np.all(np.isfinite(stable))
    assert not np.all(np.isfinite(naive))
    assert np.allclose(stable, [[0.0, -1000.0, -2000.0]])


def test_log_softmax_axis_is_honored(rng):
    values = Tensor(rng.standard_normal((4, 6)))

    assert np.allclose(np.exp(F.log_softmax(values, axis=0).data).sum(axis=0), 1.0)


# --------------------------------------------------------------------------- #
# PReLU
#
# The only activation here that owns a parameter, which is why it is a Module: a
# parameter outside the module tree is one no optimizer would ever step.
# --------------------------------------------------------------------------- #


def test_prelu_is_relu_with_a_learned_negative_slope():
    values = Tensor(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
    layer = PReLU(init=0.25)

    assert np.allclose(layer(values).data, [[-0.5, -0.25, 0.0, 1.0, 2.0]])


def test_prelu_is_a_module_with_a_parameter():
    layer = PReLU()

    assert isinstance(layer, Module)
    assert sorted(layer.named_parameters()) == ["alpha"]
    assert layer.num_parameters() == 1


def test_prelu_does_not_shadow_the_module_api():
    """`num_parameters` is a Module *method*; storing an int there breaks every call."""
    layer = PReLU(num_parameters=3)

    assert layer.num_parameters() == 3
    assert layer.num_slopes == 3
    assert layer.hyperparameters["num_parameters"] == 3


def test_prelu_supports_a_slope_per_channel():
    layer = PReLU(num_parameters=3)
    layer.parameters["alpha"].data = np.array([0.1, 0.5, 1.0])
    values = Tensor(np.full((2, 3), -2.0))

    assert np.allclose(layer(values).data, [[-0.2, -1.0, -2.0]] * 2)


def test_prelu_broadcasts_a_per_channel_slope_along_axis_one():
    """For images the slope must land on the channel axis, not the last one."""
    layer = PReLU(num_parameters=2)
    layer.parameters["alpha"].data = np.array([0.5, 1.0])
    values = Tensor(np.full((1, 2, 3, 3), -4.0))

    output = layer(values).data
    assert np.allclose(output[0, 0], -2.0)
    assert np.allclose(output[0, 1], -4.0)


def test_prelu_rejects_a_mismatched_slope_count():
    layer = PReLU(num_parameters=5)
    with pytest.raises(ValueError, match="1 or 3 elements"):
        layer(Tensor(np.zeros((2, 3))))


def test_prelu_slope_receives_a_gradient(rng):
    import pynn.core.math as pmath

    layer = PReLU()
    values = Tensor(rng.standard_normal((6, 4)))

    pmath.sum(layer(values)).backward()

    alpha = layer.parameters["alpha"]
    # dy/dalpha is x wherever x is negative, summed.
    assert np.allclose(alpha.grad, values.data[values.data < 0].sum())


def test_a_prelu_inside_a_layer_is_registered_and_trained(rng):
    """`activation="prelu"` puts a parameter inside `Linear`; it has to be stepped."""
    from pynn.core.random import set_seed
    from pynn.nn import Linear, Sequential
    from pynn.nn.losses import MeanSquaredError
    from pynn.optim import SGD

    set_seed(0)
    model = Sequential([Linear(4, 6, activation="prelu"), Linear(6, 2)])
    X = Tensor(rng.standard_normal((8, 4)))
    y = Tensor(np.zeros((8, 2)))
    model(X)

    assert "0.act_fn.alpha" in model.named_parameters()
    assert "0.act_fn.alpha" in model.state_dict()

    before = model.state_dict()["0.act_fn.alpha"].copy()
    optimizer = SGD(model, learning_rate=0.1)
    loss_fn = MeanSquaredError()
    for _ in range(20):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.step()

    assert not np.array_equal(before, model.state_dict()["0.act_fn.alpha"])

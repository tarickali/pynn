"""The class-based Activation wrappers.

Each class exists only to bind hyperparameters to the matching function in
`pynn.functional.activations`, which the verify sweep gradient-checks. What is asserted
here is that the binding is real: that `ELU(alpha=0.5)` reaches `elu` with 0.5 rather
than silently taking the default.
"""

import numpy as np
import pytest

import pynn.functional as F
from pynn.core import Tensor
from pynn.nn.activations import (
    ELU,
    SELU,
    Affine,
    Identity,
    ReLU,
    Sigmoid,
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

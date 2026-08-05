"""Numerical gradient checks for every differentiable operation.

The bulk of this file is a single parametrization over `pynn.verify.gradient_cases`,
which is the same sweep `python -m pynn.verify` runs. Driving it from pytest gives one
test per operation, so a failure names the op instead of reporting one failure for a
132-case sweep. The sweep is the single source of truth for *which* operations are
covered; keeping a parallel hand-written list here meant the two could drift, and they
did — the sweep checks each op both alone and with its input reused, while the
hand-written tests only ever used a single consumer.

What remains below are the checks the sweep cannot express: agreement between two
different formulations of the same function, a gradient compared against a closed form
rather than against central differences, and a whole model's parameters at once.
"""

import numpy as np
import pytest

import pynn.functional as F
from pynn.core import Tensor
from pynn.functional.losses import binary_crossentropy, categorical_crossentropy
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.verify import GradientCase, check_gradients, gradient_cases

# --------------------------------------------------------------------------- #
# The shipped sweep, one pytest case per operation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", gradient_cases(), ids=lambda case: case.id)
def test_gradient_case(case: GradientCase) -> None:
    result = check_gradients(case.fn, case.inputs, eps=case.eps, rtol=case.rtol)
    assert result.passed, f"\n{result}"


def test_sweep_covers_every_operation() -> None:
    """Guard against the sweep silently shrinking.

    The count is deliberately a lower bound rather than an exact number, so that adding
    a case does not fail the suite, but deleting a block of them does.
    """
    names = {case.name for case in gradient_cases()}
    assert len(names) >= 130, f"sweep shrank to {len(names)} cases"

    for expected in ["add", "matmul", "conv2d", "loss mse", "softmax", "graph diamond"]:
        assert any(expected in name for name in names), f"no {expected!r} case"

    # Every unary and binary op must appear in its reused-input form too; that is the
    # variant that catches missing gradient accumulation.
    reused = [name for name in names if "reused" in name]
    assert len(reused) >= 25, f"only {len(reused)} reuse cases"


# --------------------------------------------------------------------------- #
# Agreement between formulations
#
# The fused losses compute a gradient with respect to their logits directly. That
# has to agree with composing the activation and the probability form, which is a
# different code path through the graph.
# --------------------------------------------------------------------------- #


def test_binary_crossentropy_through_sigmoid(rng) -> None:
    targets = Tensor(rng.integers(0, 2, (6, 1)).astype(np.float64))
    logits = rng.standard_normal((6, 1))

    fused = Tensor(logits.copy())
    binary_crossentropy(targets, fused, logits=True).backward()

    composed = Tensor(logits.copy())
    binary_crossentropy(targets, F.sigmoid(composed), logits=False).backward()

    assert np.allclose(fused.grad, composed.grad)


def test_categorical_crossentropy_through_softmax(rng) -> None:
    targets = Tensor(np.eye(4)[rng.integers(0, 4, 6)])
    logits = rng.standard_normal((6, 4))

    fused = Tensor(logits.copy())
    categorical_crossentropy(targets, fused, logits=True).backward()

    composed = Tensor(logits.copy())
    categorical_crossentropy(targets, F.softmax(composed), logits=False).backward()

    assert np.allclose(fused.grad, composed.grad)


def test_categorical_crossentropy_matches_closed_form(rng) -> None:
    """dL/dlogits for softmax cross-entropy is exactly (softmax(z) - y) / batch.

    Central differences would accept a gradient that is merely close; this pins the
    exact expression, and would catch a missing or extra batch-size division.
    """
    targets = np.eye(4)[rng.integers(0, 4, 6)]
    logits = rng.standard_normal((6, 4))

    tensor = Tensor(logits.copy())
    categorical_crossentropy(Tensor(targets), tensor, logits=True).backward()

    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=-1, keepdims=True)
    expected = (probabilities - targets) / logits.shape[0]

    assert np.allclose(tensor.grad, expected)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_mlp_parameter_gradients(activation: str) -> None:
    """Every parameter of a real model at once, rather than one op in isolation.

    Catches gradients that are individually right but wired to the wrong parameter.
    """
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

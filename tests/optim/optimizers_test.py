"""Optimizer update rules against independent closed-form references.

Each reference below is a direct transcription of the update rule as written in
PyTorch's documentation for that optimizer, computed with plain floats. Comparing
against a separate implementation catches algebra errors that a "loss went down" test
cannot: a broken momentum buffer still descends, just more slowly.
"""

import numpy as np
import pytest

from pynn.core import Tensor
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.optim import SGD, Adadelta, Adagrad, Adam, RMSprop

STEPS = 6
GRADIENT = 0.7
START = 1.0


def run_optimizer(optimizer_cls, **kwargs):
    """Step one scalar parameter with a constant gradient, returning its trajectory."""
    param = Tensor(np.array([START]))
    optimizer = optimizer_cls([{"w": param}], **kwargs)

    trajectory = []
    for _ in range(STEPS):
        param.grad = np.array([GRADIENT])
        optimizer.update()
        trajectory.append(float(param.data[0]))
    return trajectory


def test_sgd_plain():
    lr = 0.1
    expected, value = [], START
    for _ in range(STEPS):
        value -= lr * GRADIENT
        expected.append(value)

    assert run_optimizer(SGD, learning_rate=lr) == pytest.approx(expected)


def test_sgd_momentum():
    lr, momentum = 0.1, 0.9
    expected, value, buffer = [], START, None
    for _ in range(STEPS):
        buffer = GRADIENT if buffer is None else momentum * buffer + GRADIENT
        value -= lr * buffer
        expected.append(value)

    assert run_optimizer(SGD, learning_rate=lr, momentum=momentum) == pytest.approx(
        expected
    )


def test_sgd_momentum_accelerates():
    """With a constant gradient, momentum must take strictly larger steps over time."""
    trajectory = run_optimizer(SGD, learning_rate=0.1, momentum=0.9)
    steps = -np.diff([START, *trajectory])

    assert np.all(np.diff(steps) > 0), f"steps did not grow: {steps}"


def test_sgd_dampening():
    lr, momentum, dampening = 0.1, 0.9, 0.5
    expected, value, buffer = [], START, None
    for _ in range(STEPS):
        if buffer is None:
            buffer = GRADIENT
        else:
            buffer = momentum * buffer + (1 - dampening) * GRADIENT
        value -= lr * buffer
        expected.append(value)

    trajectory = run_optimizer(
        SGD, learning_rate=lr, momentum=momentum, dampening=dampening
    )
    assert trajectory == pytest.approx(expected)


def test_sgd_nesterov():
    lr, momentum = 0.1, 0.9
    expected, value, buffer = [], START, None
    for _ in range(STEPS):
        buffer = GRADIENT if buffer is None else momentum * buffer + GRADIENT
        value -= lr * (GRADIENT + momentum * buffer)
        expected.append(value)

    trajectory = run_optimizer(SGD, learning_rate=lr, momentum=momentum, nesterov=True)
    assert trajectory == pytest.approx(expected)


def test_sgd_weight_decay():
    lr, weight_decay = 0.1, 0.05
    expected, value = [], START
    for _ in range(STEPS):
        value -= lr * (GRADIENT + weight_decay * value)
        expected.append(value)

    trajectory = run_optimizer(SGD, learning_rate=lr, weight_decay=weight_decay)
    assert trajectory == pytest.approx(expected)


def test_sgd_maximize_ascends():
    trajectory = run_optimizer(SGD, learning_rate=0.1, maximize=True)
    assert np.all(np.diff([START, *trajectory]) > 0)


def test_adam():
    lr, beta_1, beta_2, eps = 0.01, 0.9, 0.999, 1e-8
    expected, value, m, v = [], START, 0.0, 0.0
    for step in range(1, STEPS + 1):
        m = beta_1 * m + (1 - beta_1) * GRADIENT
        v = beta_2 * v + (1 - beta_2) * GRADIENT**2
        m_hat = m / (1 - beta_1**step)
        v_hat = v / (1 - beta_2**step)
        value -= lr * m_hat / (np.sqrt(v_hat) + eps)
        expected.append(value)

    assert run_optimizer(Adam, learning_rate=lr) == pytest.approx(expected)


def test_adam_amsgrad():
    lr, beta_1, beta_2, eps = 0.01, 0.9, 0.999, 1e-8
    expected, value, m, v, v_max = [], START, 0.0, 0.0, 0.0
    for step in range(1, STEPS + 1):
        m = beta_1 * m + (1 - beta_1) * GRADIENT
        v = beta_2 * v + (1 - beta_2) * GRADIENT**2
        m_hat = m / (1 - beta_1**step)
        v_hat = v / (1 - beta_2**step)
        v_max = max(v_max, v_hat)
        value -= lr * m_hat / (np.sqrt(v_max) + eps)
        expected.append(value)

    trajectory = run_optimizer(Adam, learning_rate=lr, amsgrad=True)
    assert trajectory == pytest.approx(expected)


def test_rmsprop():
    lr, alpha, eps = 0.01, 0.99, 1e-10
    expected, value, square_average = [], START, 0.0
    for _ in range(STEPS):
        square_average = alpha * square_average + (1 - alpha) * GRADIENT**2
        value -= lr * GRADIENT / (np.sqrt(square_average) + eps)
        expected.append(value)

    assert run_optimizer(RMSprop, learning_rate=lr) == pytest.approx(expected)


def test_rmsprop_momentum():
    lr, alpha, momentum, eps = 0.01, 0.99, 0.9, 1e-10
    expected, value, square_average, buffer = [], START, 0.0, 0.0
    for _ in range(STEPS):
        square_average = alpha * square_average + (1 - alpha) * GRADIENT**2
        buffer = momentum * buffer + GRADIENT / (np.sqrt(square_average) + eps)
        value -= lr * buffer
        expected.append(value)

    trajectory = run_optimizer(RMSprop, learning_rate=lr, momentum=momentum)
    assert trajectory == pytest.approx(expected)


def test_rmsprop_centered():
    """Centering subtracts the squared running mean, estimating variance not power."""
    lr, alpha, eps = 0.01, 0.99, 1e-10
    expected, value, square_average, average = [], START, 0.0, 0.0
    for _ in range(STEPS):
        square_average = alpha * square_average + (1 - alpha) * GRADIENT**2
        average = alpha * average + (1 - alpha) * GRADIENT
        value -= lr * GRADIENT / (np.sqrt(square_average - average**2) + eps)
        expected.append(value)

    trajectory = run_optimizer(RMSprop, learning_rate=lr, centered=True)
    assert trajectory == pytest.approx(expected)
    assert trajectory != pytest.approx(run_optimizer(RMSprop, learning_rate=lr))


def test_rmsprop_maximize_ascends():
    """maximize was accepted but silently ignored; this pins the behavior down."""
    trajectory = run_optimizer(RMSprop, learning_rate=0.01, maximize=True)
    assert np.all(np.diff([START, *trajectory]) > 0)


def test_adagrad():
    lr, eps = 0.01, 1e-10
    expected, value, total = [], START, 0.0
    for _ in range(STEPS):
        total += GRADIENT**2
        value -= lr * GRADIENT / (np.sqrt(total) + eps)
        expected.append(value)

    assert run_optimizer(Adagrad, learning_rate=lr) == pytest.approx(expected)


def test_adagrad_learning_rate_decay():
    lr, decay, eps = 0.01, 0.1, 1e-10
    expected, value, total = [], START, 0.0
    for step in range(STEPS):
        total += GRADIENT**2
        step_lr = lr / (1 + step * decay)
        value -= step_lr * GRADIENT / (np.sqrt(total) + eps)
        expected.append(value)

    trajectory = run_optimizer(Adagrad, learning_rate=lr, learning_rate_decay=decay)
    assert trajectory == pytest.approx(expected)


def test_adadelta():
    lr, rho, eps = 1.0, 0.9, 1e-10
    expected, value, average, accumulator = [], START, 0.0, 0.0
    for _ in range(STEPS):
        average = rho * average + (1 - rho) * GRADIENT**2
        delta = np.sqrt((accumulator + eps) / (average + eps)) * GRADIENT
        accumulator = rho * accumulator + (1 - rho) * delta**2
        value -= lr * delta
        expected.append(value)

    assert run_optimizer(Adadelta, learning_rate=lr) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Behavior shared by every optimizer
# --------------------------------------------------------------------------- #

ALL_OPTIMIZERS = [SGD, Adam, RMSprop, Adagrad, Adadelta]


@pytest.mark.parametrize("optimizer_cls", ALL_OPTIMIZERS, ids=lambda c: c.__name__)
def test_optimizer_reduces_loss(optimizer_cls):
    rng = np.random.default_rng(3)
    X = Tensor(rng.standard_normal((16, 4)))
    y = Tensor(rng.standard_normal((16, 1)))

    model = Sequential([Linear(4, 8, activation="tanh"), Linear(8, 1)])
    loss_fn = MeanSquaredError()
    optimizer = optimizer_cls(model, learning_rate=0.05)

    first = float(loss_fn(y, model(X)).item())
    for _ in range(50):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.update()
    last = float(loss_fn(y, model(X)).item())

    assert last < first


@pytest.mark.parametrize("optimizer_cls", ALL_OPTIMIZERS, ids=lambda c: c.__name__)
def test_optimizer_reset_reproduces_trajectory(optimizer_cls):
    param = Tensor(np.array([START]))
    optimizer = optimizer_cls([{"w": param}], learning_rate=0.05)

    def trajectory():
        values = []
        for _ in range(4):
            param.grad = np.array([GRADIENT])
            optimizer.update()
            values.append(float(param.data[0]))
        return values

    first = trajectory()

    param.data = np.array([START])
    optimizer.reset()
    assert trajectory() == pytest.approx(first)


@pytest.mark.parametrize("optimizer_cls", ALL_OPTIMIZERS, ids=lambda c: c.__name__)
def test_optimizer_handles_lazily_built_parameters(optimizer_cls):
    """Layers build lazily, so parameters appear after the optimizer exists."""
    model = Sequential([Linear(4, 3)])
    optimizer = optimizer_cls(model, learning_rate=0.1)
    assert model.parameter_groups() == [{}, {}]

    X = Tensor(np.random.default_rng(0).standard_normal((5, 4)))
    loss = MeanSquaredError()(Tensor(np.zeros((5, 3))), model(X))
    model.zero_grad()
    loss.backward()

    before = model.modules[0].parameters["W"].data.copy()
    optimizer.update()

    assert not np.allclose(before, model.modules[0].parameters["W"].data)


# --------------------------------------------------------------------------- #
# Frozen parameters
#
# A frozen parameter still receives a gradient from backward(); it is the optimizer
# that has to leave it alone. "Loss went down" cannot catch a leak here, because a
# model that trains the layer it was told to freeze trains perfectly well.
# --------------------------------------------------------------------------- #


def _trained_pair(optimizer_cls, freeze_first: bool, freeze_before_build: bool):
    """Train a two-layer model with the first layer optionally frozen."""
    rng = np.random.default_rng(5)
    X = Tensor(rng.standard_normal((8, 4)))
    y = Tensor(rng.standard_normal((8, 2)))

    model = Sequential([Linear(4, 3, activation="tanh"), Linear(3, 2)])
    if freeze_first and freeze_before_build:
        model.modules[0].freeze()
    if not freeze_before_build:
        model(X)  # force the lazy build so the parameters exist first
        if freeze_first:
            model.modules[0].freeze()

    optimizer = optimizer_cls(model, learning_rate=0.1)
    loss_fn = MeanSquaredError()

    model(X)  # ensure built before snapshotting
    before = {
        index: {name: p.data.copy() for name, p in module.parameters.items()}
        for index, module in enumerate(model.modules)
    }

    for _ in range(3):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.update()

    return model, before


@pytest.mark.parametrize("optimizer_cls", ALL_OPTIMIZERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "freeze_before_build", [True, False], ids=["prebuild", "built"]
)
def test_frozen_parameters_are_not_updated(optimizer_cls, freeze_before_build):
    model, before = _trained_pair(
        optimizer_cls, freeze_first=True, freeze_before_build=freeze_before_build
    )

    for name, param in model.modules[0].parameters.items():
        assert np.array_equal(param.data, before[0][name]), (
            f"frozen parameter {name!r} was updated"
        )
        # The gradient must still have been computed; freezing is not detaching.
        assert np.any(param.grad != 0.0), f"frozen parameter {name!r} got no gradient"

    for name, param in model.modules[1].parameters.items():
        assert not np.array_equal(param.data, before[1][name]), (
            f"trainable parameter {name!r} was skipped"
        )


@pytest.mark.parametrize("optimizer_cls", ALL_OPTIMIZERS, ids=lambda c: c.__name__)
def test_unfreeze_restores_updates(optimizer_cls):
    param = Tensor(np.array([START]))
    module = Linear(2, 1)
    module.parameters["w"] = param

    optimizer = optimizer_cls([module.parameters], learning_rate=0.1)

    module.freeze()
    param.grad = np.array([GRADIENT])
    optimizer.update()
    assert param.data.tolist() == [START]

    module.unfreeze()
    param.grad = np.array([GRADIENT])
    optimizer.update()
    assert param.data.tolist() != [START]

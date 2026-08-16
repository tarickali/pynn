"""Optimizer update rules against independent closed-form references.

Each reference below is a direct transcription of the update rule as written in
PyTorch's documentation for that optimizer, computed with plain floats. Comparing
against a separate implementation catches algebra errors that a "loss went down" test
cannot: a broken momentum buffer still descends, just more slowly.
"""

from itertools import pairwise

import numpy as np
import pytest

from pynn.core import Tensor
from pynn.core.optimizer import effective_gradient
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.optim import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    CosineAnnealingLR,
    ExponentialLR,
    NAdam,
    OneCycleLR,
    ReduceLROnPlateau,
    RMSprop,
    StepLR,
    clip_grad_norm,
    clip_grad_value,
)

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
# NAdam
#
# The reference below is the published rule, in float64 throughout. It is worth
# noting that `torch.optim.NAdam` disagrees with it by about 1e-10 on a float64
# parameter, because it keeps `mu_product` and `step` as float32 tensors regardless
# of the parameter's dtype. Transcribing the rule rather than deferring to torch is
# the point of these tests, and this is the case where it shows.
# --------------------------------------------------------------------------- #


def nadam_reference(
    lr=0.01,
    beta_1=0.9,
    beta_2=0.999,
    eps=1e-8,
    momentum_decay=0.004,
    decay=0.0,
    decoupled=False,
    gradient=GRADIENT,
):
    """The published NAdam update, transcribed with plain floats."""
    expected, value, m, v, mu_product = [], START, 0.0, 0.0, 1.0
    for step in range(1, STEPS + 1):
        g = gradient + (0.0 if decoupled else decay * value)
        mu = beta_1 * (1 - 0.5 * 0.96 ** (step * momentum_decay))
        mu_next = beta_1 * (1 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
        mu_product *= mu

        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * g**2

        mhat = mu_next * m / (1 - mu_product * mu_next) + (1 - mu) * g / (
            1 - mu_product
        )
        vhat = v / (1 - beta_2**step)

        if decoupled:
            value -= lr * decay * value
        value -= lr * mhat / (np.sqrt(vhat) + eps)
        expected.append(value)
    return expected


def test_nadam():
    assert run_optimizer(NAdam, learning_rate=0.01) == pytest.approx(
        nadam_reference(), rel=1e-12
    )


def test_nadam_weight_decay():
    trajectory = run_optimizer(NAdam, learning_rate=0.01, weight_decay=0.1)
    assert trajectory == pytest.approx(nadam_reference(decay=0.1), rel=1e-12)


def test_nadam_decoupled_weight_decay():
    trajectory = run_optimizer(
        NAdam, learning_rate=0.01, weight_decay=0.1, decoupled_weight_decay=True
    )
    assert trajectory == pytest.approx(
        nadam_reference(decay=0.1, decoupled=True), rel=1e-12
    )
    assert trajectory != pytest.approx(nadam_reference(decay=0.1))


def test_nadam_momentum_decay_changes_the_warmup():
    trajectory = run_optimizer(NAdam, learning_rate=0.01, momentum_decay=0.02)
    assert trajectory == pytest.approx(nadam_reference(momentum_decay=0.02), rel=1e-12)
    assert trajectory != pytest.approx(nadam_reference())


def test_nadam_mu_product_is_a_running_product_not_a_power():
    """The bias correction for a coefficient that changes every step.

    `beta_1 ** t` is the tempting shortcut and is wrong, because `mu` warms up rather
    than staying put. Recomputing the product here is what catches it.
    """
    param = Tensor(np.array([START]))
    optimizer = NAdam([{"w": param}], learning_rate=0.01)

    expected = 1.0
    for step in range(1, 5):
        param.grad = np.array([GRADIENT])
        optimizer.update()
        expected *= 0.9 * (1 - 0.5 * 0.96 ** (step * 0.004))
        assert optimizer.mu_product == pytest.approx(expected, rel=1e-14)
        assert optimizer.mu_product != pytest.approx(0.9**step)


def test_nadam_reset_clears_the_mu_product():
    param = Tensor(np.array([START]))
    optimizer = NAdam([{"w": param}], learning_rate=0.01)
    for _ in range(3):
        param.grad = np.array([GRADIENT])
        optimizer.update()

    optimizer.reset()
    assert optimizer.mu_product == 1.0


def test_nadam_differs_from_adam():
    """Otherwise the Nesterov terms could be present and cancelling."""
    assert run_optimizer(NAdam, learning_rate=0.01) != pytest.approx(
        run_optimizer(Adam, learning_rate=0.01)
    )


# --------------------------------------------------------------------------- #
# Behavior shared by every optimizer
# --------------------------------------------------------------------------- #

ALL_OPTIMIZERS = [SGD, Adam, RMSprop, Adagrad, Adadelta, NAdam]

# Every flag combination that takes a different path through an update rule. The
# arithmetic is covered by the closed-form references above; these exist because the
# update rules run in place, and the branches that reuse a scratch array are where an
# aliasing mistake would live.
IN_PLACE_FLAGS = [
    (SGD, {}),
    (SGD, {"momentum": 0.9}),
    (SGD, {"momentum": 0.9, "dampening": 0.4}),
    (SGD, {"momentum": 0.9, "nesterov": True}),
    (SGD, {"weight_decay": 0.1}),
    (SGD, {"maximize": True}),
    (Adam, {}),
    (Adam, {"amsgrad": True}),
    (Adam, {"weight_decay": 0.1}),
    (NAdam, {}),
    (NAdam, {"weight_decay": 0.1}),
    (NAdam, {"weight_decay": 0.1, "decoupled_weight_decay": True}),
    (NAdam, {"momentum_decay": 0.02}),
    (AdamW, {}),
    (AdamW, {"weight_decay": 0.0}),
    (AdamW, {"amsgrad": True}),
    (RMSprop, {}),
    (RMSprop, {"centered": True}),
    (RMSprop, {"momentum": 0.9}),
    (RMSprop, {"centered": True, "momentum": 0.9}),
    (Adagrad, {}),
    (Adagrad, {"initial_accumulator_value": 0.5}),
    (Adagrad, {"learning_rate_decay": 0.1}),
    (Adadelta, {}),
    (Adadelta, {"rho": 0.5}),
]
IN_PLACE_IDS = [
    f"{cls.__name__}-{'-'.join(f'{k}={v}' for k, v in flags.items()) or 'defaults'}"
    for cls, flags in IN_PLACE_FLAGS
]


@pytest.mark.parametrize("optimizer_cls,flags", IN_PLACE_FLAGS, ids=IN_PLACE_IDS)
def test_update_leaves_the_gradient_alone(optimizer_cls, flags):
    """A rule that consumed `param.grad` takes a right first step and a wrong second.

    Nothing else notices: `zero_grad()` overwrites the damage before the next backward
    pass, so the only visible symptom is a trajectory that is slightly off from the
    second step onward — which is what the reference tests would show, if the constant
    gradient they use were not restored by hand between steps.
    """
    param = Tensor(np.array([1.0, -2.0, 0.5]))
    optimizer = optimizer_cls([{"w": param}], learning_rate=0.01, **flags)

    for _ in range(3):
        param.grad = np.array([0.7, -0.3, 0.1])
        gradient, before = param.grad, param.grad.copy()
        optimizer.update()

        assert np.array_equal(gradient, before), "the update modified param.grad"


@pytest.mark.parametrize("optimizer_cls,flags", IN_PLACE_FLAGS, ids=IN_PLACE_IDS)
def test_update_writes_through_param_data(optimizer_cls, flags):
    """The step is applied to the array, not to the attribute holding it.

    This is a deliberate behavior, not an implementation detail: `Tensor.numpy()`
    hands the array out, and a caller holding it should see training happen, the way
    it would in PyTorch. `state_dict()` and `detach()` copy, so a checkpoint is still
    a snapshot.
    """
    param = Tensor(np.array([1.0, -2.0, 0.5]))
    optimizer = optimizer_cls([{"w": param}], learning_rate=0.01, **flags)
    held = param.numpy()
    initial = held.copy()

    param.grad = np.array([0.7, -0.3, 0.1])
    optimizer.update()

    assert held is param.data
    assert not np.array_equal(held, initial)


def test_effective_gradient_aliases_the_gradient_when_there_is_nothing_to_do():
    """The common case, and the whole reason it is a function rather than two lines."""
    grad = np.array([1.0, 2.0])
    data = np.array([3.0, 4.0])

    g = effective_gradient(grad, data, weight_decay=0.0, maximize=False)

    assert np.shares_memory(g, grad)
    assert g.tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "weight_decay,maximize,expected",
    [
        (0.0, False, [1.0, 2.0]),
        (0.0, True, [-1.0, -2.0]),
        (0.5, False, [2.5, 4.0]),
        (0.5, True, [0.5, 0.0]),
    ],
)
def test_effective_gradient_applies_both_adjustments(weight_decay, maximize, expected):
    grad = np.array([1.0, 2.0])
    data = np.array([3.0, 4.0])

    g = effective_gradient(grad, data, weight_decay, maximize)

    assert g.tolist() == expected
    assert grad.tolist() == [1.0, 2.0], "the caller's gradient was modified"


@pytest.mark.parametrize(
    "weight_decay,maximize", [(0.0, False), (0.0, True), (0.5, False), (0.5, True)]
)
def test_effective_gradient_refuses_to_be_written_through(weight_decay, maximize):
    """It aliases `param.grad` in the common case, so writing to it has to raise.

    Read-only in *every* case rather than only the aliasing one, or the guard would
    hold for the default flags and quietly lapse under `maximize=True`.
    """
    g = effective_gradient(
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), weight_decay, maximize
    )

    with pytest.raises(ValueError, match="read-only"):
        g *= 2.0


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


# --------------------------------------------------------------------------- #
# AdamW
#
# The difference from Adam(weight_decay=...) is one line, and comparing against a
# reference that folds the decay into the gradient is how you find out whether it is
# the right line.
# --------------------------------------------------------------------------- #


def test_adamw_matches_the_decoupled_reference():
    lr, beta_1, beta_2, eps, decay = 0.01, 0.9, 0.999, 1e-8, 0.1
    expected, value, m, v = [], START, 0.0, 0.0
    for step in range(1, STEPS + 1):
        m = beta_1 * m + (1 - beta_1) * GRADIENT
        v = beta_2 * v + (1 - beta_2) * GRADIENT**2
        mhat = m / (1 - beta_1**step)
        vhat = v / (1 - beta_2**step)
        value -= lr * decay * value  # decoupled: applied to the parameter, not g
        value -= lr * mhat / (np.sqrt(vhat) + eps)
        expected.append(value)

    trajectory = run_optimizer(AdamW, learning_rate=lr, weight_decay=decay)
    assert trajectory == pytest.approx(expected)


def test_adamw_without_decay_is_adam():
    assert run_optimizer(AdamW, learning_rate=0.01, weight_decay=0.0) == pytest.approx(
        run_optimizer(Adam, learning_rate=0.01, weight_decay=0.0)
    )


def test_adamw_decay_is_not_rescaled_by_the_second_moment():
    """The whole reason it exists: Adam's coupled decay is divided by sqrt(v).

    With the same weight_decay, Adam pushes the parameter toward zero by an amount
    that depends on its gradient history and AdamW does not, so the two trajectories
    have to differ.
    """
    coupled = run_optimizer(Adam, learning_rate=0.01, weight_decay=0.1)
    decoupled = run_optimizer(AdamW, learning_rate=0.01, weight_decay=0.1)

    assert coupled != pytest.approx(decoupled)


def test_adamw_defaults_to_a_non_zero_decay():
    """Unlike Adam — decoupled decay is the reason to pick this optimizer."""
    from pynn.core import Tensor as _Tensor

    assert AdamW([{"w": _Tensor(np.array([1.0]))}]).weight_decay == 0.01


# --------------------------------------------------------------------------- #
# Learning-rate schedules
# --------------------------------------------------------------------------- #


def make_scheduled(scheduler_cls, base_lr=0.1, **kwargs):
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=base_lr)
    return optimizer, scheduler_cls(optimizer, **kwargs)


def rates(optimizer, scheduler, epochs):
    values = [optimizer.learning_rate]
    for _ in range(epochs):
        scheduler.step()
        values.append(optimizer.learning_rate)
    return values


def test_step_lr_drops_on_a_staircase():
    optimizer, scheduler = make_scheduled(StepLR, step_size=3, gamma=0.5)

    assert rates(optimizer, scheduler, 6) == pytest.approx(
        [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025]
    )


def test_exponential_lr_decays_every_epoch():
    optimizer, scheduler = make_scheduled(ExponentialLR, gamma=0.9)

    assert rates(optimizer, scheduler, 4) == pytest.approx(
        [0.1 * 0.9**epoch for epoch in range(5)]
    )


def test_cosine_annealing_reaches_eta_min_at_t_max():
    optimizer, scheduler = make_scheduled(CosineAnnealingLR, T_max=10, eta_min=0.001)

    values = rates(optimizer, scheduler, 10)

    assert values[0] == pytest.approx(0.1)
    assert values[-1] == pytest.approx(0.001)
    assert values[5] == pytest.approx((0.1 + 0.001) / 2)


def test_cosine_annealing_is_monotone_within_a_period():
    optimizer, scheduler = make_scheduled(CosineAnnealingLR, T_max=20)
    values = rates(optimizer, scheduler, 20)

    assert all(later <= earlier for earlier, later in pairwise(values))


def test_a_schedule_is_a_function_of_the_epoch_not_the_current_rate():
    """So it can be resumed, and an extra `step` cannot compound a rounding error."""
    optimizer, scheduler = make_scheduled(ExponentialLR, gamma=0.9)
    rates(optimizer, scheduler, 5)
    expected = optimizer.learning_rate

    # Rewind by hand; a schedule that multiplied its own output could not do this.
    scheduler.last_epoch = 4
    scheduler.step()
    assert optimizer.learning_rate == pytest.approx(expected)


def test_a_schedule_round_trips_through_its_state_dict():
    optimizer, scheduler = make_scheduled(CosineAnnealingLR, T_max=10)
    rates(optimizer, scheduler, 4)
    state = scheduler.state_dict()
    expected = optimizer.learning_rate

    other_optimizer, other = make_scheduled(CosineAnnealingLR, base_lr=999.0, T_max=10)
    other.load_state_dict(state)

    assert other_optimizer.learning_rate == pytest.approx(expected)


def test_the_schedule_actually_reaches_the_optimizer():
    """A schedule that computed rates nobody read would pass every test above."""
    optimizer, scheduler = make_scheduled(StepLR, step_size=1, gamma=0.0)
    param = optimizer.parameters[0]["w"]

    scheduler.step()
    param.grad = np.array([GRADIENT])
    optimizer.update()

    assert param.data.tolist() == [START], "a zero rate must not move the parameter"


@pytest.mark.parametrize(
    "scheduler_cls,kwargs,message",
    [
        (StepLR, {"step_size": 0}, "step_size must be positive"),
        (CosineAnnealingLR, {"T_max": 0}, "T_max must be positive"),
    ],
)
def test_schedules_reject_invalid_periods(scheduler_cls, kwargs, message):
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=0.1)
    with pytest.raises(ValueError, match=message):
        scheduler_cls(optimizer, **kwargs)


# --------------------------------------------------------------------------- #
# Gradient clipping
# --------------------------------------------------------------------------- #


def gradients_of(*values):
    module = Linear(2, 1)
    module.parameters["a"] = Tensor(np.array(values[0], dtype=float))
    if len(values) > 1:
        module.parameters["b"] = Tensor(np.array(values[1], dtype=float))
    for index, parameter in enumerate(module.parameters.values()):
        parameter.grad = np.array(values[index], dtype=float)
    return module


def total_norm(module):
    return float(
        np.sqrt(sum((p.grad**2).sum() for p in module.named_parameters().values()))
    )


def test_clip_grad_norm_returns_the_norm_before_clipping():
    module = gradients_of([3.0, 4.0])

    reported = clip_grad_norm(module, max_norm=1.0)

    assert reported == pytest.approx(5.0)
    assert total_norm(module) == pytest.approx(1.0, rel=1e-4)


def test_clip_grad_norm_leaves_a_small_gradient_alone():
    module = gradients_of([0.3, 0.4])

    assert clip_grad_norm(module, max_norm=10.0) == pytest.approx(0.5)
    assert module.parameters["a"].grad.tolist() == [0.3, 0.4]


def test_clip_grad_norm_preserves_direction():
    """It shortens the step; turning it would discard what the gradient got right."""
    module = gradients_of([3.0, 4.0], [12.0, 0.0])
    before = [p.grad.copy() for p in module.parameters.values()]

    clip_grad_norm(module, max_norm=1.0)

    for original, clipped in zip(before, module.parameters.values(), strict=True):
        cosine = np.dot(original, clipped.grad) / (
            np.linalg.norm(original) * np.linalg.norm(clipped.grad)
        )
        assert cosine == pytest.approx(1.0)


def test_clip_grad_norm_scales_every_parameter_by_one_factor():
    """Per-tensor scaling would change their relative sizes, i.e. the direction."""
    module = gradients_of([3.0, 4.0], [12.0, 0.0])
    ratio_before = np.linalg.norm(module.parameters["a"].grad) / np.linalg.norm(
        module.parameters["b"].grad
    )

    clip_grad_norm(module, max_norm=1.0)

    ratio_after = np.linalg.norm(module.parameters["a"].grad) / np.linalg.norm(
        module.parameters["b"].grad
    )
    assert ratio_after == pytest.approx(ratio_before)


def test_clip_grad_norm_supports_the_infinity_norm():
    module = gradients_of([3.0, -9.0])

    assert clip_grad_norm(module, max_norm=1.0, norm_type=float("inf")) == 9.0
    assert np.abs(module.parameters["a"].grad).max() == pytest.approx(1.0, rel=1e-4)


def test_clip_grad_norm_handles_a_model_with_no_parameters():
    assert clip_grad_norm(Sequential(), max_norm=1.0) == 0.0


def test_clip_grad_value_clamps_each_element():
    module = gradients_of([3.0, -9.0, 0.2])

    clip_grad_value(module, clip_value=1.0)

    assert module.parameters["a"].grad.tolist() == [1.0, -1.0, 0.2]


@pytest.mark.parametrize(
    "clip,kwargs,message",
    [
        (clip_grad_norm, {"max_norm": 0.0}, "max_norm must be positive"),
        (clip_grad_value, {"clip_value": -1.0}, "clip_value must be positive"),
    ],
)
def test_clipping_rejects_a_non_positive_bound(clip, kwargs, message):
    with pytest.raises(ValueError, match=message):
        clip(gradients_of([1.0]), **kwargs)


def test_clipping_rescues_a_run_that_a_single_batch_would_otherwise_wreck(rng):
    """The realistic use: one enormous gradient should not undo the whole run."""
    from pynn.core.random import set_seed

    X = Tensor(rng.standard_normal((8, 4)))
    y = Tensor(rng.standard_normal((8, 1)))
    loss_fn = MeanSquaredError()

    def train(clip: bool) -> float:
        set_seed(0)
        model = Sequential([Linear(4, 6, activation="tanh"), Linear(6, 1)])
        optimizer = SGD(model, learning_rate=0.05)
        for step in range(15):
            loss = loss_fn(y, model(X))
            model.zero_grad()
            loss.backward()
            if step == 5:  # a batch that blows up
                for parameter in model.named_parameters().values():
                    parameter.grad *= 1e4
            if clip:
                clip_grad_norm(model, max_norm=1.0)
            optimizer.update()
        return float(loss_fn(y, model(X)).item())

    assert train(clip=True) < train(clip=False)


# --------------------------------------------------------------------------- #
# OneCycleLR
# --------------------------------------------------------------------------- #


def one_cycle(total_steps=10, **kwargs):
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=999.0)
    schedule = OneCycleLR(optimizer, max_lr=1.0, total_steps=total_steps, **kwargs)
    values = [optimizer.learning_rate]
    for _ in range(total_steps):
        schedule.step()
        values.append(optimizer.learning_rate)
    return values


def test_one_cycle_starts_below_the_peak_and_ends_far_below_it():
    values = one_cycle(div_factor=25.0, final_div_factor=1e4)

    assert values[0] == pytest.approx(1.0 / 25.0)
    assert max(values) == pytest.approx(1.0)
    assert values[-1] == pytest.approx(1.0 / 25.0 / 1e4)


def test_one_cycle_peaks_at_pct_start():
    """The warmup ends where it was told to, not at the midpoint."""
    values = one_cycle(total_steps=100, pct_start=0.3)

    assert np.argmax(values) == 30


def test_one_cycle_rises_then_falls():
    values = one_cycle(total_steps=40, pct_start=0.25)
    peak = int(np.argmax(values))

    assert all(a < b for a, b in pairwise(values[: peak + 1]))
    assert all(a > b for a, b in pairwise(values[peak:]))


def test_one_cycle_ignores_the_optimizers_own_rate():
    """It is defined by max_lr, unlike every other schedule here."""
    assert one_cycle()[0] == pytest.approx(1.0 / 25.0)


def test_one_cycle_linear_anneal_differs_from_cosine():
    cosine = one_cycle(total_steps=20, anneal_strategy="cos")
    linear = one_cycle(total_steps=20, anneal_strategy="linear")

    assert cosine[0] == pytest.approx(linear[0])
    assert cosine[-1] == pytest.approx(linear[-1])
    assert cosine != pytest.approx(linear)


def test_one_cycle_linear_warmup_is_a_straight_line():
    values = one_cycle(total_steps=10, pct_start=0.5, anneal_strategy="linear")
    warmup = values[:6]

    increments = np.diff(warmup)
    assert np.allclose(increments, increments[0])


def test_one_cycle_refuses_to_be_stepped_past_its_total():
    """Continuing to return the floor would hide a wrong `total_steps`."""
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=0.1)
    schedule = OneCycleLR(optimizer, max_lr=1.0, total_steps=3)
    for _ in range(3):
        schedule.step()

    with pytest.raises(ValueError, match="built for 3 steps"):
        schedule.step()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"total_steps": 0}, "total_steps must be positive"),
        ({"total_steps": 5, "pct_start": 0.0}, r"pct_start must be in \(0, 1\)"),
        ({"total_steps": 5, "pct_start": 1.0}, r"pct_start must be in \(0, 1\)"),
        ({"total_steps": 5, "div_factor": 0.0}, "must be positive"),
        ({"total_steps": 5, "final_div_factor": -1.0}, "must be positive"),
        ({"total_steps": 5, "anneal_strategy": "quadratic"}, "'cos' or 'linear'"),
    ],
)
def test_one_cycle_rejects_invalid_arguments(kwargs, message):
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=0.1)
    with pytest.raises(ValueError, match=message):
        OneCycleLR(optimizer, max_lr=1.0, **kwargs)


def test_one_cycle_round_trips_through_its_state_dict():
    optimizer, schedule = make_scheduled(OneCycleLR, max_lr=1.0, total_steps=20)
    for _ in range(7):
        schedule.step()
    expected = optimizer.learning_rate

    other_optimizer, other = make_scheduled(
        OneCycleLR, base_lr=999.0, max_lr=1.0, total_steps=20
    )
    other.load_state_dict(schedule.state_dict())

    assert other_optimizer.learning_rate == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# ReduceLROnPlateau
#
# The one schedule that is not a function of the epoch, so it is not an LRScheduler
# and its `step` takes the metric it watches.
# --------------------------------------------------------------------------- #


def plateau(metrics, **kwargs):
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=1.0)
    schedule = ReduceLROnPlateau(optimizer, **kwargs)
    return [schedule.step(metric) for metric in metrics]


def test_plateau_reduces_after_patience_is_exceeded():
    """Patience is how many bad epochs to *tolerate*, so the cut is on the next one."""
    rates = plateau([1.0, 1.0, 1.0, 1.0], patience=2, factor=0.5)

    assert rates == pytest.approx([1.0, 1.0, 1.0, 0.5])


def test_plateau_leaves_an_improving_metric_alone():
    rates = plateau([1.0, 0.5, 0.25, 0.1, 0.05], patience=0, factor=0.5)

    assert rates == pytest.approx([1.0] * 5)


def test_plateau_resets_the_counter_on_an_improvement():
    improving = plateau([1.0, 1.0, 0.1, 1.0, 1.0, 1.0], patience=1, factor=0.5)
    flat = plateau([1.0] * 6, patience=1, factor=0.5)

    # The improvement at index 2 puts the bad-epoch count back to zero, so the first
    # cut lands two epochs later than it does with no improvement at all.
    assert improving == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    assert flat == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.25, 0.25])


def test_plateau_max_mode_watches_for_a_metric_that_should_rise():
    rising = plateau([0.1, 0.5, 0.9], mode="max", patience=0, factor=0.5)
    flat = plateau([0.9, 0.9, 0.9], mode="max", patience=0, factor=0.5)

    assert rising == pytest.approx([1.0, 1.0, 1.0])
    assert flat == pytest.approx([1.0, 0.5, 0.25])


def test_plateau_threshold_ignores_an_improvement_too_small_to_count():
    """Noise around a plateau would otherwise keep the counter at zero forever."""
    noisy = [1.0, 0.99999, 0.99998, 0.99997]

    assert plateau(noisy, patience=1, factor=0.5, threshold=1e-4) == pytest.approx(
        [1.0, 1.0, 0.5, 0.5]
    )
    assert plateau(noisy, patience=1, factor=0.5, threshold=0.0) == pytest.approx(
        [1.0] * 4
    )


def test_absolute_threshold_differs_from_relative():
    """A drop of 1 from 100: 1% of the best, but twenty times an absolute 0.05."""
    metrics = [100.0, 99.0, 99.0]

    relative = plateau(metrics, patience=0, factor=0.5, threshold=0.05)
    absolute = plateau(
        metrics, patience=0, factor=0.5, threshold=0.05, threshold_mode="abs"
    )

    # Relative wants better than 95.0, so 99.0 is a plateau and both epochs cut.
    assert relative == pytest.approx([1.0, 0.5, 0.25])
    # Absolute wants better than 99.95, so the first 99.0 is an improvement.
    assert absolute == pytest.approx([1.0, 1.0, 0.5])


def test_plateau_cooldown_pauses_the_counter_after_a_reduction():
    without = plateau([1.0] * 6, patience=0, factor=0.5, cooldown=0)
    with_cooldown = plateau([1.0] * 6, patience=0, factor=0.5, cooldown=2)

    assert without == pytest.approx([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    # Two epochs of silence after each cut, so a third as many reductions.
    assert with_cooldown == pytest.approx([1.0, 0.5, 0.5, 0.5, 0.25, 0.25])


def test_plateau_clamps_at_min_lr():
    rates = plateau([1.0] * 8, patience=0, factor=0.1, min_lr=0.01)

    assert min(rates) == pytest.approx(0.01)


def test_plateau_accepts_a_scalar_tensor():
    """The metric is usually a loss that just came off the tape."""
    rates = plateau([Tensor(np.array(1.0))] * 3, patience=0, factor=0.5)

    assert rates == pytest.approx([1.0, 0.5, 0.25])


def test_plateau_round_trips_through_its_state_dict():
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=1.0)
    schedule = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    for metric in [1.0, 1.0]:
        schedule.step(metric)

    other = ReduceLROnPlateau(
        SGD([{"w": Tensor(np.array([START]))}], learning_rate=1.0),
        patience=2,
        factor=0.5,
    )
    other.load_state_dict(schedule.state_dict())

    assert other.best == schedule.best
    assert other.num_bad_epochs == schedule.num_bad_epochs
    assert other.last_epoch == schedule.last_epoch


def test_plateau_is_not_an_lr_scheduler():
    """Its `step` takes an argument, so sharing the base class would break it."""
    from pynn.optim import LRScheduler

    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=1.0)
    assert not isinstance(ReduceLROnPlateau(optimizer), LRScheduler)


def test_plateau_repr_reports_where_it_stands():
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=1.0)
    schedule = ReduceLROnPlateau(optimizer, patience=1)
    schedule.step(0.5)

    text = repr(schedule)
    assert "ReduceLROnPlateau" in text
    assert "best=0.5" in text


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"mode": "minimum"}, "mode must be"),
        ({"threshold_mode": "relative"}, "threshold_mode must be"),
        ({"factor": 1.0}, r"factor must be in \(0, 1\)"),
        ({"factor": 0.0}, r"factor must be in \(0, 1\)"),
        ({"patience": -1}, "must be non-negative"),
        ({"cooldown": -1}, "must be non-negative"),
    ],
)
def test_plateau_rejects_invalid_arguments(kwargs, message):
    optimizer = SGD([{"w": Tensor(np.array([START]))}], learning_rate=1.0)
    with pytest.raises(ValueError, match=message):
        ReduceLROnPlateau(optimizer, **kwargs)


def test_plateau_relative_threshold_handles_a_negative_metric():
    """A metric that is getting worse must never read as an improvement.

    Written PyTorch's way, `best * (1 - threshold)` with a best of -5.0 puts the bar at
    -4.9995, so -4.9996 — a *worse* value — counts as progress and the counter resets
    forever. Taking the threshold as a magnitude of `best` is the same arithmetic for a
    positive metric and correct for a negative one.
    """
    worsening = [-5.0, -4.9996, -4.9992, -4.9988]

    assert plateau(worsening, patience=1, factor=0.5) == pytest.approx(
        [1.0, 1.0, 0.5, 0.5]
    )


def test_plateau_relative_threshold_still_tracks_a_negative_metric_improving():
    improving = [-5.0, -6.0, -7.0, -8.0]

    assert plateau(improving, patience=0, factor=0.5) == pytest.approx([1.0] * 4)

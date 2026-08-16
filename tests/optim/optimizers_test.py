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
# Behavior shared by every optimizer
# --------------------------------------------------------------------------- #

ALL_OPTIMIZERS = [SGD, Adam, RMSprop, Adagrad, Adadelta]

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

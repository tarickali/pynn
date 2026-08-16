"""Learning-rate schedules.

A schedule wraps an optimizer and rewrites its `learning_rate` between epochs. Every
`LRScheduler` here is a pure function of the epoch number rather than of the previous
learning rate, which is what makes them restartable: setting `last_epoch` reproduces
the rate that epoch had, with no accumulated drift from repeated multiplication.

    optimizer = SGD(model, learning_rate=0.1)
    schedule = CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        train_one_epoch()
        schedule.step()

`ReduceLROnPlateau` is the exception and is deliberately not an `LRScheduler`: it reads
a validation metric, so it cannot be a function of the epoch and its `step` takes an
argument. Its own docstring has the details.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

from pynn.core import Optimizer, Tensor

__all__ = [
    "CosineAnnealingLR",
    "ExponentialLR",
    "LRScheduler",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "StepLR",
]


class LRScheduler(ABC):
    """Base class for learning-rate schedules.

    Subclasses implement `compute_lr`, which maps `self.last_epoch` to a rate. Writing
    it as a function of the epoch rather than of the current rate matters: a schedule
    that repeatedly multiplies drifts if `step` is called an extra time, and cannot be
    resumed from a checkpoint without replaying every step that came before.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose `learning_rate` this rewrites.
    last_epoch : int, default -1
        Epoch to resume from. The default means "nothing has happened yet", and the
        constructor immediately advances to epoch 0 so the initial rate is applied
        before the first step of training.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        #: The rate every computation is relative to, captured before any rewriting.
        self.base_lr = optimizer.learning_rate
        self.last_epoch = last_epoch
        self.step()

    @abstractmethod
    def compute_lr(self) -> float:
        """The learning rate for `self.last_epoch`."""
        raise NotImplementedError

    def step(self) -> float:
        """Advance one epoch, write the new rate onto the optimizer, and return it."""
        self.last_epoch += 1
        self.optimizer.learning_rate = self.compute_lr()
        return self.optimizer.learning_rate

    def state_dict(self) -> dict[str, float | int]:
        """Enough to resume the schedule, without the optimizer it is attached to."""
        return {"base_lr": self.base_lr, "last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.base_lr = float(state["base_lr"])
        self.last_epoch = int(state["last_epoch"])
        self.optimizer.learning_rate = self.compute_lr()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_lr={self.base_lr}, "
            f"last_epoch={self.last_epoch}, lr={self.optimizer.learning_rate:.6g})"
        )


class StepLR(LRScheduler):
    """Multiply the rate by `gamma` every `step_size` epochs.

    The staircase: constant for a while, then a sharp drop. Crude, and still the
    schedule most image classifiers were trained with.

    Parameters
    ----------
    optimizer : Optimizer
    step_size : int
        Epochs between drops.
    gamma : float, default 0.1
        Factor applied at each drop.
    last_epoch : int, default -1

    Raises
    ------
    ValueError
        If `step_size` is not positive.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def compute_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class ExponentialLR(LRScheduler):
    """Multiply the rate by `gamma` every epoch.

    StepLR's staircase smoothed into a curve. Decays fast — `gamma=0.95` is already an
    18x reduction over 60 epochs — so it is usually set much closer to 1 than a
    `StepLR` gamma.

    Parameters
    ----------
    optimizer : Optimizer
    gamma : float
        Per-epoch decay factor.
    last_epoch : int, default -1
    """

    def __init__(
        self, optimizer: Optimizer, gamma: float, last_epoch: int = -1
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def compute_lr(self) -> float:
        return self.base_lr * self.gamma**self.last_epoch


class CosineAnnealingLR(LRScheduler):
    """Anneal from the base rate to `eta_min` along a half cosine over `T_max` epochs.

    Spends longer near both ends than a linear decay: slow to leave the high rate where
    the model is still moving, slow to reach the floor where it is settling. That shape
    is the reason it usually beats a staircase without any tuning.

    Past `T_max` the cosine continues, so the rate climbs back up — which is the warm
    restart the paper describes, and a surprise if you only meant to decay. Stop
    stepping at `T_max`, or use `T_max` equal to the total number of epochs.

    Parameters
    ----------
    optimizer : Optimizer
    T_max : int
        Epochs in a half period, i.e. epochs to reach `eta_min`.
    eta_min : float, default 0.0
        Rate at the bottom of the curve.
    last_epoch : int, default -1

    Raises
    ------
    ValueError
        If `T_max` is not positive.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def compute_lr(self) -> float:
        cosine = 1.0 + math.cos(math.pi * self.last_epoch / self.T_max)
        return self.eta_min + (self.base_lr - self.eta_min) * cosine / 2.0


class OneCycleLR(LRScheduler):
    """One warmup to `max_lr` and one long anneal to well below where it started.

    Stepped per *batch*, not per epoch — that is the whole point, and it is why
    `total_steps` is `epochs * batches_per_epoch` rather than a number of epochs. The
    rate rises from `max_lr / div_factor` to `max_lr` over the first `pct_start` of
    training and then anneals to `max_lr / div_factor / final_div_factor`, which is
    typically four orders of magnitude below the peak.

    The warmup is the part that does the work. A high rate reached immediately diverges;
    reached gradually, it acts as regularization, and the long tail afterwards is what
    lets the model settle into whatever basin the high rate found. `base_lr` from the
    optimizer is ignored — this schedule is defined by `max_lr`.

    **Momentum is not cycled**, unlike PyTorch's. Cycling it means writing an attribute
    only some optimizers have, and `LRScheduler`'s contract is that a schedule sets the
    learning rate. Set `momentum` on the optimizer directly if you want the pairing.

    Parameters
    ----------
    optimizer : Optimizer
    max_lr : float
        The peak, reached at the end of the warmup.
    total_steps : int
        Total number of `step()` calls the run will make: `epochs * steps_per_epoch`.
    pct_start : float, default 0.3
        Fraction of `total_steps` spent warming up, in (0, 1).
    anneal_strategy : {"cos", "linear"}, default "cos"
    div_factor : float, default 25.0
        `max_lr / div_factor` is the starting rate.
    final_div_factor : float, default 1e4
        The starting rate divided by this is the final rate.
    last_epoch : int, default -1

    Raises
    ------
    ValueError
        If `total_steps` is not positive, `pct_start` is outside (0, 1), either divisor
        is not positive, or `anneal_strategy` is not one of the two names. Also if
        `step()` is called more than `total_steps` times, rather than continuing to
        return the floor — running past the end of a one-cycle schedule means the
        `total_steps` it was built with was wrong, and silently flat is how that goes
        unnoticed.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not 0.0 < pct_start < 1.0:
            raise ValueError(f"pct_start must be in (0, 1), got {pct_start}")
        if div_factor <= 0 or final_div_factor <= 0:
            raise ValueError(
                "div_factor and final_div_factor must be positive, got "
                f"{div_factor} and {final_div_factor}"
            )
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(
                f"anneal_strategy must be 'cos' or 'linear', got {anneal_strategy!r}"
            )

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        #: The rate the cycle starts from, and the one the final anneal divides down.
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        super().__init__(optimizer, last_epoch)

    def _anneal(self, start: float, end: float, fraction: float) -> float:
        if self.anneal_strategy == "linear":
            return start + (end - start) * fraction
        cosine = 1.0 + math.cos(math.pi * fraction)
        return end + (start - end) * cosine / 2.0

    def compute_lr(self) -> float:
        step = self.last_epoch
        if step > self.total_steps:
            raise ValueError(
                f"OneCycleLR was built for {self.total_steps} steps and has been "
                f"stepped {step} times. Pass total_steps=epochs * steps_per_epoch."
            )
        warmup = self.pct_start * self.total_steps
        if step <= warmup:
            return self._anneal(self.initial_lr, self.max_lr, step / warmup)
        remaining = (step - warmup) / (self.total_steps - warmup)
        return self._anneal(self.max_lr, self.min_lr, remaining)


class ReduceLROnPlateau:
    """Cut the learning rate when a metric stops improving.

    **Deliberately not an `LRScheduler`.** Every schedule in this module is a pure
    function of the epoch number, which is what makes them resumable from `last_epoch`
    alone; this one is a function of the metrics it has been shown, so its `step` takes
    the metric as an argument and its state includes the best value seen, how many bad
    epochs have passed, and where it is in its cooldown. Giving it the same base class
    would mean a `step` with a different signature, which is worse than a separate
    type — the two are used differently and the difference is the interesting part.

    ```python
    schedule = ReduceLROnPlateau(optimizer, patience=5)
    for epoch in range(epochs):
        train_one_epoch()
        schedule.step(validation_loss)
    ```

    Parameters
    ----------
    optimizer : Optimizer
    mode : {"min", "max"}, default "min"
        Whether an improvement means a smaller metric (a loss) or a larger one (an
        accuracy).
    factor : float, default 0.1
        Multiplier applied to the rate when the metric plateaus.
    patience : int, default 10
        Epochs without improvement to tolerate before reducing. Reducing happens on the
        `patience + 1`-th, matching PyTorch.
    threshold : float, default 1e-4
        How much better counts as better, so that noise around a plateau does not keep
        resetting the counter.
    threshold_mode : {"rel", "abs"}, default "rel"
        Whether `threshold` is a fraction of the best value's *magnitude* or an
        absolute amount. The magnitude is the one divergence from PyTorch here: it
        writes the relative bar as `best * (1 - threshold)`, under which a negative
        metric improves by getting worse.
    cooldown : int, default 0
        Epochs to wait after a reduction before counting bad epochs again — the metric
        needs time to respond to the new rate.
    min_lr : float, default 0.0
        Floor the rate is clamped to.

    Raises
    ------
    ValueError
        If `mode` or `threshold_mode` is not one of its two names, if `factor` is not
        in (0, 1), or if `patience` or `cooldown` is negative.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(
                f"threshold_mode must be 'rel' or 'abs', got {threshold_mode!r}"
            )
        if not 0.0 < factor < 1.0:
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        if patience < 0 or cooldown < 0:
            raise ValueError(
                f"patience and cooldown must be non-negative, got {patience} "
                f"and {cooldown}"
            )

        self.optimizer = optimizer
        #: The rate at construction, so a reduction is readable against where it began.
        self.base_lr = optimizer.learning_rate
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr

        self.last_epoch = -1
        #: Worst possible starting point, so the first metric is always an improvement.
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def _is_better(self, metric: float) -> bool:
        if math.isinf(self.best):
            # Nothing recorded yet, so anything counts. Handled here rather than by a
            # separate flag because the relative margin below would be `inf * eps`,
            # and `inf - inf` is `nan`, which loses every comparison.
            return True

        # `abs(best)` rather than `best`, which is where this differs from PyTorch and
        # deliberately so. Written as `best * (1 - threshold)`, a *negative* metric
        # improves by getting worse: with a best of -5.0 the bar becomes -4.9995, so
        # -4.9996 reads as progress. Taking the threshold as a magnitude is the same
        # arithmetic wherever the metric is positive, which is nearly always, and right
        # where it is not.
        margin = (
            abs(self.best) * self.threshold
            if self.threshold_mode == "rel"
            else self.threshold
        )
        if self.mode == "min":
            return metric < self.best - margin
        return metric > self.best + margin

    def step(self, metric: float | Tensor) -> float:
        """Record `metric` for this epoch and return the resulting learning rate.

        Parameters
        ----------
        metric : float | Tensor
            The quantity being watched. A scalar Tensor is accepted, since the value
            being watched is usually a loss that just came off the tape.

        Returns
        -------
        float
            The optimizer's learning rate after this epoch, reduced or not.
        """
        value = float(metric.item()) if isinstance(metric, Tensor) else float(metric)
        self.last_epoch += 1

        if self._is_better(value):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            # A rate that has just changed needs a few epochs to show its effect, so
            # the bad-epoch count does not run during the cooldown.
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self.optimizer.learning_rate = max(
                self.optimizer.learning_rate * self.factor, self.min_lr
            )
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.optimizer.learning_rate

    def state_dict(self) -> dict[str, float | int]:
        """Enough to resume, without the optimizer it is attached to.

        The learning rate itself is not here, deliberately: it lives on the optimizer,
        which is checkpointed separately, and storing a second copy is how the two come
        back disagreeing.
        """
        return {
            "base_lr": self.base_lr,
            "last_epoch": self.last_epoch,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.base_lr = float(state["base_lr"])
        self.last_epoch = int(state["last_epoch"])
        self.best = float(state["best"])
        self.num_bad_epochs = int(state["num_bad_epochs"])
        self.cooldown_counter = int(state["cooldown_counter"])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(mode={self.mode!r}, factor={self.factor}, "
            f"patience={self.patience}, best={self.best:.6g}, "
            f"bad_epochs={self.num_bad_epochs}, "
            f"lr={self.optimizer.learning_rate:.6g})"
        )

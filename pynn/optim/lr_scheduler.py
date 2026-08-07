"""Learning-rate schedules.

A schedule wraps an optimizer and rewrites its `learning_rate` between epochs. Every
one here is a pure function of the epoch number rather than of the previous learning
rate, which is what makes them restartable: setting `last_epoch` reproduces the rate
that epoch had, with no accumulated drift from repeated multiplication.

    optimizer = SGD(model, learning_rate=0.1)
    schedule = CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        train_one_epoch()
        schedule.step()
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pynn.core import Optimizer

__all__ = [
    "CosineAnnealingLR",
    "ExponentialLR",
    "LRScheduler",
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

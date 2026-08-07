from typing import Any

import numpy as np

from pynn.core import Optimizer
from pynn.core.optimizer import ParameterSource
from pynn.utils.tensor import get_data_and_grad

__all__ = ["AdamW"]


class AdamW(Optimizer):
    """Adam with decoupled weight decay.

    The difference from `Adam(weight_decay=...)` is one line, and it is the whole point.
    Adam folds the decay into the gradient::

        g = grad + weight_decay * theta

    which then goes through the same `1 / (sqrt(v) + eps)` rescaling as everything else.
    A parameter with a large running second moment therefore gets *less* decay than one
    with a small moment — the regularization strength ends up depending on the gradient
    history, which is not what anyone means by weight decay.

    AdamW applies it directly to the parameter instead::

        theta -= learning_rate * weight_decay * theta
        theta -= learning_rate * mhat / (sqrt(vhat) + eps)

    so every parameter decays at the same rate regardless of its moments. That is the
    "decoupled" in the paper's title, and why `weight_decay=0.01` here is not comparable
    to `weight_decay=0.01` on `Adam`.

    Parameters
    ----------
    parameters : ParameterSource
        A Module, or explicit parameter groups.
    learning_rate : float, default 0.001
    lr : float | None
        Alias for `learning_rate`.
    beta_1, beta_2 : float
        Exponential decay rates for the first and second moment estimates.
    eps : float, default 1e-8
        Added to the denominator for numerical stability.
    weight_decay : float, default 0.01
        Decoupled decay. PyTorch's default is 0.01 rather than Adam's 0.0, since
        decoupled decay is the reason to reach for this optimizer at all.
    amsgrad : bool, default False
        Keep the running maximum of the second moment rather than the current value.
    maximize : bool, default False
        Ascend the objective rather than descend it.
    """

    def __init__(
        self,
        parameters: ParameterSource,
        learning_rate: float = 0.001,
        lr: float | None = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        learning_rate = lr if lr is not None else learning_rate
        super().__init__(parameters, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.cache: list[dict[str, dict[str, Any]]] = [
            {"momentum": {}, "velocity": {}, "vhat_max": {}}
            for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        t = self.time + 1
        for params, cache in zip(self.trainable_parameters(), self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = -grad if self.maximize else grad
                # Deliberately *not* folded into g — that is what makes it decoupled.
                if key not in cache["momentum"]:
                    cache["momentum"][key] = np.zeros_like(g)
                    cache["velocity"][key] = np.zeros_like(g)
                    cache["vhat_max"][key] = np.zeros_like(g)

                cache["momentum"][key] = (
                    self.beta_1 * cache["momentum"][key] + (1 - self.beta_1) * g
                )
                cache["velocity"][key] = self.beta_2 * cache["velocity"][key] + (
                    1 - self.beta_2
                ) * (g**2)

                mhat = cache["momentum"][key] / (1 - self.beta_1**t)
                vhat = cache["velocity"][key] / (1 - self.beta_2**t)
                if self.amsgrad:
                    cache["vhat_max"][key] = np.maximum(cache["vhat_max"][key], vhat)
                    vhat = cache["vhat_max"][key]

                decayed = data - self.learning_rate * self.weight_decay * data
                param.data = decayed - self.learning_rate * mhat / (
                    np.sqrt(vhat) + self.eps
                )
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"momentum": {}, "velocity": {}, "vhat_max": {}}
            for _ in range(len(self.parameters))
        ]

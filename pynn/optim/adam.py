from typing import Any

import numpy as np

from pynn.core import Optimizer, Tensor
from pynn.utils.tensor import get_data_and_grad

__all__ = ["Adam"]


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        parameters: list[dict[str, Tensor]],
        learning_rate: float = 0.001,
        lr: float | None = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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
        for params, cache in zip(self.parameters, self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = -grad if self.maximize else grad
                g = g + self.weight_decay * data
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

                param.data = param.data - self.learning_rate * mhat / (
                    np.sqrt(vhat) + self.eps
                )
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"momentum": {}, "velocity": {}, "vhat_max": {}}
            for _ in range(len(self.parameters))
        ]

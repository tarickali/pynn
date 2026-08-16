from typing import Any

import numpy as np

from pynn.core import Optimizer
from pynn.core.optimizer import ParameterSource, effective_gradient, state_buffer
from pynn.utils.tensor import get_data_and_grad

__all__ = ["Adam"]


class Adam(Optimizer):
    """Adam optimizer.

    Written in place — see `pynn.core.optimizer` for what that costs and why. Sixteen
    full-size allocations per parameter per step became three: the scratch array the
    two moment updates share, and the two bias-corrected moments.
    """

    def __init__(
        self,
        parameters: ParameterSource,
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
        for params, cache in zip(self.trainable_parameters(), self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = effective_gradient(grad, data, self.weight_decay, self.maximize)
                momentum = state_buffer(cache["momentum"], key, g)
                velocity = state_buffer(cache["velocity"], key, g)

                # momentum = beta_1 * momentum + (1 - beta_1) * g
                scratch = (1 - self.beta_1) * g
                momentum *= self.beta_1
                momentum += scratch

                # velocity = beta_2 * velocity + (1 - beta_2) * g**2, reusing the array
                # the moment update above already allocated.
                np.square(g, out=scratch)
                scratch *= 1 - self.beta_2
                velocity *= self.beta_2
                velocity += scratch

                mhat = momentum / (1 - self.beta_1**t)
                vhat = velocity / (1 - self.beta_2**t)
                if self.amsgrad:
                    vhat_max = state_buffer(cache["vhat_max"], key, g)
                    np.maximum(vhat_max, vhat, out=vhat)
                    np.copyto(vhat_max, vhat)

                # mhat and vhat are this step's own arrays, so the rest of the update
                # runs through them: sqrt(vhat) + eps, then learning_rate * mhat over
                # it, and only then does anything touch the parameter.
                np.sqrt(vhat, out=vhat)
                vhat += self.eps
                mhat *= self.learning_rate
                mhat /= vhat
                data -= mhat
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"momentum": {}, "velocity": {}, "vhat_max": {}}
            for _ in range(len(self.parameters))
        ]

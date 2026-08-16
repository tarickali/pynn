import numpy as np

from pynn.core import Optimizer
from pynn.core.optimizer import ParameterSource, effective_gradient, state_buffer
from pynn.utils.tensor import get_data_and_grad

__all__ = ["Adadelta"]


class Adadelta(Optimizer):
    """Adadelta optimizer.

    Written in place — see `pynn.core.optimizer` for what that costs and why.
    """

    def __init__(
        self,
        parameters: ParameterSource,
        learning_rate: float = 1.0,
        lr: float | None = None,
        rho: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        learning_rate = lr if lr is not None else learning_rate
        super().__init__(parameters, learning_rate)
        self.rho = rho
        self.weight_decay = weight_decay
        self.eps = eps
        self.maximize = maximize
        self.cache: list[dict[str, dict[str, np.ndarray]]] = [
            {"average": {}, "accumulator": {}} for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        for params, cache in zip(self.trainable_parameters(), self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = effective_gradient(grad, data, self.weight_decay, self.maximize)
                average = state_buffer(cache["average"], key, g)
                accumulator = state_buffer(cache["accumulator"], key, g)

                # average = rho * average + (1 - rho) * g**2
                scratch = np.square(g)
                scratch *= 1 - self.rho
                average *= self.rho
                average += scratch

                # delta = sqrt((accumulator + eps) / (average + eps)) * g
                np.add(average, self.eps, out=scratch)
                delta = accumulator + self.eps
                delta /= scratch
                delta **= 0.5
                delta *= g

                # accumulator = rho * accumulator + (1 - rho) * delta**2
                np.square(delta, out=scratch)
                scratch *= 1 - self.rho
                accumulator *= self.rho
                accumulator += scratch

                delta *= self.learning_rate
                data -= delta
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"average": {}, "accumulator": {}} for _ in range(len(self.parameters))
        ]

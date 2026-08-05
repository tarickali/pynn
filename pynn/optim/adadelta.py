import numpy as np

from pynn.core import Optimizer, Tensor
from pynn.utils.tensor import get_data_and_grad

__all__ = ["Adadelta"]


class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(
        self,
        parameters: list[dict[str, Tensor]],
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
        for params, cache in zip(self.parameters, self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = -grad if self.maximize else grad
                g = g + self.weight_decay * data
                if key not in cache["average"]:
                    cache["average"][key] = np.zeros_like(g)
                    cache["accumulator"][key] = np.zeros_like(g)

                cache["average"][key] = self.rho * cache["average"][key] + (
                    1 - self.rho
                ) * (g**2)
                delta = (
                    (cache["accumulator"][key] + self.eps)
                    / (cache["average"][key] + self.eps)
                ) ** 0.5 * g
                cache["accumulator"][key] = self.rho * cache["accumulator"][key] + (
                    1 - self.rho
                ) * (delta**2)

                param.data = param.data - self.learning_rate * delta
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"average": {}, "accumulator": {}} for _ in range(len(self.parameters))
        ]

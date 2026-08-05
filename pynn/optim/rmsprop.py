import numpy as np

from pynn.core import Optimizer, Tensor
from pynn.utils.tensor import get_data_and_grad

__all__ = ["RMSprop"]


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        parameters: list[dict[str, Tensor]],
        learning_rate: float = 0.01,
        lr: float | None = None,
        alpha: float = 0.99,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        centered: bool = False,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        learning_rate = lr if lr is not None else learning_rate
        super().__init__(parameters, learning_rate)
        self.alpha = alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.centered = centered
        self.eps = eps
        self.maximize = maximize
        self.cache: list[dict[str, dict[str, np.ndarray]]] = [
            {"square_average": {}, "buffer": {}, "g_av": {}}
            for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        for params, cache in zip(self.parameters, self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = -grad if self.maximize else grad
                g = g + self.weight_decay * data
                if key not in cache["square_average"]:
                    cache["square_average"][key] = np.zeros_like(g)
                    cache["buffer"][key] = np.zeros_like(g)
                    cache["g_av"][key] = np.zeros_like(g)
                cache["square_average"][key] = self.alpha * cache["square_average"][
                    key
                ] + (1 - self.alpha) * (g**2)
                v_hat = cache["square_average"][key]
                if self.centered:
                    cache["g_av"][key] = (
                        cache["g_av"][key] * self.alpha + (1 - self.alpha) * g
                    )
                    v_hat = v_hat - (cache["g_av"][key] ** 2)
                if self.momentum > 0:
                    cache["buffer"][key] = self.momentum * cache["buffer"][key] + g / (
                        np.sqrt(v_hat) + self.eps
                    )
                    param.data = param.data - self.learning_rate * cache["buffer"][key]
                else:
                    param.data = param.data - self.learning_rate * g / (
                        np.sqrt(v_hat) + self.eps
                    )
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"square_average": {}, "buffer": {}, "g_av": {}}
            for _ in range(len(self.parameters))
        ]

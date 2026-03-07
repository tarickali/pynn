import numpy as np

from pynn.core import Tensor, Optimizer
from pynn.utils.tensor import get_data_and_grad

__all__ = ["SGD"]


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(
        self,
        parameters: list[dict[str, Tensor]],
        learning_rate: float = 0.01,
        lr: float | None = None,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        learning_rate = lr if lr is not None else learning_rate
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.cache: list[dict[str, dict[str, np.ndarray]]] = [
            {"velocity": {}} for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        L = len(self.parameters)
        t = self.time + 1
        for l in range(L):
            params = self.parameters[l]
            cache = self.cache[l]
            for key, param in params.items():
                data, g = get_data_and_grad(param)
                g = g + self.weight_decay * data
                if self.momentum != 0.0:
                    if t == 1:
                        cache["velocity"][key] = np.copy(g)
                    else:
                        cache["velocity"][key] = (
                            self.momentum * cache["velocity"][key] + self.dampening * g
                        )
                    if self.nesterov:
                        g = g + self.momentum * cache["velocity"][key]
                    else:
                        g = cache["velocity"][key]
                if self.maximize:
                    param.data = param.data + self.learning_rate * g
                else:
                    param.data = param.data - self.learning_rate * g
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [{"velocity": {}} for _ in range(len(self.parameters))]

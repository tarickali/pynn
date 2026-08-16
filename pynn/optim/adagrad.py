import numpy as np

from pynn.core import Optimizer
from pynn.core.optimizer import ParameterSource, effective_gradient
from pynn.utils.tensor import get_data_and_grad

__all__ = ["Adagrad"]


class Adagrad(Optimizer):
    """Adagrad optimizer.

    Written in place — see `pynn.core.optimizer` for what that costs and why.
    """

    def __init__(
        self,
        parameters: ParameterSource,
        learning_rate: float = 0.01,
        lr: float | None = None,
        learning_rate_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        learning_rate = lr if lr is not None else learning_rate
        super().__init__(parameters, learning_rate)
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.maximize = maximize
        self.cache: list[dict[str, dict[str, np.ndarray]]] = [
            {"sum": {}} for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        t = self.time + 1
        for params, cache in zip(self.trainable_parameters(), self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = effective_gradient(grad, data, self.weight_decay, self.maximize)
                lr = self.learning_rate / (1 + (t - 1) * self.learning_rate_decay)

                total = cache["sum"].get(key)
                if total is None:
                    # `data` rather than `g`, since the accumulator's starting value is
                    # a hyperparameter and the gradient only supplies the shape.
                    total = np.full_like(data, self.initial_accumulator_value)
                    cache["sum"][key] = total
                total += g**2

                denominator = np.sqrt(total)
                denominator += self.eps
                step = lr * g
                step /= denominator
                data -= step
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [{"sum": {}} for _ in range(len(self.parameters))]

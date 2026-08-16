import numpy as np

from pynn.core import Optimizer
from pynn.core.optimizer import ParameterSource, effective_gradient, state_buffer
from pynn.utils.tensor import get_data_and_grad

__all__ = ["RMSprop"]


class RMSprop(Optimizer):
    """RMSprop optimizer.

    Written in place — see `pynn.core.optimizer` for what that costs and why.
    """

    def __init__(
        self,
        parameters: ParameterSource,
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
        for params, cache in zip(self.trainable_parameters(), self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = effective_gradient(grad, data, self.weight_decay, self.maximize)
                square_average = state_buffer(cache["square_average"], key, g)

                # square_average = alpha * square_average + (1 - alpha) * g**2
                scratch = np.square(g)
                scratch *= 1 - self.alpha
                square_average *= self.alpha
                square_average += scratch

                v_hat = square_average
                if self.centered:
                    # Subtracting the squared running mean turns the second moment into
                    # a variance estimate, which is the whole of "centered".
                    g_av = state_buffer(cache["g_av"], key, g)
                    g_av *= self.alpha
                    g_av += (1 - self.alpha) * g
                    # Into `scratch`, so that `square_average` is left alone: `v_hat`
                    # aliases it in the uncentered branch below.
                    np.square(g_av, out=scratch)
                    np.subtract(square_average, scratch, out=scratch)
                    v_hat = scratch

                denominator = np.sqrt(v_hat)
                denominator += self.eps
                if self.momentum > 0:
                    buffer = state_buffer(cache["buffer"], key, g)
                    buffer *= self.momentum
                    buffer += g / denominator
                    step = self.learning_rate * buffer
                else:
                    step = self.learning_rate * g
                    step /= denominator
                data -= step
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"square_average": {}, "buffer": {}, "g_av": {}}
            for _ in range(len(self.parameters))
        ]

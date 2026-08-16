import numpy as np

from pynn.core import Optimizer
from pynn.core.optimizer import ParameterSource, effective_gradient, state_buffer
from pynn.utils.tensor import get_data_and_grad

__all__ = ["NAdam"]


class NAdam(Optimizer):
    """Adam with Nesterov momentum.

    Adam's bias-corrected first moment is an average of past gradients, so the step it
    takes is aimed at where the parameter has been. Nesterov's correction aims it at
    where the parameter is going, by folding the *next* step's momentum coefficient
    into the current estimate::

        mhat = mu_next * m / (1 - prod(mu_1..mu_next))
             + (1 - mu) * g / (1 - prod(mu_1..mu))

    Two things fall out of that expression and are worth reading carefully, because
    both are easy to get subtly wrong in a way that still trains.

    First, `mu` is not a constant. It follows a warmup schedule of its own,
    `beta_1 * (1 - 0.5 * 0.96 ** (t * momentum_decay))`, which starts near
    `beta_1 / 2` and approaches `beta_1`. Early steps therefore lean on the current
    gradient rather than on a momentum buffer that has barely any history in it.

    Second, the denominators are running *products* of every `mu` so far, not powers of
    one of them — which is what makes them the correct bias correction for a
    coefficient that changes each step. That product is optimizer state, carried across
    steps and reset by `reset`.

    Written in place — see `pynn.core.optimizer` for what that costs and why.

    Parameters
    ----------
    parameters : ParameterSource
        A Module, or explicit parameter groups.
    learning_rate : float, default 0.002
        PyTorch's default, and higher than Adam's 0.001.
    lr : float | None
        Alias for `learning_rate`.
    beta_1, beta_2 : float
        Exponential decay rates for the first and second moment estimates.
    eps : float, default 1e-8
        Added to the denominator for numerical stability.
    weight_decay : float, default 0.0
        Coupled by default, i.e. folded into the gradient the way `Adam` does it.
    momentum_decay : float, default 0.004
        Rate of the `mu` warmup above. Larger reaches `beta_1` sooner.
    decoupled_weight_decay : bool, default False
        Apply the decay to the parameter rather than to the gradient, as `AdamW` does.
        The distinction is the same one and matters for the same reason: coupled decay
        goes through the `1 / sqrt(v)` rescaling, so how hard a parameter is pulled
        toward zero ends up depending on its gradient history.
    maximize : bool, default False
        Ascend the objective rather than descend it.
    """

    def __init__(
        self,
        parameters: ParameterSource,
        learning_rate: float = 0.002,
        lr: float | None = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
        decoupled_weight_decay: bool = False,
        maximize: bool = False,
    ) -> None:
        learning_rate = lr if lr is not None else learning_rate
        super().__init__(parameters, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.maximize = maximize
        #: Running product of every `mu` so far. A scalar, and shared by every
        #: parameter, since it is a function of the step count alone.
        self.mu_product = 1.0
        self.cache: list[dict[str, dict[str, np.ndarray]]] = [
            {"momentum": {}, "velocity": {}} for _ in range(len(self.parameters))
        ]

    def _mu(self, step: int) -> float:
        """The momentum coefficient at `step`, one-based."""
        return self.beta_1 * (1.0 - 0.5 * 0.96 ** (step * self.momentum_decay))

    def update(self) -> None:
        t = self.time + 1
        mu = self._mu(t)
        mu_next = self._mu(t + 1)
        # Advanced once per step rather than once per parameter: it is a property of
        # the schedule, not of any weight.
        self.mu_product *= mu
        mu_product = self.mu_product
        mu_product_next = mu_product * mu_next

        coupled = 0.0 if self.decoupled_weight_decay else self.weight_decay
        for params, cache in zip(self.trainable_parameters(), self.cache, strict=True):
            for key, param in params.items():
                data, grad = get_data_and_grad(param)
                g = effective_gradient(grad, data, coupled, self.maximize)
                momentum = state_buffer(cache["momentum"], key, g)
                velocity = state_buffer(cache["velocity"], key, g)

                # momentum = beta_1 * momentum + (1 - beta_1) * g
                scratch = (1 - self.beta_1) * g
                momentum *= self.beta_1
                momentum += scratch

                # velocity = beta_2 * velocity + (1 - beta_2) * g**2
                np.square(g, out=scratch)
                scratch *= 1 - self.beta_2
                velocity *= self.beta_2
                velocity += scratch

                # mhat = mu_next * m / (1 - mu_product_next)
                #      + (1 - mu) * g / (1 - mu_product)
                mhat = momentum * (mu_next / (1 - mu_product_next))
                np.multiply(g, (1 - mu) / (1 - mu_product), out=scratch)
                mhat += scratch

                vhat = velocity / (1 - self.beta_2**t)
                np.sqrt(vhat, out=vhat)
                vhat += self.eps
                mhat *= self.learning_rate
                mhat /= vhat

                if self.decoupled_weight_decay and self.weight_decay:
                    np.multiply(
                        data, self.learning_rate * self.weight_decay, out=scratch
                    )
                    data -= scratch
                data -= mhat
        self.increment()

    def reset(self) -> None:
        super().reset()
        self.mu_product = 1.0
        self.cache = [
            {"momentum": {}, "velocity": {}} for _ in range(len(self.parameters))
        ]

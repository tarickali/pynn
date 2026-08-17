from typing import Any

import numpy as np

from pynn.core import Activation, Module, Tensor
from pynn.functional.activations import (
    affine,
    elu,
    gelu,
    identity,
    log_softmax,
    prelu,
    relu,
    selu,
    sigmoid,
    silu,
    softmax,
    softplus,
    tanh,
)

__all__ = [
    "ELU",
    "GELU",
    "SELU",
    "Affine",
    "Identity",
    "LogSoftmax",
    "PReLU",
    "ReLU",
    "SiLU",
    "Sigmoid",
    "SoftPlus",
    "Softmax",
    "Tanh",
]


class Identity(Activation):
    """Returns its input unchanged: `f(x) = x`.

    The default activation for every layer, so it is on the tape of nearly every
    model — `docs/DESIGN.md` §2 has the node it leaves behind and why it is there.
    For the `Module` form, which is what a container holds, see `pynn.nn.Identity`.

    """

    def compute(self, x: Tensor) -> Tensor:
        return identity(x)


class Affine(Activation):
    """A straight line: `f(x) = slope * x + intercept`.

    Parameters
    ----------
    slope : float, default 1.0
    intercept : float, default 0.0

    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def compute(self, x: Tensor) -> Tensor:
        return affine(x, self.slope, self.intercept)


class ReLU(Activation):
    """`f(x) = x` above zero, `alpha * x` below it.

    `alpha=0.0` is the plain rectifier; any positive value makes it a leaky ReLU,
    which keeps a gradient flowing through units the plain form would switch off
    permanently.

    Parameters
    ----------
    alpha : float, default 0.0
        Slope on the negative side.

    """

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha

    def compute(self, x: Tensor) -> Tensor:
        return relu(x, self.alpha)


class Sigmoid(Activation):
    """`f(x) = 1 / (1 + exp(-x))`, squashing the line onto (0, 1).

    Evaluated by the branch-on-sign kernel in `pynn.core.numeric`, since the naive
    form overflows for large negative `x`.

    """

    def compute(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Activation):
    """`f(x) = tanh(x)`, squashing the line onto (-1, 1)."""

    def compute(self, x: Tensor) -> Tensor:
        return tanh(x)


class ELU(Activation):
    """`f(x) = x` above zero, `alpha * (exp(x) - 1)` below it.

    Saturates to `-alpha` rather than to zero, so the mean activation sits nearer
    zero than a ReLU's does. Computed with `expm1` on a clamped input, or the two
    terms cancel catastrophically near zero.

    Parameters
    ----------
    alpha : float, default 1.0
        The negative saturation value.

    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def compute(self, x: Tensor) -> Tensor:
        return elu(x, self.alpha)


class SELU(Activation):
    """ELU with the two constants that make it self-normalizing.

    `f(x) = SCALE * x` above zero and `SCALE * ALPHA * (exp(x) - 1)` below it, with
    `SCALE = 1.0507009873554805` and `ALPHA = 1.6732632423543772`. The constants are
    not free parameters: they are the fixed point at which activations keep unit mean
    and variance from layer to layer, which is the whole claim of the paper. Changing
    either one gives an ELU with unusual constants, not a SELU.

    """

    def compute(self, x: Tensor) -> Tensor:
        return selu(x)


class SoftPlus(Activation):
    """`f(x) = log(1 + exp(x))`, a smooth ReLU.

    Evaluated as `logaddexp(0, x)`, since the literal form returns `inf` past
    `x ~ 709`, where `exp` leaves float64 range.

    """

    def compute(self, x: Tensor) -> Tensor:
        return softplus(x)


class Softmax(Activation):
    """`f(x) = exp(x) / sum(exp(x))` over the given axis.
    Default axis=-1 (last axis, e.g. class logits).

    The gradient uses the full softmax Jacobian: dL/dz = s * (g - sum(s*g)),
    so it is correct for any axis and any downstream loss. When the next layer
    is Categorical Cross-Entropy, the upstream g has the form (s - y); the
    Jacobian applied to that still yields the correct dL/dz for any axis.

    Parameters
    ----------
    axis : int, default -1
        Axis over which to apply softmax (e.g. -1 for class dimension).
    """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def compute(self, x: Tensor) -> Tensor:
        return softmax(x, axis=self.axis)


class GELU(Activation):
    """GELU Activation

    Computes `f(x) = x * P(Z <= x)` for a standard normal `Z` — a smooth gate on how
    far through the distribution the input is, rather than ReLU's hard gate on its sign.

    Parameters
    ----------
    approximate : str, default "tanh"
        "tanh" for the original paper's approximation, "none" for the exact Gaussian
        CDF. See `pynn.functional.gelu`.

    """

    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def compute(self, x: Tensor) -> Tensor:
        return gelu(x, self.approximate)


class SiLU(Activation):
    """SiLU Activation, also called Swish

    Computes `f(x) = x * sigmoid(x)`.

    """

    def compute(self, x: Tensor) -> Tensor:
        return silu(x)


class LogSoftmax(Activation):
    """LogSoftmax Activation

    Computes `f(x) = x - logsumexp(x)` over the given axis, which is `log(softmax(x))`
    without ever forming the softmax — the literal composition underflows to `-inf` for
    any class the softmax rounds to zero.

    Parameters
    ----------
    axis : int, default -1
        Axis to normalize over.

    """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def compute(self, x: Tensor) -> Tensor:
        return log_softmax(x, axis=self.axis)


class PReLU(Module):
    """Parametric ReLU: a ReLU whose negative slope is learned.

    A `Module` rather than an `Activation` because it owns a parameter, and a parameter
    outside the module tree is one no optimizer would ever step. Assigning it inside a
    layer registers it like any other child, so `Linear(4, 3, activation="prelu")` puts
    its slope at `act_fn.alpha` and trains it along with the weights.

    Parameters
    ----------
    num_parameters : int, default 1
        One slope shared across channels, or one per channel — meaning axis 1 of the
        input when it has one.
    init : float, default 0.25
        Starting value for every slope, matching PyTorch.

    """

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, name: str = "PReLU"
    ) -> None:
        super().__init__()
        # Stored as `num_slopes`, not `num_parameters`: the keyword matches PyTorch,
        # but an attribute of that name would shadow `Module.num_parameters()` on every
        # instance, and the method would then raise "int object is not callable".
        self.num_slopes = num_parameters
        self.init = init
        self.name = name
        self.register_parameter("alpha", np.full(num_parameters, init))
        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        return prelu(X, self.parameters["alpha"])

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"num_parameters": self.num_slopes, "init": self.init}

import numpy as np

from pynn.core import Tensor
from pynn.core.numeric import erf, stable_sigmoid
from pynn.core.types import Array, Number
from pynn.core.utils import unbroadcast

TensorLike = Tensor | Array | Number

__all__ = [
    "affine",
    "elu",
    "gelu",
    "identity",
    "log_softmax",
    "prelu",
    "relu",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "tanh",
]


def identity(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = array.copy()
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += output.grad

    output.forward = "identity"
    output.reverse = reverse

    return output


def affine(x: TensorLike, slope: float = 1.0, intercept: float = 0.0) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = slope * array + intercept
    output = Tensor(data)

    output.add_children((x,))

    def reverse():
        x.grad += slope * output.grad

    output.forward = "affine"
    output.reverse = reverse

    return output


def relu(x: TensorLike, alpha: float = 0.0) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.maximum(0, array) + alpha * np.minimum(0, array)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        # Not differentiable at 0; the subgradient alpha is taken there, matching
        # the forward pass where max(0, 0) + alpha * min(0, 0) == 0.
        x.grad += np.where(array > 0, 1.0, alpha) * output.grad

    output.forward = "relu"
    output.reverse = reverse

    return output


def sigmoid(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    data = stable_sigmoid(x.data)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += data * (1 - data) * output.grad

    output.forward = "sigmoid"
    output.reverse = reverse

    return output


def tanh(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.tanh(array)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += (1 - data**2) * output.grad

    output.forward = "tanh"
    output.reverse = reverse

    return output


def elu(x: TensorLike, alpha: float = 1.0) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.where(array >= 0, array, alpha * (np.expm1(np.minimum(array, 0.0))))
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = np.where(array >= 0, 1.0, alpha * np.exp(np.minimum(array, 0.0)))
        x.grad += grad * output.grad

    output.forward = "elu"
    output.reverse = reverse

    return output


def selu(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    array = x.data
    negative = np.minimum(array, 0.0)
    data = scale * np.where(array >= 0, array, alpha * np.expm1(negative))
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = scale * np.where(array >= 0, 1.0, alpha * np.exp(negative))
        x.grad += grad * output.grad

    output.forward = "selu"
    output.reverse = reverse

    return output


def softplus(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    # logaddexp(0, x) == log(1 + exp(x)) without the overflow of computing exp(x)
    # first, which returns inf for x greater than about 709.
    data = np.logaddexp(0.0, array)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += stable_sigmoid(array) * output.grad

    output.forward = "softplus"
    output.reverse = reverse

    return output


def softmax(x: TensorLike, axis: int = -1) -> Tensor:
    """Softmax over the given axis. Default axis=-1 (last axis, e.g. classes).

    Gradient: dL/dz = s * (g - sum(s*g, axis=axis, keepdims=True))
    where s = softmax(z), g = upstream dL/ds.
    """
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    e = np.exp(array - np.max(array, axis=axis, keepdims=True))
    data = e / np.sum(e, axis=axis, keepdims=True)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        s = output.data
        g = output.grad
        # Jacobian-vector product: dL/dz = s * (g - sum(s*g, axis, keepdims))
        x.grad += s * (g - np.sum(s * g, axis=axis, keepdims=True))

    output.forward = "softmax"
    output.reverse = reverse

    return output


#: sqrt(2/pi), the constant in GELU's tanh approximation.
_GELU_TANH_SCALE = 0.7978845608028654
#: The cubic coefficient from the original GELU paper's approximation.
_GELU_TANH_CUBIC = 0.044715
#: 1/sqrt(2*pi), the standard normal density's normalizing constant.
_NORMAL_SCALE = 0.3989422804014327


def gelu(x: TensorLike, approximate: str = "tanh") -> Tensor:
    """Gaussian Error Linear Unit: ``x * P(Z <= x)`` for a standard normal ``Z``.

    Where ReLU gates on the sign of its input, GELU weights the input by how far
    through the normal distribution it is — a smooth gate rather than a hard one, and
    the reason it is the activation in most transformers.

    Parameters
    ----------
    x : TensorLike
        Input of any shape.
    approximate : str, default "tanh"
        ``"tanh"`` uses the original paper's approximation, which is what BERT and GPT
        actually shipped and needs nothing but `np.tanh`. ``"none"`` evaluates the
        Gaussian CDF exactly through `erf`, which is correctly rounded but roughly an
        order of magnitude slower, since NumPy has no vectorized erf.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Raises
    ------
    ValueError
        If `approximate` is not "tanh" or "none".
    """
    x = x if isinstance(x, Tensor) else Tensor(x)
    if approximate not in ("tanh", "none"):
        raise ValueError(f"approximate must be 'tanh' or 'none', got {approximate!r}")

    array = x.data
    if approximate == "tanh":
        inner = _GELU_TANH_SCALE * (array + _GELU_TANH_CUBIC * array**3)
        tanh_inner = np.tanh(inner)
        data = 0.5 * array * (1.0 + tanh_inner)
    else:
        cdf = 0.5 * (1.0 + erf(array / np.sqrt(2.0)))
        data = array * cdf

    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        if approximate == "tanh":
            # d/dx [0.5 x (1 + tanh u)] = 0.5 (1 + tanh u) + 0.5 x sech^2(u) du/dx
            d_inner = _GELU_TANH_SCALE * (1.0 + 3.0 * _GELU_TANH_CUBIC * array**2)
            grad = (
                0.5 * (1.0 + tanh_inner) + 0.5 * array * (1.0 - tanh_inner**2) * d_inner
            )
        else:
            # d/dx [x Phi(x)] = Phi(x) + x phi(x)
            density = _NORMAL_SCALE * np.exp(-0.5 * array**2)
            grad = cdf + array * density
        x.grad += grad * output.grad

    output.forward = "gelu"
    output.reverse = reverse

    return output


def silu(x: TensorLike) -> Tensor:
    """Sigmoid Linear Unit, also called Swish: ``x * sigmoid(x)``.

    Like GELU it is a smooth gate, and unlike ReLU it is non-monotonic — slightly
    negative just below zero, which is where the small gradient it keeps alive there
    comes from.

    Parameters
    ----------
    x : TensorLike
        Input of any shape.

    Returns
    -------
    Tensor
        Same shape as ``x``.
    """
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    gate = stable_sigmoid(array)
    data = array * gate
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        # d/dx [x s(x)] = s(x) + x s(x)(1 - s(x))
        x.grad += (gate + array * gate * (1.0 - gate)) * output.grad

    output.forward = "silu"
    output.reverse = reverse

    return output


def log_softmax(x: TensorLike, axis: int = -1) -> Tensor:
    """Log of the softmax, computed without forming the softmax first.

    ``log(softmax(z))`` evaluated literally underflows to ``-inf`` for any class the
    softmax rounds to zero, which is exactly the class a cross-entropy loss cares most
    about. Subtracting the log-sum-exp instead keeps every value finite.

    Parameters
    ----------
    x : TensorLike
        Input of any shape.
    axis : int, default -1
        Axis to normalize over.

    Returns
    -------
    Tensor
        Same shape as ``x``.
    """
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    shifted = array - np.max(array, axis=axis, keepdims=True)
    log_sum = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    data = shifted - log_sum
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        # y = z - logsumexp(z), so dy_i/dz_j = delta_ij - softmax(z)_j, and the
        # Jacobian-vector product collapses to g - softmax * sum(g).
        probabilities = np.exp(data)
        x.grad += output.grad - probabilities * np.sum(
            output.grad, axis=axis, keepdims=True
        )

    output.forward = "log_softmax"
    output.reverse = reverse

    return output


def _prelu_broadcast_shape(x: Tensor, alpha: Tensor) -> tuple[int, ...]:
    """The shape `alpha` takes to line up with `x`'s channel axis.

    A single slope broadcasts against anything. A per-channel slope has to land on
    axis 1 for ``(batch, channels, ...)`` input, which for 4-D input means ``(C, 1, 1)``
    rather than the ``(C,)`` that would silently align with the last axis instead.
    """
    if alpha.size == 1:
        return (1,) * x.ndim
    if x.ndim <= 1:
        return (alpha.size,)
    return (1, alpha.size) + (1,) * (x.ndim - 2)


def prelu(x: TensorLike, alpha: Tensor) -> Tensor:
    """Parametric ReLU: ``x`` where positive, ``alpha * x`` where negative.

    Unlike every other activation here, the slope is *learned*, so `alpha` is a Tensor
    on the tape rather than a Python float and receives a gradient of its own.

    Parameters
    ----------
    x : TensorLike
        Input of any shape.
    alpha : Tensor
        Negative slope. Either a single element, shared across channels, or one per
        channel — meaning axis 1 of ``x`` when it has one.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Raises
    ------
    ValueError
        If `alpha` has neither one element nor one per channel.
    """
    x = x if isinstance(x, Tensor) else Tensor(x)

    channels = x.shape[1] if x.ndim > 1 else (x.shape[0] if x.ndim else 1)
    if alpha.size not in (1, channels):
        raise ValueError(
            f"prelu expects alpha to hold 1 or {channels} elements for an input of "
            f"shape {x.shape}, got {alpha.size}"
        )

    array = x.data
    slope = alpha.data.reshape(_prelu_broadcast_shape(x, alpha))
    negative = array < 0
    data = np.where(negative, slope * array, array)

    output = Tensor(data)
    output.add_children((x, alpha))

    def reverse():
        x.grad += np.where(negative, slope, 1.0) * output.grad
        # dy/dalpha is x where x is negative and 0 elsewhere, summed back down over
        # every axis alpha was broadcast along.
        alpha.grad += unbroadcast(
            np.where(negative, array, 0.0) * output.grad, slope.shape
        ).reshape(alpha.shape)

    output.forward = "prelu"
    output.reverse = reverse

    return output

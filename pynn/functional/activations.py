import numpy as np
from pynn.core.types import Array, Number
from pynn.core import Tensor
from pynn.core.numeric import stable_sigmoid

TensorLike = Tensor | Array | Number

__all__ = [
    "identity",
    "affine",
    "relu",
    "sigmoid",
    "tanh",
    "elu",
    "selu",
    "softplus",
    "softmax",
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

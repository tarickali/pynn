import numpy as np
from pynn.core.types import Array, Number
from pynn.core import Tensor

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
        x.grad += np.ones_like(array) * np.asarray(output.grad)

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
        grad = np.full_like(array, slope)
        x.grad = grad * output.grad

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
        grad = alpha + (1 - alpha) * np.heaviside(array, 0.0, dtype=array.dtype)
        x.grad = grad * output.grad

    output.forward = "relu"
    output.reverse = reverse

    return output


def sigmoid(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = 1 / (1 + np.exp(-array))
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = data * (1 - data)
        x.grad = grad * output.grad

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
        grad = 1 - data**2
        x.grad = grad * output.grad

    output.forward = "tanh"
    output.reverse = reverse

    return output


def elu(x: TensorLike, alpha: float) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.where(array >= 0, array, alpha * (np.exp(array) - 1))
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = np.where(array >= 0, np.ones_like(array), alpha * np.exp(array))
        x.grad = grad * output.grad

    output.forward = "elu"
    output.reverse = reverse

    return output


def selu(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    array = x.data
    data = scale * (np.maximum(0, array) + np.minimum(0, alpha * (np.exp(array) - 1)))
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = np.where(
            array >= 0, scale * np.ones_like(array), alpha * scale * np.exp(array)
        )
        x.grad = grad * output.grad

    output.forward = "selu"
    output.reverse = reverse

    return output


def softplus(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    e = np.exp(array)
    data = np.log(1 + e)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = e / (1 + e)
        x.grad = grad * output.grad

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
        # Jacobian: dL/dz = s * (g - sum(s*g, axis, keepdims))
        x.grad = s * (g - np.sum(s * g, axis=axis, keepdims=True))

    output.forward = "softmax"
    output.reverse = reverse

    return output

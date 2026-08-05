import numpy as np

from .tensor import Tensor
from .types import Array, Number
from .constants import EPSILON


__all__ = ["abs", "sum", "mean", "exp", "log"]


TensorLike = Tensor | Array | Number


def abs(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.abs(array)

    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += np.sign(array) * output.grad

    output.forward = "abs"
    output.reverse = reverse

    return output


def sum(x: TensorLike, axis: int | tuple[int, ...] | None = None) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.sum(array, axis=axis, keepdims=True)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        # keepdims=True above leaves the reduced axes as length 1, so the incoming
        # gradient broadcasts back over them without any reshaping here.
        x.grad += np.broadcast_to(output.grad, array.shape)

    output.forward = "sum"
    output.reverse = reverse

    return output


def mean(x: TensorLike, axis: int | tuple[int, ...] | None = None) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)
    axis = axis if isinstance(axis, tuple) or axis is None else (axis,)

    array = x.data
    data = np.mean(array, axis=axis, keepdims=True)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        count = (
            np.prod([array.shape[i] for i in axis]) if axis is not None else array.size
        )
        x.grad += np.broadcast_to(output.grad, array.shape) / count

    output.forward = "mean"
    output.reverse = reverse

    return output


def exp(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.exp(array)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += data * output.grad

    output.forward = "exp"
    output.reverse = reverse

    return output


def log(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    # Clamp rather than offsetting the input by EPSILON: an offset biases the result
    # everywhere, and at EPSILON ~ 2.2e-16 it is far too small to tame log(0) anyway.
    safe = np.maximum(array, EPSILON)
    data = np.log(safe)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += output.grad / safe

    output.forward = "log"
    output.reverse = reverse

    return output

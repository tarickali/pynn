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
        grad = np.sign(array)
        x.grad = grad * output.grad

    output.forward = "abs"
    output.reverse = reverse

    return output


def sum(x: TensorLike, axis: int | tuple[int] = None) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    # TODO NOTE Can use the following to squeeze the array if it is a scalar:
    # `keepdims=False if axis is None or len(axis) == len(x.arr.shape) else True`
    data = np.sum(array, axis=axis, keepdims=True)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = np.ones_like(array)
        x.grad = grad * output.grad

    output.forward = "sum"
    output.reverse = reverse

    return output


def mean(x: TensorLike, axis: int | tuple[int] = None) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)
    axis = axis if isinstance(axis, tuple) or axis is None else (axis,)

    array = x.data
    # TODO NOTE Can use the following to squeeze the array if it is a scalar:
    # `keepdims=False if axis is None or len(axis) == len(x.arr.shape) else True`
    data = np.mean(array, axis=axis, keepdims=True)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        norm = (
            np.prod([np.size(array, axis=i) for i in axis])
            if axis is not None
            else array.size
        )
        grad = np.full_like(array, 1.0 / norm)
        x.grad = grad * output.grad

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
        grad = data
        x.grad = grad * output.grad

    output.forward = "exp"
    output.reverse = reverse

    return output


def log(x: TensorLike) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)

    array = x.data
    data = np.log(array + EPSILON)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = 1 / (array + EPSILON)
        x.grad = grad * output.grad

    output.forward = "log"
    output.reverse = reverse

    return output

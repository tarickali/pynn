from __future__ import annotations
from typing import Any
import numpy as np
from numba import njit

from pynn.core import Array, Number, ArrayLike, DataType, Shape
from pynn.core.primitives import *
from pynn.core.utils import expand_array, shrink_array

__all__ = ["Tensor"]

TensorLike = ArrayLike


class Tensor:
    def __init__(self, data: Tensor | TensorLike, dtype: DataType = np.float64) -> None:
        data = data.data if isinstance(data, Tensor) else data
        self.data: Array = np.array(data, dtype=dtype)
        self.grad: Array = np.zeros_like(self.data, dtype=np.float64)

        self.forward: str = None
        self.reverse = lambda: None

        self.children: tuple[Tensor] = ()
        self.trainable = True

    # TODO: Should I have cast be inplace?
    def cast(self, dtype: DataType) -> None:
        if dtype != self.dtype:
            self.data = self.data.astype(dtype)

    def numpy(self) -> Array:
        return self.data

    def item(self) -> Number:
        return self.data.item()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.grad)

    def add_children(self, tensors: tuple[Tensor, ...]) -> None:
        self.children += tensors

    def backward(self) -> None:
        order = list[Tensor]()
        visited = set[Tensor]()

        def build(x: Tensor) -> None:
            if x not in visited:
                visited.add(x)
                for child in x.children:
                    build(child)
                order.append(x)

        build(self)

        self.grad = np.ones_like(self.data)
        for x in reversed(order):
            x.reverse()

    def transpose(self) -> Tensor:
        return Tensor(transpose(self.data))

    # ------------------------------------------------------------------------ #
    # Getter and Setter
    # ------------------------------------------------------------------------ #
    @njit
    def __getitem__(self, key: int | tuple[int] | slice) -> Array | Number:
        return self.data[key]

    @njit
    def __setitem__(self, key: int | tuple[int] | slice, value: ArrayLike) -> None:
        self.data[key] = value

    # ------------------------------------------------------------------------ #
    # Binary Operations
    # ------------------------------------------------------------------------ #
    def __add__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(add(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            self.grad = expand_array(self.grad, output.grad.shape)
            other.grad = expand_array(other.grad, output.grad.shape)
            self.grad += output.grad
            other.grad += output.grad
            self.grad = shrink_array(self.grad, self.data.shape)
            other.grad = shrink_array(other.grad, other.data.shape)

        output.forward = "add"
        output.reverse = reverse

        return output

    def __sub__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(subtract(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            self.grad = expand_array(self.grad, output.grad.shape)
            other.grad = expand_array(other.grad, output.grad.shape)
            self.grad += output.grad
            other.grad += -output.grad
            self.grad = shrink_array(self.grad, self.data.shape)
            other.grad = shrink_array(other.grad, other.data.shape)

        output.forward = "sub"
        output.reverse = reverse

        return output

    def __mul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(multiply(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            self.grad = expand_array(self.grad, output.grad.shape)
            other.grad = expand_array(other.grad, output.grad.shape)
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
            self.grad = shrink_array(self.grad, self.data.shape)
            other.grad = shrink_array(other.grad, other.data.shape)

        output.forward = "mul"
        output.reverse = reverse

        return output

    def __matmul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(matrix_multiply(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            self.grad = expand_array(self.grad, output.grad.shape)
            other.grad = expand_array(other.grad, output.grad.shape)
            self.grad += output.grad @ other.data.T
            other.grad += self.data.T @ output.grad
            self.grad = shrink_array(self.grad, self.data.shape)
            other.grad = shrink_array(other.grad, other.data.shape)

        output.forward = "matmul"
        output.reverse = reverse

        return output

    def __truediv__(self, other: Tensor | TensorLike) -> Tensor:
        return self * other**-1

    def __radd__(self, other: Tensor | TensorLike) -> Tensor:
        return self + other

    def __rsub__(self, other: Tensor | TensorLike) -> Tensor:
        return -self + other

    def __rmul__(self, other: Tensor | TensorLike) -> Tensor:
        return self * other

    def __rtruediv__(self, other: Tensor | TensorLike) -> Tensor:
        return self**-1 * other

    # ------------------------------------------------------------------------ #
    # Unary Operations
    # ------------------------------------------------------------------------ #
    def __pow__(self, other: Number) -> Tensor:
        if not isinstance(other, Number):
            raise ValueError(f"Cannot perform operation on {type(other)}")

        output = Tensor(power(self.data, other))
        output.add_children((self,))

        def reverse():
            grad = Tensor(other * self.data.array ** (other - 1))
            self.grad += grad * output.grad

        output.forward = "pow"
        output.reverse = reverse

        return output

    def __neg__(self) -> Tensor:
        output = Tensor(negate(self.data))
        output.add_children((self,))

        def reverse():
            self.grad += -output.grad

        output.forward = "neg"
        output.reverse = reverse

        return output

    # ------------------------------------------------------------------------ #
    # Comparison Operations
    # ------------------------------------------------------------------------ #
    def __eq__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=equal(self.data, other.data))

    def __ne__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=not_equal(self.data, other.data))

    def __ge__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=greater_than_equal(self.data, other.data))

    def __gt__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=greater_than(self.data, other.data))

    def __le__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=less_than_equal(self.data, other.data))

    def __lt__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=less_than(self.data, other.data))

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, dtype={self.dtype}, shape={self.shape})"

    @property
    def T(self) -> Tensor:
        return self.transpose()

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> DataType:
        return self.data.dtype


def convert_tensor_input(value: Any) -> Tensor:
    if not isinstance(value, Tensor | TensorLike):
        raise ValueError(f"Cannot perform operation on {type(value)}")
    return value if isinstance(value, Tensor) else Tensor(value)

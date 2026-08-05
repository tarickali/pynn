from __future__ import annotations

from typing import Any

import numpy as np

from pynn.core.primitives import (
    add,
    equal,
    greater_than,
    greater_than_equal,
    less_than,
    less_than_equal,
    matrix_multiply,
    multiply,
    negate,
    not_equal,
    power,
    subtract,
    true_division,
)
from pynn.core.types import Array, ArrayLike, DataType, Number, Shape
from pynn.core.utils import matrix_multiply_gradients, unbroadcast

__all__ = ["Tensor"]

TensorLike = ArrayLike


class Tensor:
    def __init__(self, data: Tensor | TensorLike, dtype: DataType = np.float64) -> None:
        data = data.data if isinstance(data, Tensor) else data
        self.data: Array = np.array(data, dtype=dtype)
        self.grad: Array = np.zeros_like(self.data, dtype=np.float64)

        self.forward: str | None = None
        self.reverse = lambda: None

        self.children: tuple[Tensor, ...] = ()
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

    def backward(self, gradient: Array | Number | None = None) -> None:
        """Accumulate gradients into every tensor this one was computed from.

        Gradients are added to any already present, so call ``zero_grad`` between
        optimization steps.

        Parameters
        ----------
        gradient : Array | Number | None
            Seed gradient of the same shape as this tensor. Required unless this
            tensor holds a single element, in which case it defaults to 1.0.

        Raises
        ------
        ValueError
            If this tensor is not a single element and no seed gradient is given,
            or if the seed gradient's shape does not match.
        """

        if gradient is None:
            if self.size != 1:
                raise ValueError(
                    "backward() on a Tensor with more than one element requires an "
                    f"explicit gradient, but this Tensor has shape {self.shape}. "
                    "Reduce it to a scalar (e.g. with pynn.core.math.sum) or pass "
                    "gradient=... to specify the seed."
                )
            seed = np.ones_like(self.data, dtype=np.float64)
        else:
            seed = np.asarray(gradient, dtype=np.float64)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"gradient shape {seed.shape} does not match Tensor shape "
                    f"{self.data.shape}"
                )

        self.grad = self.grad + seed

        for tensor in reversed(self._topological_order()):
            tensor.reverse()

    def _topological_order(self) -> list[Tensor]:
        """Tensors in this graph, children before parents.

        Iterative rather than recursive so that deep graphs (long chains of
        operations, or unrolled recurrences) do not exhaust the Python stack.
        """

        order: list[Tensor] = []
        visited: set[Tensor] = set()
        # Each frame is (tensor, index of the next child to visit).
        stack: list[tuple[Tensor, int]] = [(self, 0)]
        visited.add(self)

        while stack:
            tensor, child_index = stack.pop()
            if child_index < len(tensor.children):
                stack.append((tensor, child_index + 1))
                child = tensor.children[child_index]
                if child not in visited:
                    visited.add(child)
                    stack.append((child, 0))
            else:
                order.append(tensor)

        return order

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        output = Tensor(np.transpose(self.data, axes))
        output.add_children((self,))

        if axes is None:
            inverse: tuple[int, ...] | None = None
        else:
            inverse = tuple(int(i) for i in np.argsort(axes))

        def reverse():
            self.grad += np.transpose(output.grad, inverse)

        output.forward = "transpose"
        output.reverse = reverse

        return output

    # ------------------------------------------------------------------------ #
    # Getter and Setter
    # ------------------------------------------------------------------------ #
    def __getitem__(self, key: int | tuple[int] | slice) -> Array | Number:
        return self.data[key]

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
            self.grad += unbroadcast(output.grad, self.data.shape)
            other.grad += unbroadcast(output.grad, other.data.shape)

        output.forward = "add"
        output.reverse = reverse

        return output

    def __sub__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(subtract(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            self.grad += unbroadcast(output.grad, self.data.shape)
            other.grad -= unbroadcast(output.grad, other.data.shape)

        output.forward = "sub"
        output.reverse = reverse

        return output

    def __mul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(multiply(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            self.grad += unbroadcast(other.data * output.grad, self.data.shape)
            other.grad += unbroadcast(self.data * output.grad, other.data.shape)

        output.forward = "mul"
        output.reverse = reverse

        return output

    def __matmul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(matrix_multiply(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            left, right = matrix_multiply_gradients(output.grad, self.data, other.data)
            self.grad += left
            other.grad += right

        output.forward = "matmul"
        output.reverse = reverse

        return output

    def __truediv__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)

        output = Tensor(true_division(self.data, other.data))
        output.add_children((self, other))

        def reverse():
            self.grad += unbroadcast(output.grad / other.data, self.data.shape)
            other.grad -= unbroadcast(
                output.grad * self.data / other.data**2, other.data.shape
            )

        output.forward = "truediv"
        output.reverse = reverse

        return output

    def __radd__(self, other: Tensor | TensorLike) -> Tensor:
        return self + other

    def __rsub__(self, other: Tensor | TensorLike) -> Tensor:
        return -self + other

    def __rmul__(self, other: Tensor | TensorLike) -> Tensor:
        return self * other

    def __rtruediv__(self, other: Tensor | TensorLike) -> Tensor:
        return convert_tensor_input(other) / self

    def __rmatmul__(self, other: Tensor | TensorLike) -> Tensor:
        return convert_tensor_input(other) @ self

    # ------------------------------------------------------------------------ #
    # Unary Operations
    # ------------------------------------------------------------------------ #
    def __pow__(self, other: Number) -> Tensor:
        if not isinstance(other, Number):
            raise TypeError(f"Cannot perform operation on {type(other)}")

        output = Tensor(power(self.data, other))
        output.add_children((self,))

        def reverse():
            self.grad += other * np.power(self.data, other - 1) * output.grad

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
    #
    # These compare element-wise and return a Tensor of booleans, matching numpy and
    # PyTorch. That deliberately breaks the `object.__eq__ -> bool` contract, which is
    # what the `override` and `misc` ignores below are for: mypy is right that this is
    # a Liskov violation, and every array library makes the same trade.
    #
    # The consequence is that `if a == b:` does not mean what it looks like. Use
    # `bool((a == b).data.all())` for an all-elements-equal test. Note that identity
    # semantics are preserved separately by `__hash__`, which the backward pass relies
    # on to put Tensors in a visited set.
    # ------------------------------------------------------------------------ #
    def __eq__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[override]
        other = convert_tensor_input(other)
        return Tensor(data=equal(self.data, other.data))

    def __ne__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[override]
        other = convert_tensor_input(other)
        return Tensor(data=not_equal(self.data, other.data))

    def __ge__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
        other = convert_tensor_input(other)
        return Tensor(data=greater_than_equal(self.data, other.data))

    def __gt__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
        other = convert_tensor_input(other)
        return Tensor(data=greater_than(self.data, other.data))

    def __le__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
        other = convert_tensor_input(other)
        return Tensor(data=less_than_equal(self.data, other.data))

    def __lt__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
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
        raise TypeError(f"Cannot perform operation on {type(value)}")
    return value if isinstance(value, Tensor) else Tensor(value)

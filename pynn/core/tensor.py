from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pynn.core.grad_mode import is_grad_enabled
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


def _no_reverse() -> None:
    """The reverse pass of a leaf: nothing to propagate."""
    return


def gradient_dtype(dtype: DataType) -> DataType:
    """The dtype a gradient takes for data of the given dtype.

    A float32 parameter keeps a float32 gradient, so a half- or single-precision model
    does not silently pay for double-precision gradients. Anything that is not a
    floating type — integer data, the boolean results of a comparison — gets float64,
    since a gradient is real-valued regardless of what it is a gradient of.
    """
    return dtype if np.issubdtype(dtype, np.floating) else np.float64


class Tensor:
    #: Decline to participate in NumPy's ufunc dispatch. Without this, NumPy handles
    #: `array + tensor` itself by coercing the Tensor to a 0-d object array, so the
    #: result is an object-dtype array of Tensors and `__radd__` is never called —
    #: silently wrong rather than an error. Setting this to None makes NumPy return
    #: NotImplemented, which is what sends Python to the reflected method (NEP 13).
    __array_ufunc__ = None

    def __init__(
        self, data: Tensor | TensorLike, dtype: DataType | None = None
    ) -> None:
        """Wrap `data` as a Tensor.

        Parameters
        ----------
        data : Tensor | TensorLike
            Values to hold. A Tensor is unwrapped, and its history is not carried over.
        dtype : DataType | None
            Element type. The default preserves a floating-point array's precision
            rather than promoting it — a float32 input used to become float64
            silently — and promotes everything else (integers, Python scalars, lists)
            to float64.
        """
        data = data.data if isinstance(data, Tensor) else data
        if dtype is None:
            dtype = (
                data.dtype
                if isinstance(data, np.ndarray)
                and np.issubdtype(data.dtype, np.floating)
                else np.float64
            )
        self.data: Array = np.array(data, dtype=dtype)
        self.grad: Array = np.zeros(self.data.shape, dtype=gradient_dtype(self.dtype))

        self.forward: str | None = None
        self._reverse: Callable[[], None] = _no_reverse

        self.children: tuple[Tensor, ...] = ()
        #: Whether this Tensor is connected to the graph. False for one produced under
        #: `no_grad` or by `detach`, which is what makes `backward` on it an error
        #: rather than a silent zero.
        self.requires_grad = True
        #: Whether an optimizer may step this Tensor. Distinct from `requires_grad`:
        #: a frozen parameter still receives gradients, the optimizer just skips it.
        self.trainable = True

    # TODO: Should I have cast be inplace?
    def cast(self, dtype: DataType) -> None:
        if dtype != self.dtype:
            self.data = self.data.astype(dtype)
            self.grad = self.grad.astype(gradient_dtype(self.dtype))

    def numpy(self) -> Array:
        return self.data

    def detach(self) -> Tensor:
        """This Tensor's values, off the tape.

        The result holds a copy of the data with no children and no reverse function,
        so gradients stop here. Use it to take a value out of the graph — a running
        statistic, a target computed from a model's own output — without turning
        recording off everywhere.
        """
        detached = Tensor(self.data, dtype=self.dtype)
        detached.requires_grad = False
        detached.trainable = self.trainable
        return detached

    @property
    def reverse(self) -> Callable[[], None]:
        """Push this Tensor's gradient back to its children."""
        return self._reverse

    @reverse.setter
    def reverse(self, function: Callable[[], None]) -> None:
        # Dropped rather than stored when recording is off. The closure captures the
        # forward pass's intermediate arrays, so keeping it would hold the entire
        # graph alive behind an output that can never be differentiated — which is
        # exactly what happens when a validation loop collects its predictions.
        self._reverse = function if is_grad_enabled() else _no_reverse

    def item(self) -> Number:
        return self.data.item()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.grad)

    def add_children(self, tensors: tuple[Tensor, ...]) -> None:
        """Record the Tensors this one was computed from.

        The single gate for tape recording. With recording off the edges are dropped
        and the output becomes a leaf that does not require gradients, so no operation
        needs to check the mode itself.
        """
        if not is_grad_enabled():
            self.requires_grad = False
            return
        self.children += tensors
        self.requires_grad = any(child.requires_grad for child in tensors)

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
        RuntimeError
            If this tensor is not connected to the graph, because it was produced
            under `no_grad` or detached from it.
        """

        if not self.requires_grad:
            raise RuntimeError(
                "backward() on a Tensor that does not require gradients. It was "
                "produced under no_grad(), or detached from the graph, so there is "
                "nothing recorded to differentiate."
            )

        dtype = self.grad.dtype
        if gradient is None:
            if self.size != 1:
                raise ValueError(
                    "backward() on a Tensor with more than one element requires an "
                    f"explicit gradient, but this Tensor has shape {self.shape}. "
                    "Reduce it to a scalar (e.g. with pynn.core.math.sum) or pass "
                    "gradient=... to specify the seed."
                )
            seed = np.ones(self.data.shape, dtype=dtype)
        else:
            seed = np.asarray(gradient, dtype=dtype)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"gradient shape {seed.shape} does not match Tensor shape "
                    f"{self.data.shape}"
                )

        self.grad = self.grad + seed

        for tensor in reversed(self._topological_order()):
            if tensor.requires_grad:
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
    # `if a == b:` therefore cannot mean what it looks like. `__bool__` raises for any
    # Tensor that is not a single element, so that form fails loudly rather than
    # silently returning True for every non-empty Tensor. Use `(a == b).all()` for an
    # all-elements-equal test. Identity semantics are preserved separately by
    # `__hash__`, which the backward pass relies on to put Tensors in a visited set.
    # ------------------------------------------------------------------------ #
    def __eq__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[override]
        other = convert_tensor_input(other)
        return Tensor(data=equal(self.data, other.data), dtype=bool)

    def __ne__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[override]
        other = convert_tensor_input(other)
        return Tensor(data=not_equal(self.data, other.data), dtype=bool)

    def __ge__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
        other = convert_tensor_input(other)
        return Tensor(data=greater_than_equal(self.data, other.data), dtype=bool)

    def __gt__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
        other = convert_tensor_input(other)
        return Tensor(data=greater_than(self.data, other.data), dtype=bool)

    def __le__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
        other = convert_tensor_input(other)
        return Tensor(data=less_than_equal(self.data, other.data), dtype=bool)

    def __lt__(self, other: Tensor | TensorLike) -> Tensor:  # type: ignore[misc]
        other = convert_tensor_input(other)
        return Tensor(data=less_than(self.data, other.data), dtype=bool)

    def __bool__(self) -> bool:
        """Truth value of a single-element Tensor.

        Raises
        ------
        ValueError
            If this Tensor holds more than one element. Element-wise comparisons return
            a Tensor of booleans, so `if a == b:` would otherwise be True for every
            non-empty Tensor under Python's default object truthiness.
        """
        if self.size != 1:
            raise ValueError(
                "the truth value of a Tensor with more than one element is ambiguous. "
                "Use .any() or .all() to reduce it to a single boolean."
            )
        return bool(self.data)

    def all(self) -> bool:
        """True if every element is truthy."""
        return bool(self.data.all())

    def any(self) -> bool:
        """True if any element is truthy."""
        return bool(self.data.any())

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

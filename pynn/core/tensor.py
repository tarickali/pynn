from __future__ import annotations
from typing import Any
import numpy as np
import numba as nb

from pynn.core import Array, List, Number, ArrayLike, DataType, Shape

__all__ = ["Tensor"]

TensorLike = ArrayLike


class Tensor:
    def __init__(self, data: Tensor | TensorLike, dtype: DataType = np.float64) -> None:
        data = data.data if isinstance(data, Tensor) else data
        self.data: Array = np.array(data, dtype=dtype)

    def cast(self, dtype: DataType) -> None:
        if dtype != self.dtype:
            self.data = self.data.astype(dtype)

    def transpose(self) -> Tensor:
        return Tensor(self.data.T)

    def numpy(self) -> Array:
        return self.data

    def item(self) -> Number:
        return self.data.item()

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
        return Tensor(data=self.data + other.data)

    def __sub__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data - other.data)

    def __mul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data * other.data)

    def __matmul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data @ other.data)

    def __truediv__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data / other.data)

    def __radd__(self, other: Tensor | TensorLike) -> Tensor:
        return self + other

    def __rsub__(self, other: Tensor | TensorLike) -> Tensor:
        return -self + other

    def __rmul__(self, other: Tensor | TensorLike) -> Tensor:
        return self * other

    def __rtruediv__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=other.data / self.data)

    # ------------------------------------------------------------------------ #
    # Unary Operations
    # ------------------------------------------------------------------------ #
    def __pow__(self, other: Number) -> Tensor:
        if not isinstance(other, Number):
            raise ValueError(f"Cannot perform operation on {type(other)}")
        return Tensor(data=self.data**other)

    def __neg__(self) -> Tensor:
        return Tensor(data=-self.data)

    # ------------------------------------------------------------------------ #
    # Comparison Operations
    # ------------------------------------------------------------------------ #
    def __eq__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data == other.data)

    def __ne__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data != other.data)

    def __ge__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data >= other.data)

    def __gt__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data > other.data)

    def __le__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data <= other.data)

    def __lt__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(data=self.data < other.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, dtype={self.dtype}, shape={self.shape})"

    @property
    def T(self) -> Tensor:
        return Tensor(self.data.T)

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

import numpy as np
from numba import njit

from pynn.core.types import Array, Number, Shape, DataType
from pynn.core.tensor import Tensor

__all__ = [
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full_like",
    "expand_tensor",
    "shrink_tensor",
]


def zeros(shape: Shape, dtype: DataType) -> Tensor:
    return Tensor(np.zeros(shape=shape, dtype=dtype))


def zeros_like(tensor: Tensor | Array) -> Tensor:
    data = tensor.data if isinstance(tensor, Tensor) else tensor
    return Tensor(np.zeros_like(a=data))


def ones(shape: Shape, dtype: DataType) -> Tensor:
    return Tensor(np.ones(shape=shape, dtype=dtype))


def ones_like(tensor: Tensor | Array) -> Tensor:
    data = tensor.data if isinstance(tensor, Tensor) else tensor
    return Tensor(np.ones_like(a=data))


def full(shape: Shape, value: Number) -> Tensor:
    return Tensor(np.full(shape=shape, fill_value=value))


def full_like(tensor: Tensor | Array, value: Number) -> Tensor:
    data = tensor.data if isinstance(tensor, Tensor) else tensor
    return Tensor(np.full_like(a=data, fill_value=value))


def expand_tensor(tensor: Tensor, shape: Shape) -> Tensor:
    """Expand the shape of a tensor to a broadcastable shape.

    Parameters
    ----------
    tensor : Tensor
    shape : Shape

    Returns
    -------
    Tensor

    """

    # @njit
    # TODO: FIXME: Fix np.prod issue with njit
    def func(data: Array) -> Array:
        # If the shapes already align, do nothing
        if data.shape != shape:
            # If the size of the data is the same as the shape, then just reshape
            if data.size == np.prod(shape):
                data = np.reshape(data, shape)
            # Otherwise, try to broadcast the data to the shape
            # Do nothing if it fails
            else:
                try:
                    data = np.array(np.broadcast_to(data, shape))
                except:
                    pass
        return data

    data = func(tensor.data)

    return Tensor(data)


def shrink_tensor(tensor: Tensor, shape: Shape) -> Tensor:
    """Shrink the shape of a tensor from a broadcastable shape.

    Parameters
    ----------
    tensor : Tensor
    shape : Shape

    Returns
    -------
    Tensor

    """

    # @njit
    # TODO: FIXME: Fix np.prod issue with njit
    def func(data: Array) -> Array:
        # If the shapes already align, do nothing
        if data.shape != shape:
            # If the size of the data is the same as the shape, then just reshape
            if data.size == np.prod(shape):
                data = np.reshape(data, shape)
            else:
                # If the broadcastable shape is a scalar, then take the full mean
                if len(shape) < 1:
                    data = np.sum(data).reshape(shape)
                # Otherwise, try to broadcast the data to the shape
                # Do nothing if it fails
                else:
                    try:
                        broad_shape = np.broadcast_shapes(data.shape, shape)
                    except:
                        pass
                    else:
                        # Get intermediate broadcast shape according to numpy broadcasting rules
                        inter_shape = [1] * (len(broad_shape) - len(shape)) + list(
                            shape
                        )
                        # Get the axis indices that are not the same
                        axes = []
                        for i in range(len(broad_shape) - 1, -1, -1):
                            if data.shape[i] != inter_shape[i]:
                                axes.append(i)
                        # Take the mean across the axes that are not the same to collect values
                        data = np.sum(data, axis=tuple(axes)).reshape(shape)
        return data

    data = func(tensor.data)

    return Tensor(data)

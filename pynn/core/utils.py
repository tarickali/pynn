import numpy as np

from pynn.core.types import Array, Shape

__all__ = [
    "expand_array",
    "shrink_array",
]


# TODO: FIXME: Fix np.prod issue with njit
# @njit
def expand_array(array: Array, shape: Shape) -> Array:
    """Expand the shape of an array to a broadcastable shape.

    Parameters
    ----------
    array : Array
    shape : Shape

    Returns
    -------
    Array

    """
    # TODO: FIXME: Fix np.prod issue with njit

    output = array
    # If the shapes already align, do nothing
    if output.shape != shape:
        # If the size of the data is the same as the shape, then just reshape
        if output.size == np.prod(shape):
            output = np.reshape(output, shape)
        # Otherwise, try to broadcast the data to the shape
        # Do nothing if it fails
        else:
            try:
                output = np.array(np.broadcast_to(output, shape))
            except:
                pass

    return output


# TODO: FIXME: Fix np.prod issue with njit
# @njit
def shrink_array(array: Array, shape: Shape) -> Array:
    """Shrink the shape of a array from a broadcastable shape.

    Parameters
    ----------
    array : Array
    shape : Shape

    Returns
    -------
    Array

    """

    output = array
    # If the shapes already align, do nothing
    if output.shape != shape:
        # If the size of the data is the same as the shape, then just reshape
        if output.size == np.prod(shape):
            output = np.reshape(output, shape)
        else:
            # If the broadcastable shape is a scalar, then take the full mean
            if len(shape) < 1:
                output = np.sum(output).reshape(shape)
            # Otherwise, try to broadcast the data to the shape
            # Do nothing if it fails
            else:
                try:
                    broad_shape = np.broadcast_shapes(output.shape, shape)
                except:
                    pass
                else:
                    # Get intermediate broadcast shape according to numpy broadcasting rules
                    inter_shape = [1] * (len(broad_shape) - len(shape)) + list(shape)
                    # Get the axis indices that are not the same
                    axes = []
                    for i in range(len(broad_shape) - 1, -1, -1):
                        if output.shape[i] != inter_shape[i]:
                            axes.append(i)
                    # Take the mean across the axes that are not the same to collect values
                    output = np.sum(output, axis=tuple(axes)).reshape(shape)
    return output

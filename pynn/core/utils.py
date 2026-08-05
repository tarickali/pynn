import numpy as np

from pynn.core.types import Array, Shape

__all__ = [
    "matrix_multiply_gradients",
    "unbroadcast",
]


def unbroadcast(gradient: Array, shape: Shape) -> Array:
    """Reduce a gradient computed at broadcast shape back to an operand's shape.

    When NumPy broadcasts an operand during the forward pass, it implicitly copies
    that operand along the broadcast axes. The gradient of a copy is a sum, so the
    reverse pass must sum the incoming gradient over exactly those axes.

    NumPy broadcasting aligns shapes from the right and treats missing leading axes
    and axes of length 1 as broadcastable, so this reduction has two parts: sum away
    the leading axes the operand never had, then sum (keeping dimensions) the axes
    where the operand had length 1.

    Parameters
    ----------
    gradient : Array
        Gradient with respect to the broadcast result.
    shape : Shape
        Shape of the original operand.

    Returns
    -------
    Array
        Gradient with respect to the operand, of shape ``shape``.

    Examples
    --------
    >>> unbroadcast(np.ones((4, 3)), (3,)).tolist()
    [4.0, 4.0, 4.0]
    >>> unbroadcast(np.ones((4, 3)), (1, 3)).tolist()
    [[4.0, 4.0, 4.0]]
    """

    if gradient.shape == shape:
        return gradient

    leading = gradient.ndim - len(shape)
    if leading > 0:
        gradient = gradient.sum(axis=tuple(range(leading)))

    axes = tuple(
        axis
        for axis, size in enumerate(shape)
        if size == 1 and gradient.shape[axis] != 1
    )
    if axes:
        gradient = gradient.sum(axis=axes, keepdims=True)

    return gradient.reshape(shape)


def matrix_multiply_gradients(
    gradient: Array, left: Array, right: Array
) -> tuple[Array, Array]:
    """Gradients of ``left @ right`` with respect to each operand.

    For 2-D operands this is just ``gradient @ right.T`` and ``left.T @ gradient``.
    The complications are what ``np.matmul`` special-cases: a 1-D operand is promoted
    to a matrix for the multiply and the promoted axis is then dropped from the result,
    and leading axes are batched with ordinary broadcasting.

    Both are handled by promoting the operands to at least 2-D, reshaping the gradient
    to the promoted output shape, applying the matrix rule over the last two axes, and
    then undoing each adjustment in turn: a promoted axis is squeezed out (it was never
    a real axis) whereas a broadcast batch axis is summed over.

    Parameters
    ----------
    gradient : Array
        Gradient with respect to ``left @ right``.
    left, right : Array
        Forward-pass operands.

    Returns
    -------
    tuple[Array, Array]
        Gradients with respect to ``left`` and ``right``, matching their shapes.
    """

    left_is_vector = left.ndim == 1
    right_is_vector = right.ndim == 1

    # Promote vectors: a leading vector becomes a row, a trailing one becomes a column.
    left_2d = left[np.newaxis, :] if left_is_vector else left
    right_2d = right[:, np.newaxis] if right_is_vector else right

    # Rebuild the shape the product would have had without matmul's axis dropping.
    # Any dropped axis had length 1, so the element count is unchanged.
    batch_shape = np.broadcast_shapes(left_2d.shape[:-2], right_2d.shape[:-2])
    gradient_2d = gradient.reshape(
        (*batch_shape, left_2d.shape[-2], right_2d.shape[-1])
    )

    left_gradient = gradient_2d @ np.swapaxes(right_2d, -1, -2)
    right_gradient = np.swapaxes(left_2d, -1, -2) @ gradient_2d

    if left_is_vector:
        left_gradient = np.squeeze(left_gradient, axis=-2)
    if right_is_vector:
        right_gradient = np.squeeze(right_gradient, axis=-1)

    return (
        unbroadcast(left_gradient, left.shape),
        unbroadcast(right_gradient, right.shape),
    )

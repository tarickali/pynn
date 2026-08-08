"""Shape operations that stay on the tape.

Joining and splitting tensors is where a graph stops being a chain. `concat` has several
inputs and one output; `split` has one input and several outputs, each of which gets its
own gradient that has to land back in the right slice of the same tensor. Both are the
shapes a reverse pass gets wrong quietly — the result has the right dimensions either
way, and only the gradient is off.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pynn.core.tensor import Tensor

__all__ = ["concat", "split", "stack"]


def _axis_index(axis: int, position: slice | int, ndim: int) -> tuple:
    """An index tuple selecting `position` along `axis` and everything else whole."""
    index: list[slice | int] = [slice(None)] * ndim
    index[axis] = position
    return tuple(index)


def concat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join tensors along an existing axis.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Two or more tensors, matching in every axis but `axis`.
    axis : int, default 0
        Axis to join along.

    Returns
    -------
    Tensor
        The concatenation. Its gradient is split back along `axis` and added to each
        input, so a tensor passed twice correctly receives both contributions.

    Raises
    ------
    ValueError
        If no tensors are given.

    Examples
    --------
    >>> import numpy as np
    >>> a, b = Tensor(np.ones((2, 3))), Tensor(np.zeros((2, 4)))
    >>> concat([a, b], axis=1).shape
    (2, 7)
    """
    tensors = list(tensors)
    if not tensors:
        raise ValueError("concat needs at least one tensor")

    data = np.concatenate([tensor.data for tensor in tensors], axis=axis)
    output = Tensor(data)
    output.add_children(tuple(tensors))

    # Resolved once, against the output, so a negative axis means the same thing in the
    # forward and reverse passes.
    resolved = axis % output.ndim
    sizes = [tensor.shape[resolved] for tensor in tensors]

    def reverse() -> None:
        offset = 0
        for tensor, size in zip(tensors, sizes, strict=True):
            index = _axis_index(resolved, slice(offset, offset + size), output.ndim)
            tensor.grad += output.grad[index]
            offset += size

    output.forward = "concat"
    output.reverse = reverse

    return output


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join tensors along a new axis.

    Where `concat` extends an existing axis, this adds one, so every input must have
    exactly the same shape.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Two or more tensors of identical shape.
    axis : int, default 0
        Position of the new axis in the result.

    Returns
    -------
    Tensor
        Shape `tensors[0].shape` with `axis` inserted.

    Raises
    ------
    ValueError
        If no tensors are given.

    Examples
    --------
    >>> import numpy as np
    >>> stack([Tensor(np.ones(3)), Tensor(np.zeros(3))]).shape
    (2, 3)
    """
    tensors = list(tensors)
    if not tensors:
        raise ValueError("stack needs at least one tensor")

    data = np.stack([tensor.data for tensor in tensors], axis=axis)
    output = Tensor(data)
    output.add_children(tuple(tensors))

    resolved = axis % output.ndim

    def reverse() -> None:
        for position, tensor in enumerate(tensors):
            index = _axis_index(resolved, position, output.ndim)
            tensor.grad += output.grad[index]

    output.forward = "stack"
    output.reverse = reverse

    return output


def split(
    tensor: Tensor, sections: int | Sequence[int], axis: int = 0
) -> tuple[Tensor, ...]:
    """Cut a tensor into pieces along an axis.

    Each piece is its own node on the tape. Their gradients land in disjoint slices of
    the same input, so the input ends up with the gradient it would have had if the
    pieces had never been separated — including when only some of them are used.

    Parameters
    ----------
    tensor : Tensor
        Tensor to cut.
    sections : int | Sequence[int]
        A count of equal pieces, or the size of each piece.
    axis : int, default 0
        Axis to cut along.

    Returns
    -------
    tuple[Tensor, ...]
        The pieces, in order.

    Raises
    ------
    ValueError
        If `sections` does not divide the axis evenly, or the given sizes do not sum
        to its length.

    Examples
    --------
    >>> import numpy as np
    >>> [piece.shape for piece in split(Tensor(np.ones((6, 2))), 3)]
    [(2, 2), (2, 2), (2, 2)]
    """
    resolved = axis % tensor.ndim
    length = tensor.shape[resolved]

    if isinstance(sections, int):
        if sections <= 0:
            raise ValueError(f"sections must be positive, got {sections}")
        if length % sections:
            raise ValueError(
                f"cannot split axis {axis} of length {length} into {sections} equal "
                "sections"
            )
        sizes = [length // sections] * sections
    else:
        sizes = list(sections)
        if sum(sizes) != length:
            raise ValueError(
                f"sizes {sizes} sum to {sum(sizes)}, but axis {axis} has "
                f"length {length}"
            )

    pieces = []
    offset = 0
    for size in sizes:
        index = _axis_index(resolved, slice(offset, offset + size), tensor.ndim)
        piece = Tensor(tensor.data[index])
        piece.add_children((tensor,))

        # Bound as defaults: the loop variables are rebound on every iteration, and a
        # closure over them would leave every piece writing into the last slice.
        def reverse(index: tuple = index, piece: Tensor = piece) -> None:
            tensor.grad[index] += piece.grad

        piece.forward = "split"
        piece.reverse = reverse
        pieces.append(piece)
        offset += size

    return tuple(pieces)

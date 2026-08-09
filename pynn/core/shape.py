"""Operations whose reverse pass routes a gradient rather than transforms it.

Everything here moves values around without doing arithmetic on them, so the interesting
half is always the backward direction: which input each piece of the incoming gradient
belongs to. `concat` has several inputs and one output, `split` has one input and
several outputs, and `where` decides per element. All three produce correctly-shaped
results with a wrong reverse pass, which is why the gradient sweep checks a tensor
concatenated with itself, a split with an unused piece, and a `where` whose branches are
the same tensor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pynn.core.tensor import Tensor, convert_tensor_input
from pynn.core.types import Array, ArrayLike
from pynn.core.utils import unbroadcast

__all__ = ["concat", "masked_fill", "split", "stack", "where"]


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


def where(
    condition: Array | Tensor,
    x: Tensor | ArrayLike,
    y: Tensor | ArrayLike,
) -> Tensor:
    """Select from `x` where `condition` holds and from `y` where it does not.

    The reverse pass routes rather than transforms: each element of the incoming
    gradient goes to exactly one of the two inputs, and the other receives zero there.
    That is the same job `concat` and `split` do along an axis, decided per element.

    `condition` is data, not a differentiable input — a boolean has no useful
    derivative, and the gradient with respect to it is zero wherever it is defined.

    Parameters
    ----------
    condition : Array | Tensor
        Boolean selector. Broadcast against `x` and `y`.
    x, y : Tensor | ArrayLike
        Values taken where `condition` is true and false respectively. Either may be a
        scalar.

    Returns
    -------
    Tensor
        Broadcast shape of the three arguments.

    Examples
    --------
    >>> import numpy as np
    >>> x = Tensor(np.array([-1.0, 2.0, -3.0]))
    >>> where(x.data > 0, x, 0.0).data.tolist()
    [0.0, 2.0, 0.0]

    Notes
    -----
    Several operations — `relu`, `elu`, `selu`, `prelu`, `huber` — use `np.where`
    internally on raw arrays with a hand-written reverse rather than composing this.
    That is deliberate: a fused reverse for a known condition is one pass, where
    composing would allocate both branches and route a gradient through each.
    """
    mask = np.asarray(
        condition.data if isinstance(condition, Tensor) else condition, dtype=bool
    )
    x = convert_tensor_input(x)
    y = convert_tensor_input(y)

    output = Tensor(np.where(mask, x.data, y.data))
    output.add_children((x, y))

    def reverse() -> None:
        # unbroadcast, because a scalar or a lower-rank branch was replicated to the
        # output shape, and the gradient of a copy is a sum.
        x.grad += unbroadcast(np.where(mask, output.grad, 0.0), x.shape)
        y.grad += unbroadcast(np.where(mask, 0.0, output.grad), y.shape)

    output.forward = "where"
    output.reverse = reverse

    return output


def masked_fill(x: Tensor, mask: Array | Tensor, value: float) -> Tensor:
    """Replace elements of `x` where `mask` holds with the constant `value`.

    `where(mask, value, x)` with the constant kept off the tape. That matters for the
    case this exists for — attention masks — where `value` is a large negative number
    the softmax is meant to send to zero, and making it a graph node would give it a
    gradient nobody reads.

    A replaced element is overwritten, not scaled, so its gradient is exactly zero: it
    had no influence on the output.

    Parameters
    ----------
    x : Tensor
        Values to fill into.
    mask : Array | Tensor
        Boolean, broadcast against `x`. True marks positions to replace.
    value : float
        Constant written where `mask` holds.

    Returns
    -------
    Tensor
        Broadcast shape of `x` and `mask`.

    Examples
    --------
    Causal self-attention, masking each position's view of the future:

    >>> import numpy as np
    >>> import pynn.functional as F
    >>> scores = Tensor(np.zeros((1, 4, 4)))
    >>> future = np.triu(np.ones((4, 4), dtype=bool), k=1)
    >>> weights = F.softmax(masked_fill(scores, future, -1e9), axis=-1)
    >>> weights.data[0, 0].round(3).tolist()
    [1.0, 0.0, 0.0, 0.0]
    """
    selector = np.asarray(mask.data if isinstance(mask, Tensor) else mask, dtype=bool)

    output = Tensor(np.where(selector, value, x.data))
    output.add_children((x,))

    def reverse() -> None:
        x.grad += unbroadcast(np.where(selector, 0.0, output.grad), x.shape)

    output.forward = "masked_fill"
    output.reverse = reverse

    return output

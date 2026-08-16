from __future__ import annotations

import numpy as np

from pynn.core import Tensor
from pynn.core.random import default_rng
from pynn.core.types import Array, Shape
from pynn.core.utils import unbroadcast
from pynn.utils.array import col2im, im2col, pad_for_conv

__all__ = [
    "avg_pool2d",
    "batch_norm",
    "conv2d",
    "dropout",
    "embedding",
    "flatten",
    "layer_norm",
    "linear",
    "max_pool2d",
    "unflatten",
]


def linear(X: Tensor, W: Tensor, b: Tensor | None) -> Tensor:
    return X @ W if b is None else X @ W + b


def flatten(x: Tensor) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)
    array = x.data
    # int(), because np.prod returns a NumPy scalar and reshape wants a plain index —
    # for an empty trailing shape it returns 1.0, a float, which reshape rejects.
    data = array.reshape(-1, int(np.prod(array.shape[1:])))
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = output.grad.reshape(array.shape)
        x.grad += grad

    output.forward = "flatten"
    output.reverse = reverse

    return output


def unflatten(x: Tensor, shape: Shape) -> Tensor:
    """Inverse of `flatten`: restore the trailing axes it collapsed.

    `flatten` maps ``(batch, d1, d2, ...)`` to ``(batch, d1*d2*...)``; this maps
    ``(batch, n)`` back to ``(batch, *shape)``. The batch axis is read from the input
    rather than taken from `shape`, which is the difference between this and a general
    reshape: a layer that carried the batch size in its target shape would work for
    every batch of an epoch except the last, smaller one.

    One entry of `shape` may be ``-1`` and is inferred, as it is in NumPy.

    Parameters
    ----------
    x : Tensor
        Input whose axes past the first are to be re-laid out.
    shape : Shape
        Trailing axes of the result.

    Returns
    -------
    Tensor
        Shape ``(batch, *shape)``.

    Raises
    ------
    ValueError
        If one example's elements cannot be laid out as `shape`.

    Examples
    --------
    >>> unflatten(Tensor(np.zeros((8, 50))), (2, 5, 5)).shape
    (8, 2, 5, 5)
    """
    x = x if isinstance(x, Tensor) else Tensor(x)
    array = x.data
    target = (array.shape[0], *shape)
    try:
        data = array.reshape(target)
    except ValueError as error:
        raise ValueError(
            f"cannot unflatten {array.shape} into {target}: one example holds "
            f"{int(np.prod(array.shape[1:]))} elements"
        ) from error

    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        # Re-laying out an array moves no data, so the reverse is the inverse layout.
        x.grad += output.grad.reshape(array.shape)

    output.forward = "unflatten"
    output.reverse = reverse

    return output


def conv2d(
    X: Tensor,
    K: Tensor,
    B: Tensor | None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> Tensor:
    """2D convolution via im2col and a single matrix multiply.

    Sliding-window convolution is equivalent to unrolling every input patch into a
    column of a matrix and multiplying by the reshaped kernel. That replaces a Python
    loop over output positions with one BLAS gemm, which is the difference between a
    toy convolution and one that trains a small CNN in reasonable time.

    Parameters
    ----------
    X : Tensor
        Input of shape ``(batch, in_ch, in_h, in_w)``.
    K : Tensor
        Kernel of shape ``(out_ch, in_ch, kh, kw)``.
    B : Tensor | None
        Bias of shape ``(out_ch, 1, 1)``, or any shape that broadcasts against
        ``(out_ch, out_h, out_w)``. One bias per output channel, shared across spatial
        positions, is what makes the layer translation-equivariant; a bias per output
        position would also tie the parameter count to the input resolution.
    stride : tuple[int, int]
        ``(stride_h, stride_w)``.
    padding : tuple[int, int]
        Zero-padding added to each side of the spatial axes.

    Returns
    -------
    Tensor
        Output of shape ``(batch, out_ch, out_h, out_w)``.
    """
    batch_size = X.shape[0]
    out_ch, _, kh, kw = K.shape
    sh, sw = stride
    ph, pw = padding

    X_arr = X.data
    K_arr = K.data
    X_pad = pad_for_conv(X_arr, ph, pw)
    _, _, padded_h, padded_w = X_pad.shape

    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1
    out_size = (out_h, out_w)
    kernel_size = (kh, kw)
    # Concrete 4-D shape for col2im; `X.shape` is typed as a variable-length Shape.
    input_shape = (batch_size, X.shape[1], X.shape[2], X.shape[3])

    # (N*oh*ow, in_ch*kh*kw) @ (in_ch*kh*kw, out_ch) -> (N*oh*ow, out_ch)
    cols = im2col(X_pad, kernel_size, stride, out_size)
    K_mat = K_arr.reshape(out_ch, -1)
    out_mat = cols @ K_mat.T
    data = out_mat.reshape(batch_size, out_h, out_w, out_ch).transpose(0, 3, 1, 2)

    if B is not None:
        data = data + B.data

    output = Tensor(data)
    output.add_children((X, K) if B is None else (X, K, B))

    def reverse():
        # Undo the forward reshape so the chain rule is a pair of matrix products.
        O_mat = output.grad.transpose(0, 2, 3, 1).reshape(
            batch_size * out_h * out_w, out_ch
        )

        K.grad += (O_mat.T @ cols).reshape(K.shape)
        X.grad += col2im(
            O_mat @ K_mat,
            input_shape,
            kernel_size,
            stride,
            padding,
            out_size,
        )
        if B is not None:
            # Sums over batch, and over the spatial axes the bias was broadcast along.
            B.grad += unbroadcast(output.grad, B.shape)

    output.reverse = reverse
    output.forward = "conv2d"
    return output


def dropout(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    rng: int | np.random.Generator | None = None,
) -> Tensor:
    """Randomly zero elements of `x` during training, rescaling the survivors.

    Inverted dropout: the kept elements are divided by the keep probability, so the
    expected value of the output matches the input and evaluation needs no
    compensating factor. That is why `training=False` is exactly the identity and not
    "the same thing scaled by p".

    Parameters
    ----------
    x : Tensor
        Input of any shape.
    p : float, default 0.5
        Probability of zeroing each element, in [0, 1].
    training : bool, default True
        When False, returns `x` unchanged. `nn.Dropout` passes `self.training`.
    rng : int | np.random.Generator | None
        Seed or generator for the mask. The default draws from the shared generator
        that `pynn.core.random.set_seed` controls.

    Returns
    -------
    Tensor
        Same shape as `x`.

    Raises
    ------
    ValueError
        If `p` is outside [0, 1].
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    if not training or p == 0.0:
        return x

    keep = 1.0 - p
    if keep == 0.0:
        mask = np.zeros(x.shape, dtype=x.dtype)
    else:
        mask = (default_rng(rng).random(x.shape) < keep).astype(x.dtype) / keep

    output = Tensor(mask * x.data)
    output.add_children((x,))

    def reverse() -> None:
        # The mask is a constant here, so the gradient is routed through exactly the
        # elements that survived, scaled the same way the forward pass scaled them.
        x.grad += mask * output.grad

    output.forward = "dropout"
    output.reverse = reverse

    return output


def _normalize(
    x: Tensor,
    gamma: Tensor | None,
    beta: Tensor | None,
    mean: Array,
    variance: Array,
    axes: tuple[int, ...],
    eps: float,
    name: str,
    differentiate_statistics: bool,
) -> Tensor:
    """Shared core of `layer_norm` and `batch_norm`.

    The two differ only in which axes the statistics are taken over, and in whether
    those statistics depend on `x` at all: batch normalization at evaluation time uses
    fixed running estimates, which makes it an affine function of `x` and its gradient
    the simple `dy * gamma / std`.

    The training-time gradient is the standard fused form,

        dx = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / std

    where the means are over the normalized axes. Written out rather than composed from
    primitives because the mean and the variance both depend on every element of `x`,
    so the naive graph re-derives that dependency once per element.
    """
    inverse_std = 1.0 / np.sqrt(variance + eps)
    centered = x.data - mean
    normalized = centered * inverse_std
    scale = None if gamma is None else gamma.data

    data = normalized if scale is None else normalized * scale
    if beta is not None:
        data = data + beta.data

    output = Tensor(data)
    children: tuple[Tensor, ...] = (x,)
    if gamma is not None:
        children += (gamma,)
    if beta is not None:
        children += (beta,)
    output.add_children(children)

    def reverse() -> None:
        upstream = output.grad
        if beta is not None:
            beta.grad += unbroadcast(upstream, beta.shape)
        if gamma is not None:
            gamma.grad += unbroadcast(upstream * normalized, gamma.shape)

        d_normalized = upstream if scale is None else upstream * scale
        if differentiate_statistics:
            # The two subtracted means are the paths through the mean and the variance;
            # both are exact for the biased variance numpy's `var` computes.
            x.grad += (
                d_normalized
                - d_normalized.mean(axis=axes, keepdims=True)
                - normalized
                * (d_normalized * normalized).mean(axis=axes, keepdims=True)
            ) * inverse_std
        else:
            x.grad += d_normalized * inverse_std

    output.forward = name
    output.reverse = reverse

    return output


def layer_norm(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    normalized_shape: Shape | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Normalize each example over its trailing axes, then scale and shift.

    Unlike batch normalization, the statistics are per example, so the result does not
    depend on what else is in the batch and training and evaluation are the same
    computation. That is the reason it is the default in sequence models, where the
    batch axis is not the one carrying comparable statistics.

    Parameters
    ----------
    x : Tensor
        Input of shape `(..., *normalized_shape)`.
    gamma, beta : Tensor | None
        Learnable scale and shift, broadcastable against `normalized_shape`.
    normalized_shape : Shape | None
        Trailing axes to normalize over. Defaults to the last axis.
    eps : float, default 1e-5
        Added to the variance before the square root.

    Returns
    -------
    Tensor
        Same shape as `x`.
    """
    if normalized_shape is None:
        normalized_shape = (x.shape[-1],)
    axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    if x.shape[len(x.shape) - len(normalized_shape) :] != tuple(normalized_shape):
        raise ValueError(
            f"input of shape {x.shape} does not end with normalized_shape "
            f"{tuple(normalized_shape)}"
        )

    mean = x.data.mean(axis=axes, keepdims=True)
    variance = x.data.var(axis=axes, keepdims=True)
    return _normalize(x, gamma, beta, mean, variance, axes, eps, "layer_norm", True)


def batch_norm(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    running_mean: Array | None = None,
    running_var: Array | None = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Normalize each feature over the batch (and any spatial axes), scale, and shift.

    Axes are taken to be every axis except axis 1, so `(N, C)` normalizes over the
    batch and `(N, C, H, W)` normalizes over the batch and both spatial axes — one
    statistic per channel either way.

    Parameters
    ----------
    x : Tensor
        Input of shape `(batch, features, ...)`.
    gamma, beta : Tensor | None
        Learnable scale and shift, shaped to broadcast against `x` along axis 1.
    running_mean, running_var : Array | None
        Running estimates, updated in place during training and used in place of the
        batch statistics at evaluation time. Updated in place because they are buffers
        the module owns, not parameters the optimizer steps.
    training : bool, default True
        Use the batch statistics and update the running ones, rather than reading them.
    momentum : float, default 0.1
        Weight of the current batch in the running estimates, matching PyTorch's
        convention (`new = (1 - momentum) * old + momentum * batch`).
    eps : float, default 1e-5
        Added to the variance before the square root.

    Returns
    -------
    Tensor
        Same shape as `x`.
    """
    axes = (0, *range(2, x.ndim))
    statistics_shape = tuple(1 if axis != 1 else x.shape[1] for axis in range(x.ndim))

    if training or running_mean is None or running_var is None:
        mean = x.data.mean(axis=axes, keepdims=True)
        variance = x.data.var(axis=axes, keepdims=True)
        if running_mean is not None and running_var is not None:
            count = int(np.prod([x.shape[axis] for axis in axes]))
            # The running variance tracks the unbiased estimate, as PyTorch does, so
            # that evaluation on a single batch is not biased low by 1/count.
            unbiased = variance * count / (count - 1) if count > 1 else variance
            running_mean *= 1.0 - momentum
            running_mean += momentum * mean.reshape(running_mean.shape)
            running_var *= 1.0 - momentum
            running_var += momentum * unbiased.reshape(running_var.shape)
    else:
        mean = running_mean.reshape(statistics_shape)
        variance = running_var.reshape(statistics_shape)

    return _normalize(x, gamma, beta, mean, variance, axes, eps, "batch_norm", training)


def _pool_windows(
    x: Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    fill: float,
) -> tuple[Array, tuple[int, int], tuple[int, int, int, int]]:
    """Each channel's sliding windows as rows: `(batch * ch * out_h * out_w, kh * kw)`.

    Pooling acts on each channel independently, so the channels are folded into the
    batch axis and `im2col` — which unrolls windows across channels for a convolution —
    unrolls one channel per row here.
    """
    batch, channels, height, width = (
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
    )
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding

    merged_shape = (batch * channels, 1, height, width)
    merged = x.data.reshape(merged_shape)
    padded = pad_for_conv(merged, ph, pw, value=fill)
    out_h = (height + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1

    return (
        im2col(padded, kernel_size, stride, (out_h, out_w)),
        (out_h, out_w),
        merged_shape,
    )


def _check_pool_shape(x: Tensor, name: str) -> None:
    if x.ndim != 4:
        raise ValueError(
            f"{name} expects an input of shape (batch, channels, height, width), "
            f"got {x.shape}"
        )


def max_pool2d(
    x: Tensor,
    kernel_size: tuple[int, int] = (2, 2),
    stride: tuple[int, int] | None = None,
    padding: tuple[int, int] = (0, 0),
) -> Tensor:
    """Take the maximum over each sliding window of every channel.

    The reverse pass routes each output's gradient to the single input position that
    won its window, and adds where windows overlap. Ties go to the first position, as
    they do in PyTorch; the function is not differentiable there in any case.

    Parameters
    ----------
    x : Tensor
        Input of shape `(batch, channels, height, width)`.
    kernel_size : tuple[int, int], default (2, 2)
        Window height and width.
    stride : tuple[int, int] | None
        Window step. Defaults to `kernel_size`, giving non-overlapping windows.
    padding : tuple[int, int], default (0, 0)
        Padding per side, filled with negative infinity so it can never win a window.

    Returns
    -------
    Tensor
        Output of shape `(batch, channels, out_h, out_w)`.
    """
    _check_pool_shape(x, "max_pool2d")
    stride = kernel_size if stride is None else stride
    cols, out_size, merged_shape = _pool_windows(
        x, kernel_size, stride, padding, -np.inf
    )

    batch, channels = x.shape[0], x.shape[1]
    out_h, out_w = out_size
    rows = np.arange(cols.shape[0])
    argmax = cols.argmax(axis=1)
    data = cols[rows, argmax].reshape(batch, channels, out_h, out_w)

    output = Tensor(data)
    output.add_children((x,))

    def reverse() -> None:
        column_gradient = np.zeros(cols.shape, dtype=output.grad.dtype)
        column_gradient[rows, argmax] = output.grad.reshape(-1)
        x.grad += col2im(
            column_gradient, merged_shape, kernel_size, stride, padding, out_size
        ).reshape(x.shape)

    output.forward = "max_pool2d"
    output.reverse = reverse

    return output


def avg_pool2d(
    x: Tensor,
    kernel_size: tuple[int, int] = (2, 2),
    stride: tuple[int, int] | None = None,
    padding: tuple[int, int] = (0, 0),
) -> Tensor:
    """Average over each sliding window of every channel.

    Padded positions are counted in the denominator, matching PyTorch's default
    `count_include_pad=True`.

    Parameters
    ----------
    x : Tensor
        Input of shape `(batch, channels, height, width)`.
    kernel_size : tuple[int, int], default (2, 2)
        Window height and width.
    stride : tuple[int, int] | None
        Window step. Defaults to `kernel_size`.
    padding : tuple[int, int], default (0, 0)
        Zero padding per side.

    Returns
    -------
    Tensor
        Output of shape `(batch, channels, out_h, out_w)`.
    """
    _check_pool_shape(x, "avg_pool2d")
    stride = kernel_size if stride is None else stride
    cols, out_size, merged_shape = _pool_windows(x, kernel_size, stride, padding, 0.0)

    batch, channels = x.shape[0], x.shape[1]
    out_h, out_w = out_size
    window = kernel_size[0] * kernel_size[1]
    data = cols.mean(axis=1).reshape(batch, channels, out_h, out_w)

    output = Tensor(data)
    output.add_children((x,))

    def reverse() -> None:
        # Every position in a window contributed equally, so each gets 1/window of the
        # output's gradient; col2im adds the overlaps.
        column_gradient = np.repeat(output.grad.reshape(-1, 1) / window, window, axis=1)
        x.grad += col2im(
            column_gradient, merged_shape, kernel_size, stride, padding, out_size
        ).reshape(x.shape)

    output.forward = "avg_pool2d"
    output.reverse = reverse

    return output


def embedding(weight: Tensor, indices: Array | Tensor) -> Tensor:
    """Look up rows of `weight` by integer index.

    A differentiable gather. The reverse is a scatter-add rather than a scatter: a token
    appearing twice in a batch contributes twice to its row, and the whole point of an
    embedding is that common tokens appear often. Writing instead of adding would keep
    one occurrence per row per step and quietly train frequent tokens as if they were
    rare.

    Parameters
    ----------
    weight : Tensor
        Embedding table of shape ``(num_embeddings, dim)``.
    indices : Array | Tensor
        Integer indices of any shape. Not differentiated.

    Returns
    -------
    Tensor
        Shape ``(*indices.shape, dim)``.

    Raises
    ------
    ValueError
        If `weight` is not a matrix, or an index is out of range.
    """
    if weight.ndim != 2:
        raise ValueError(
            "embedding weight must have shape (num_embeddings, dim), got "
            f"{weight.shape}"
        )

    lookup = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
    lookup = lookup.astype(np.intp)
    count = weight.shape[0]
    if lookup.size and (lookup.min() < 0 or lookup.max() >= count):
        raise ValueError(
            f"indices must be in [0, {count}), got range "
            f"[{lookup.min()}, {lookup.max()}]"
        )

    flat = lookup.reshape(-1)
    data = weight.data[flat].reshape(*lookup.shape, weight.shape[1])

    output = Tensor(data)
    output.add_children((weight,))

    def reverse() -> None:
        # np.add.at, not `weight.grad[flat] = ...`: repeated indices must accumulate.
        np.add.at(weight.grad, flat, output.grad.reshape(-1, weight.shape[1]))

    output.forward = "embedding"
    output.reverse = reverse

    return output

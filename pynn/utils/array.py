"""Array helpers behind `conv2d` and the pooling layers.

`im2col` and `col2im` are what turn a sliding-window convolution into a single matrix
multiply. The forward direction is a strided view and a `gemm`, with no Python loop
left in it. The reverse direction cannot be: overlapping windows contribute to the same
input pixel, so the scatter has to accumulate, and `+=` on overlapping slices is not
something NumPy can vectorize in one call.

That scatter is the only Python-level loop left in the library, and on a small CNN it
is around 40% of a training step. It is therefore the one place where compiling
actually pays — see `scatter_windows` below.
"""

from collections.abc import Callable

import numpy as np

__all__ = ["NUMBA_AVAILABLE", "col2im", "im2col", "make_pair", "pad_for_conv"]


def make_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Return (x, x) if x is int, else x. Used for kernel_size, stride, etc."""
    if isinstance(x, int):
        return (x, x)
    return x


def pad_for_conv(
    x: np.ndarray, pad_h: int, pad_w: int, value: float = 0.0
) -> np.ndarray:
    """Pad spatial dims (last two) of array (batch, ch, h, w).

    `value` is what the padding is filled with: zero for convolution and average
    pooling, but negative infinity for max pooling, where a padded position must never
    be able to win the maximum.
    """
    if pad_h == 0 and pad_w == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=value,
    )


def im2col(
    x: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    out_size: tuple[int, int],
) -> np.ndarray:
    """Unroll the sliding windows of a padded image into columns.

    Turns a convolution into a single matrix multiply: each column of the result is one
    patch of the input, flattened, so ``cols @ kernel.T`` produces every output position
    at once. Implemented as a strided view rather than a Python loop over positions, so
    the subsequent matmul is the only work that scales with output size.

    Parameters
    ----------
    x : np.ndarray
        Padded input of shape ``(batch, channels, height, width)``.
    kernel_size : tuple[int, int]
        ``(kh, kw)``.
    stride : tuple[int, int]
        ``(sh, sw)``.
    out_size : tuple[int, int]
        ``(out_h, out_w)`` of the convolution that will consume these columns.

    Returns
    -------
    np.ndarray
        Array of shape ``(batch * out_h * out_w, channels * kh * kw)``. Contiguous; safe
        to keep across the reverse pass even if ``x`` is later mutated.
    """
    batch, channels, _, _ = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    out_h, out_w = out_size

    # shape[i] * strides[i] walks one step in dimension i of the view. The out_h/out_w
    # axes advance by a stride of patches; the kh/kw axes advance by one pixel.
    item_h, item_w = x.strides[-2:]
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(batch, channels, out_h, out_w, kh, kw),
        strides=(
            x.strides[0],
            x.strides[1],
            sh * item_h,
            sw * item_w,
            item_h,
            item_w,
        ),
        writeable=False,
    )
    # (N, C, oh, ow, kh, kw) -> (N, oh, ow, C, kh, kw) -> (N*oh*ow, C*kh*kw)
    return np.ascontiguousarray(
        patches.transpose(0, 2, 3, 1, 4, 5).reshape(
            batch * out_h * out_w, channels * kh * kw
        )
    )


def _scatter_windows_numpy(
    patches: np.ndarray,
    padded: np.ndarray,
    out_h: int,
    out_w: int,
    kh: int,
    kw: int,
    sh: int,
    sw: int,
) -> None:
    """Add every window's contribution back into `padded`, in place.

    One iteration per output position, each adding a whole `(batch, channels, kh, kw)`
    block. NumPy does the inner work, so the loop is over the output grid rather than
    over pixels — but it is still a Python loop, and it is where the reverse pass of a
    convolution spends most of its time.
    """
    for oh in range(out_h):
        h_start = oh * sh
        for ow in range(out_w):
            w_start = ow * sw
            padded[:, :, h_start : h_start + kh, w_start : w_start + kw] += patches[
                :, :, oh, ow
            ]


def _scatter_windows_scalar(
    patches: np.ndarray,
    padded: np.ndarray,
    out_h: int,
    out_w: int,
    kh: int,
    kw: int,
    sh: int,
    sw: int,
) -> None:
    """The same scatter as explicit scalar loops, which is what a JIT wants.

    Deliberately the slower shape in pure Python — six nested loops over individual
    elements — and the faster one once compiled, since the compiler removes the loop
    overhead that made it slow and NumPy's per-call dispatch along with it. Kept
    byte-for-byte equivalent to `_scatter_windows_numpy`; `tests/utils/array_test.py`
    asserts the two agree.
    """
    for n in range(patches.shape[0]):
        for c in range(patches.shape[1]):
            for oh in range(out_h):
                h_start = oh * sh
                for ow in range(out_w):
                    w_start = ow * sw
                    for i in range(kh):
                        for j in range(kw):
                            padded[n, c, h_start + i, w_start + j] += patches[
                                n, c, oh, ow, i, j
                            ]


ScatterWindows = Callable[[np.ndarray, np.ndarray, int, int, int, int, int, int], None]

try:
    from numba import njit
except ImportError:  # pragma: no cover - depends on an optional extra being absent
    #: Whether the compiled scatter is in use. `pynn` is fully functional without it;
    #: `numba` is an optional extra precisely because a 150 MB dependency should be a
    #: choice, not a condition of installing a NumPy library.
    NUMBA_AVAILABLE = False
    scatter_windows: ScatterWindows = _scatter_windows_numpy
else:  # pragma: no cover - depends on an optional extra being present
    NUMBA_AVAILABLE = True
    # cache=True writes the compiled form next to the source, so the ~1s compile is
    # paid once per machine rather than once per process. Numba degrades to an
    # in-memory cache if the directory is read-only.
    scatter_windows = njit(cache=True)(_scatter_windows_scalar)


def col2im(
    cols: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    out_size: tuple[int, int],
) -> np.ndarray:
    """Inverse of `im2col`: scatter columns back into an image, summing overlaps.

    The reverse of a convolution's im2col step. Patches that overlapped in the forward
    pass contribute to the same input pixel, so those contributions are added.

    Parameters
    ----------
    cols : np.ndarray
        Gradient w.r.t. the im2col matrix, shape
        ``(batch * out_h * out_w, channels * kh * kw)``.
    input_shape : tuple[int, int, int, int]
        Shape of the *unpadded* input, ``(batch, channels, height, width)``.
    kernel_size, stride, padding, out_size
        The same convolution geometry that produced ``cols``.

    Returns
    -------
    np.ndarray
        Gradient w.r.t. the unpadded input, of shape ``input_shape``.

    Notes
    -----
    This is the hot spot of a convolutional backward pass — roughly 40% of a training
    step for a small CNN. Installing the `numba` extra compiles the scatter and cuts
    that step by about a third; see `scatter_windows`.
    """
    batch, channels, height, width = input_shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    out_h, out_w = out_size

    padded = np.zeros(
        (batch, channels, height + 2 * ph, width + 2 * pw), dtype=cols.dtype
    )
    # (N*oh*ow, C*kh*kw) -> (N, C, oh, ow, kh, kw), matching the im2col layout.
    patches = cols.reshape(batch, out_h, out_w, channels, kh, kw).transpose(
        0, 3, 1, 2, 4, 5
    )
    # Overlapping patches contribute to the same input pixel, so each window is added
    # rather than assigned. `scatter_windows` is the compiled scalar loop when Numba is
    # installed and the NumPy block loop otherwise; the two agree exactly.
    if NUMBA_AVAILABLE:
        patches = np.ascontiguousarray(patches)
    scatter_windows(patches, padded, out_h, out_w, kh, kw, sh, sw)

    if ph == 0 and pw == 0:
        return padded
    return padded[:, :, ph : ph + height, pw : pw + width]

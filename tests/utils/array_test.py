"""Direct tests for the im2col / col2im helpers that back `conv2d`."""

import numpy as np
import pytest

from pynn.utils.array import col2im, im2col, pad_for_conv


def _naive_im2col(x, kernel_size, stride, out_size):
    """Reference: gather each patch with ordinary slicing."""
    batch, channels, _, _ = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    out_h, out_w = out_size
    cols = np.empty((batch * out_h * out_w, channels * kh * kw), dtype=x.dtype)
    row = 0
    for n in range(batch):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = x[n, :, oh * sh : oh * sh + kh, ow * sw : ow * sw + kw]
                cols[row] = patch.reshape(-1)
                row += 1
    return cols


@pytest.mark.parametrize(
    "stride,padding",
    [((1, 1), (0, 0)), ((2, 2), (0, 0)), ((1, 1), (1, 1)), ((2, 2), (1, 2))],
    ids=str,
)
def test_im2col_matches_naive_gathering(rng, stride, padding):
    x = rng.standard_normal((3, 2, 7, 7))
    kh, kw = 3, 3
    ph, pw = padding
    x_pad = pad_for_conv(x, ph, pw)
    _, _, padded_h, padded_w = x_pad.shape
    out_h = (padded_h - kh) // stride[0] + 1
    out_w = (padded_w - kw) // stride[1] + 1

    assert np.allclose(
        im2col(x_pad, (kh, kw), stride, (out_h, out_w)),
        _naive_im2col(x_pad, (kh, kw), stride, (out_h, out_w)),
    )


def test_col2im_is_the_adjoint_of_im2col(rng):
    """``<im2col(x), cols> == <x, col2im(cols)>`` for every ``x`` and ``cols``.

    That identity is exactly the statement that col2im is the reverse-mode of im2col,
    so a gradient check on `conv2d` can only pass if this holds. Checked without
    padding so the comparison is against the same array im2col read; padded geometry
    is covered by the `conv2d` gradient cases.
    """
    x = rng.standard_normal((2, 3, 6, 6))
    kh, kw, sh, sw = 3, 3, 1, 1
    out_h = (6 - kh) // sh + 1
    out_w = (6 - kw) // sw + 1

    cols = im2col(x, (kh, kw), (sh, sw), (out_h, out_w))
    cols_grad = rng.standard_normal(cols.shape)
    x_grad = col2im(cols_grad, x.shape, (kh, kw), (sh, sw), (0, 0), (out_h, out_w))

    assert np.allclose((cols * cols_grad).sum(), (x * x_grad).sum())


def test_im2col_result_is_detached_from_the_input():
    """Columns must outlive input mutations, or the reverse pass reads stale data."""
    x = np.ones((1, 1, 3, 3))
    cols = im2col(x, (2, 2), (1, 1), (2, 2))
    x[:] = 0
    assert cols.sum() == 4 * 4  # four patches of four ones

import numpy as np

from pynn.core import Tensor
from pynn.core.utils import unbroadcast
from pynn.utils.array import col2im, im2col, pad_for_conv

__all__ = ["conv2d", "flatten", "linear"]


def linear(X: Tensor, W: Tensor, b: Tensor | None) -> Tensor:
    return X @ W if b is None else X @ W + b


def flatten(x: Tensor) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)
    array = x.data
    data = array.reshape(-1, np.prod(array.shape[1:]))
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        grad = output.grad.reshape(array.shape)
        x.grad += grad

    output.forward = "flatten"
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

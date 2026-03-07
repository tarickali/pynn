import numpy as np
from pynn.core import Tensor
from pynn.utils.array import pad_for_conv

__all__ = ["linear", "flatten", "conv2d"]


def linear(X: Tensor, W: Tensor, b: Tensor | None) -> Tensor:
    # Compute linear transformation
    if b is None:
        Z = X @ W
    else:
        Z = X @ W + b
    return Z


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
    """2D convolution with stride and padding.

    Parameters:
    -----------
    X : Tensor
        Input tensor of shape (batch, in_ch, in_h, in_w).
    K : Tensor
        Kernel tensor of shape (out_ch, in_ch, kh, kw).
    B : Tensor | None
        Bias tensor of shape (out_ch, out_h, out_w).
    stride : tuple[int, int]
        Stride for the convolution.
    padding : tuple[int, int]
        Padding for the convolution.

    Returns:
    --------
    Tensor
        Output tensor of shape (batch, out_ch, out_h, out_w).
    """
    batch_size, input_shape = X.shape[0], X.shape[1:]
    out_ch, in_ch, kh, kw = K.shape
    _, in_h, in_w = X.shape[1:]
    sh, sw = stride
    ph, pw = padding

    # Pad input
    X_arr = X.data
    X_pad = pad_for_conv(X_arr, ph, pad_w=pw)
    _, _, padded_h, padded_w = X_pad.shape

    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1
    output_shape = (out_ch, out_h, out_w)

    data = np.zeros((batch_size,) + output_shape, dtype=X_arr.dtype)
    K_arr = K.data

    for oh in range(out_h):
        for ow in range(out_w):
            h_start, w_start = oh * sh, ow * sw
            patch = X_pad[:, :, h_start : h_start + kh, w_start : w_start + kw]
            # patch (batch, in_ch, kh, kw), K (out_ch, in_ch, kh, kw) -> (batch, out_ch)
            data[:, :, oh, ow] = np.einsum("bijk,oijk->bo", patch, K_arr)

    if B is not None:
        data += B.data

    output = Tensor(data)
    output.add_children((X, K) if B is None else (X, K, B))

    def reverse():
        O_grad = output.grad
        K_grad = np.zeros(K.shape, dtype=K_arr.dtype)
        X_pad_grad = np.zeros_like(X_pad)

        for oh in range(out_h):
            for ow in range(out_w):
                h_start, w_start = oh * sh, ow * sw
                # d(out[b,o,oh,ow])/d(patch) = K[o]; d/dK = patch * out_grad
                patch = X_pad[:, :, h_start : h_start + kh, w_start : w_start + kw]
                # O_grad (batch, out_ch) -> broadcast to (batch, out_ch, in_ch, kh, kw)
                X_pad_grad[
                    :, :, h_start : h_start + kh, w_start : w_start + kw
                ] += np.einsum("bo,oijk->bijk", O_grad[:, :, oh, ow], K_arr)
                K_grad += np.einsum("bo,bijk->oijk", O_grad[:, :, oh, ow], patch)

        # Unpad gradient back to input shape
        if ph == 0 and pw == 0:
            X.grad = X_pad_grad
        else:
            X.grad = X_pad_grad[:, :, ph : ph + in_h, pw : pw + in_w]

        K.grad = K_grad
        if B is not None:
            # B has shape (out_ch, out_h, out_w); gradient is sum over batch
            B.grad = np.sum(O_grad, axis=0)

    output.reverse = reverse
    output.forward = "conv2d"
    return output

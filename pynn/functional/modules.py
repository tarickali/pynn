import numpy as np
from scipy import signal
from pynn.core import Tensor

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


def conv2d(X: Tensor, K: Tensor, B: Tensor | None) -> Tensor:
    batch_size, input_shape = X.shape[0], X.shape[1:]
    out_channels, in_channels, kernel_height, kernel_width = K.shape
    in_channels, in_height, in_width = X.shape[1:]
    output_shape = (
        out_channels,
        in_height - kernel_height + 1,
        in_width - kernel_width + 1,
    )

    # Setup children for output node
    children = (X, K)

    # Compute output
    X_arr = X.data
    K_arr = K.data
    data = np.zeros((batch_size,) + output_shape)
    for i in range(out_channels):
        for j in range(in_channels):
            data[:, i] += signal.correlate(X_arr[:, j], K_arr[None, i, j], mode="valid")

    # Include bias if present
    if B is not None:
        children += (B,)
        data += B.data

    # Create output node
    output = Tensor(data)
    output.add_children(children)

    def reverse():
        K_grad = np.zeros(K.shape)
        X_grad = np.zeros((batch_size,) + input_shape)
        O_grad = output.grad

        for i in range(out_channels):
            for j in range(in_channels):
                K_grad[i, j] = signal.correlate(X_arr[:, j], O_grad[:, i], mode="valid")
                X_grad[:, j] += signal.convolve(
                    O_grad[:, i], K_arr[None, i, j], mode="full"
                )
        K.grad = K_grad
        X.grad = X_grad

        if B is not None:
            B.grad = O_grad

    output.reverse = reverse
    output.forward = "conv2d"

    return output

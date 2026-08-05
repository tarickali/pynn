from typing import Any

from pynn.core import Module, Tensor
from pynn.core.types import Shape
from pynn.functional.modules import conv2d, flatten, linear
from pynn.nn.factories import activation_factory, initializer_factory
from pynn.utils.array import make_pair

__all__ = ["Activation", "Conv2d", "Flatten", "Linear"]


class Linear(Module):
    """Linear (fully connected) layer: output = activation(X @ W + b).

    Shape can be specified in two ways:
    - Linear(in_features, out_features, ...) — both dimensions (PyTorch-style).
    - Linear(out_features, ...) — output size only; in_features inferred on first call.
    """

    def __init__(
        self,
        *dims: int,
        activation: str | dict[str, Any] = "identity",
        weight_initializer: str | dict[str, Any] = "xavier_normal",
        bias_initializer: str | dict[str, Any] = "zeros",
        include_bias: bool = True,
        name: str = "Linear",
    ) -> None:
        super().__init__()

        if len(dims) == 1:
            self.in_features = None  # inferred at build
            self.out_features = dims[0]
        elif len(dims) == 2:
            self.in_features, self.out_features = dims[0], dims[1]
        else:
            raise TypeError(
                "Linear() takes 1 or 2 positional dimension arguments "
                f"(out_features, or in_features and out_features), got {len(dims)}"
            )
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.include_bias = include_bias
        self.name = name

        self.act_fn = activation_factory(self.activation)
        self.weight_init = initializer_factory(self.weight_initializer)
        if self.include_bias:
            self.bias_init = initializer_factory(self.bias_initializer)

    def build(self, input_shape: Shape) -> None:
        assert len(input_shape) == 2
        if self.in_features is None:
            self.in_features = input_shape[1]
        else:
            assert input_shape[1] == self.in_features, (
                f"Expected in_features {self.in_features}, got {input_shape[1]}"
            )

        W = self.register_parameter(
            "W", self.weight_init((self.in_features, self.out_features))
        )
        assert W.shape == (self.in_features, self.out_features)

        if self.include_bias:
            b = self.register_parameter("b", self.bias_init((self.out_features,)))
            assert b.shape == (self.out_features,)

        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        if not self.initialized:
            self.build(X.shape)
        assert X.shape[1] == self.in_features

        W, b = self.parameters["W"], self.parameters.get("b", None)
        Z = linear(X, W, b)
        assert Z.shape == (X.shape[0], self.out_features)

        A = self.act_fn(Z)
        assert A.shape == (X.shape[0], self.out_features)

        return A

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation,
            "weight_initializer": self.weight_initializer,
            "bias_initializer": self.bias_initializer,
            "include_bias": self.include_bias,
        }


def _same_padding(in_size: int, kernel_size: int, stride: int) -> int:
    """Padding per side so that out_size = ceil(in_size / stride)."""
    out_size = (in_size + stride - 1) // stride
    total = (out_size - 1) * stride + kernel_size - in_size
    return max(0, (total + 1) // 2)


class Conv2d(Module):
    """2D Convolution Layer: output = activation(X * K + B)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        activation: str = "identity",
        kernel_initializer: str | dict[str, Any] = "xavier_normal",
        bias_initializer: str | dict[str, Any] = "zeros",
        include_bias: bool = True,
        name: str = "Conv2D",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_pair(kernel_size)
        self.stride = make_pair(stride)
        self._padding_spec = padding  # int, (int,int), "valid", or "same"
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.include_bias = include_bias
        self.name = name

        self.act_fn = activation_factory(self.activation)
        self.kernel_init = initializer_factory(self.kernel_initializer)
        if self.include_bias:
            self.bias_init = initializer_factory(self.bias_initializer)

        # Unknown until build() sees an input; the spatial dimensions determine them.
        self.input_shape: Shape | None = None
        self.output_shape: Shape | None = None
        self.kernel_shape: Shape | None = None
        self._padding: tuple[int, int] = (0, 0)  # resolved in build()

    def _resolve_padding(self, in_h: int, in_w: int) -> tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        p = self._padding_spec
        if isinstance(p, str):
            if p.lower() == "valid":
                return (0, 0)
            if p.lower() == "same":
                ph = _same_padding(in_h, kh, sh)
                pw = _same_padding(in_w, kw, sw)
                return (ph, pw)
            raise ValueError("padding must be int, (int,int), 'valid', or 'same'")
        if isinstance(p, int):
            return (p, p)
        return make_pair(p)

    def build(self, input_shape: Shape) -> None:
        assert len(input_shape) == 4
        _, in_ch, in_h, in_w = input_shape
        assert in_ch == self.in_channels

        kh, kw = self.kernel_size
        sh, sw = self.stride
        self._padding = self._resolve_padding(in_h, in_w)
        ph, pw = self._padding

        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1

        self.input_shape = (in_ch, in_h, in_w)
        self.output_shape = (self.out_channels, out_h, out_w)
        self.kernel_shape = (self.out_channels, self.in_channels, kh, kw)
        # One bias per output channel, broadcast across every spatial position. The
        # trailing 1s make it broadcast against (batch, out_ch, out_h, out_w) directly.
        self.bias_shape: Shape = (self.out_channels, 1, 1)

        K = self.register_parameter("K", self.kernel_init(self.kernel_shape))
        assert K.shape == self.kernel_shape

        if self.include_bias:
            B = self.register_parameter("B", self.bias_init(self.bias_shape))
            assert B.shape == self.bias_shape

        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        if not self.initialized:
            self.build(X.shape)
        assert X.shape[1:] == self.input_shape

        K = self.parameters["K"]
        B = self.parameters.get("B", None)
        output = conv2d(X, K, B, stride=self.stride, padding=self._padding)
        return self.act_fn(output)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "input_shape": self.input_shape,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self._padding_spec,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "include_bias": self.include_bias,
        }


class Flatten(Module):
    def __init__(self, name: str = "Flatten") -> None:
        super().__init__()
        self.name = name

    def forward(self, X: Tensor) -> Tensor:
        return flatten(X)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {}


class Activation(Module):
    def __init__(
        self, activation: str | dict[str, Any], name: str = "Activation"
    ) -> None:
        super().__init__()
        self.activation = activation
        self.name = name

        self.act_fn = activation_factory(activation)

    def forward(self, X: Tensor) -> Tensor:
        return self.act_fn(X)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"activation": self.activation}

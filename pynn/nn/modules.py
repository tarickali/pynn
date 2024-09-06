from typing import Any

from pynn.core.types import Shape
from pynn.core import Tensor, Module
from pynn.functional.modules import *
from pynn.nn.factories import activation_factory, initializer_factory

__all__ = ["Linear", "Conv2d", "Flatten", "Activation"]


class Linear(Module):
    def __init__(
        self,
        output_dim: int,
        activation: str | dict[str, Any] = "identity",
        weight_initializer: str | dict[str, Any] = "xavier_normal",
        bias_initializer: str | dict[str, Any] = "zeros",
        include_bias: bool = True,
        name: str = "Linear",
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.include_bias = include_bias
        self.name = name

        self.act_fn = activation_factory(self.activation)
        self.weight_init = initializer_factory(self.weight_initializer)
        if self.include_bias:
            self.bias_init = initializer_factory(self.bias_initializer)

        # Set input_dim to None until parameters are initialized
        self.input_dim = None

    def build(self, input_shape: int | Shape) -> None:
        assert len(input_shape) == 2
        self.input_dim = input_shape[1]

        self.parameters["W"] = Tensor(
            self.weight_init((self.input_dim, self.output_dim))
        )
        assert self.parameters["W"].shape == (self.input_dim, self.output_dim)

        if self.include_bias:
            self.parameters["b"] = Tensor(self.bias_init((self.output_dim,)))
            assert self.parameters["b"].shape == (self.output_dim,)

        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        if not self.initialized:
            self.build(X.shape)
        assert X.shape[1] == self.input_dim

        # Get weights and bias
        W, b = self.parameters["W"], self.parameters.get("b", None)

        # Compute linear transformation
        Z = linear(X, W, b)

        assert Z.shape == (X.shape[0], self.output_dim)

        # Compute activation
        A = self.act_fn(Z)
        assert A.shape == (X.shape[0], self.output_dim)

        return A

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "weight_initializer": self.weight_initializer,
            "bias_initializer": self.bias_initializer,
            "include_bias": self.include_bias,
        }


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        # stride: int | tuple[int, int] = 1,
        # padding: str = "valid",
        activation: str = "identity",
        kernel_initializer: str | dict[str, Any] = "xavier_normal",
        bias_initializer: str | dict[str, Any] = "zeros",
        include_bias: bool = True,
        name: str = "Conv2D",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.include_bias = include_bias
        self.name = name

        self.act_fn = activation_factory(self.activation)
        self.kernel_init = initializer_factory(self.kernel_initializer)
        if self.include_bias:
            self.bias_init = initializer_factory(self.bias_initializer)

        # Set input, output, kernel shapes to None until parameters are initialized
        self.input_shape = None
        self.output_shape = None
        self.kernel_shape = None

    def build(self, input_shape: int | Shape) -> None:
        assert len(input_shape) == 4
        _, in_channels, in_height, in_width = input_shape
        assert in_channels == self.in_channels

        self.input_shape = (in_channels, in_height, in_width)
        self.output_shape = (
            self.out_channels,
            in_height - self.kernel_size + 1,
            in_width - self.kernel_size + 1,
        )
        self.kernel_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        self.parameters["K"] = Tensor(self.kernel_init(self.kernel_shape))
        assert self.parameters["K"].shape == self.kernel_shape

        if self.include_bias:
            self.parameters["B"] = Tensor(self.bias_init(self.output_shape))
            assert self.parameters["B"].shape == self.output_shape

        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        if not self.initialized:
            self.build(X.shape)
        assert X.shape[1:] == self.input_shape

        # Get kernel and bias
        K = self.parameters["K"]
        B = self.parameters.get("B", None)

        # Compute conv2d transformation
        output = conv2d(X, K, B)

        # Compute activation
        output = self.act_fn(output)

        return output

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "input_shape": self.input_shape,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
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

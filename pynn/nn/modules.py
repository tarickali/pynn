from typing import Any, cast

import numpy as np

from pynn.core import Module, Tensor
from pynn.core.types import Shape
from pynn.functional.modules import (
    avg_pool2d,
    batch_norm,
    conv2d,
    dropout,
    flatten,
    layer_norm,
    linear,
    max_pool2d,
    unflatten,
)
from pynn.nn.factories import activation_factory, initializer_factory
from pynn.utils.array import make_pair

__all__ = [
    "Activation",
    "AvgPool2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "Dropout",
    "Flatten",
    "Identity",
    "LayerNorm",
    "Linear",
    "MaxPool2d",
    "Unflatten",
]


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
    """Collapse every axis past the batch axis into one."""

    def __init__(self, name: str = "Flatten") -> None:
        super().__init__()
        self.name = name

    def forward(self, X: Tensor) -> Tensor:
        return flatten(X)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {}


class Unflatten(Module):
    """Inverse of `Flatten`: split the flattened axis back into `shape`.

    Takes the trailing shape of *one example*, not the whole output shape — the batch
    axis is read from the input:

    ```python
    Sequential([Linear(64, 50), Unflatten(2, 5, 5), Conv2d(2, 4, 3)])
    ```

    That is deliberately narrower than a general `Reshape` layer, which this library
    does not have. A reshape whose target includes the batch size is right for every
    batch of an epoch except the last, smaller one, and the symptom is a shape error
    partway through the first epoch rather than at the line that caused it. Anything
    genuinely needing the general form has `Tensor.reshape`, which is differentiable.

    Parameters
    ----------
    *shape : int | tuple[int, ...]
        The trailing axes, as `Unflatten(2, 5, 5)` or `Unflatten((2, 5, 5))`. One entry
        may be `-1` and is inferred.
    name : str, default "Unflatten"
    """

    def __init__(self, *shape: int | tuple[int, ...], name: str = "Unflatten") -> None:
        super().__init__()
        if len(shape) == 1 and isinstance(shape[0], tuple):
            resolved = shape[0]
        else:
            resolved = cast("tuple[int, ...]", shape)
        self.shape = resolved
        self.name = name

    def forward(self, X: Tensor) -> Tensor:
        return unflatten(X, self.shape)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"shape": self.shape}


class Activation(Module):
    """A `Module` wrapping one of the stateless activations, so a container can hold it.

    `pynn.functional`'s activations are functions and `pynn.nn.activations`' are
    stateless `Activation` objects (`pynn.core.Activation`, a different class from this
    one); neither is a `Module`, so neither can go into a `Sequential`. This is the
    adapter, and `activation="relu"` on a layer goes through the same factory.
    """

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


class Identity(Activation):
    """A layer that returns its input unchanged.

    The placeholder: `BatchNorm2d() if normalize else Identity()` keeps a `Sequential`
    the same length either way, so an ablation changes one line rather than the shape
    of the model — which is what `nn.Identity` is for in PyTorch too.

    Note the pair, since the name appears twice. This is the **layer**, and it is what a
    container holds. `pynn.nn.activations.Identity` is the stateless `Activation` that
    `activation_factory("identity")` returns and that a layer's `activation=` argument
    binds to; it is not a `Module` and cannot go into a `Sequential`. Only this one is
    exported from `pynn.nn`.
    """

    def __init__(self, name: str = "Identity") -> None:
        super().__init__("identity", name=name)


class Dropout(Module):
    """Randomly zero a fraction of the input during training.

    Active only in training mode: `Module.train()` / `Module.eval()` set the flag this
    layer reads, which is the canonical reason a model needs those modes at all.
    Evaluating a model that was left in training mode gives a different answer every
    call, and nothing raises.
    """

    def __init__(self, p: float = 0.5, name: str = "Dropout") -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.name = name

    def forward(self, X: Tensor) -> Tensor:
        return dropout(X, self.p, training=self.training)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"p": self.p}


class LayerNorm(Module):
    """Normalize each example over its trailing axes, then scale and shift.

    Shape can be given or inferred: `LayerNorm(10)` normalizes over a trailing axis of
    length 10, and `LayerNorm()` takes the last axis of whatever it first sees.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | None = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        name: str = "LayerNorm",
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: tuple[int, ...] | None = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.name = name

    def build(self, input_shape: Shape) -> None:
        if self.normalized_shape is None:
            self.normalized_shape = (input_shape[-1],)
        if self.elementwise_affine:
            self.register_parameter("gamma", np.ones(self.normalized_shape))
            self.register_parameter("beta", np.zeros(self.normalized_shape))
        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        if not self.initialized:
            self.build(X.shape)
        return layer_norm(
            X,
            self.parameters.get("gamma"),
            self.parameters.get("beta"),
            self.normalized_shape,
            self.eps,
        )

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "normalized_shape": self.normalized_shape,
            "eps": self.eps,
            "elementwise_affine": self.elementwise_affine,
        }


class _BatchNorm(Module):
    """Shared implementation of BatchNorm1d and BatchNorm2d.

    The two differ only in the rank of the input they accept; the statistics are
    per-channel either way, taken over every axis except axis 1.
    """

    #: Accepted input ranks, checked so that feeding (N, C, H, W) to BatchNorm1d fails
    #: with a shape error rather than normalizing over the wrong axes.
    ranks: tuple[int, ...] = ()

    def __init__(
        self,
        num_features: int | None = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if name is not None:
            self.name = name

    def build(self, input_shape: Shape) -> None:
        if self.num_features is None:
            self.num_features = input_shape[1]
        elif input_shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features, got {input_shape[1]}"
            )

        # Shaped to broadcast against (batch, channels, ...) along axis 1.
        shape = (self.num_features, *([1] * (len(input_shape) - 2)))
        if self.affine:
            self.register_parameter("gamma", np.ones(shape))
            self.register_parameter("beta", np.zeros(shape))
        if self.track_running_stats:
            self.register_buffer("running_mean", np.zeros(self.num_features))
            self.register_buffer("running_var", np.ones(self.num_features))
        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        if self.ranks and X.ndim not in self.ranks:
            ranks = " or ".join(str(rank) for rank in self.ranks)
            raise ValueError(
                f"{self.name} expects an input of rank {ranks}, got shape {X.shape}"
            )
        if not self.initialized:
            self.build(X.shape)

        return batch_norm(
            X,
            self.parameters.get("gamma"),
            self.parameters.get("beta"),
            self._buffers.get("running_mean"),
            self._buffers.get("running_var"),
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "track_running_stats": self.track_running_stats,
        }


class BatchNorm1d(_BatchNorm):
    """Batch normalization for (batch, features) or (batch, features, length) input."""

    ranks = (2, 3)

    def __init__(self, *args: Any, name: str = "BatchNorm1d", **kwargs: Any) -> None:
        super().__init__(*args, name=name, **kwargs)


class BatchNorm2d(_BatchNorm):
    """Batch normalization for (batch, channels, height, width) input."""

    ranks = (4,)

    def __init__(self, *args: Any, name: str = "BatchNorm2d", **kwargs: Any) -> None:
        super().__init__(*args, name=name, **kwargs)


class _Pool2d(Module):
    """Shared shape handling for the pooling layers."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = make_pair(kernel_size)
        # PyTorch's default: non-overlapping windows.
        self.stride = self.kernel_size if stride is None else make_pair(stride)
        self.padding = make_pair(padding)
        if name is not None:
            self.name = name

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
        }


class MaxPool2d(_Pool2d):
    """Take the maximum over each sliding window, per channel."""

    def __init__(self, *args: Any, name: str = "MaxPool2d", **kwargs: Any) -> None:
        super().__init__(*args, name=name, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return max_pool2d(X, self.kernel_size, self.stride, self.padding)


class AvgPool2d(_Pool2d):
    """Average over each sliding window, per channel."""

    def __init__(self, *args: Any, name: str = "AvgPool2d", **kwargs: Any) -> None:
        super().__init__(*args, name=name, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return avg_pool2d(X, self.kernel_size, self.stride, self.padding)

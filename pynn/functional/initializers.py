"""Weight initialization functions.

The random initializers draw from the generator in `pynn.core.random`, shared with the
stochastic layers, rather than from the legacy global `numpy.random` functions. Call
`set_seed` for a reproducible run, or pass an explicit `rng` to a single initializer.
"""

import numpy as np

from pynn.core import Tensor
from pynn.core.random import default_rng, set_seed
from pynn.core.types import Number, Shape

__all__ = [
    "constant",
    "fans",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "ones",
    "random_normal",
    "random_uniform",
    "set_seed",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]


def fans(shape: Shape) -> tuple[int, int]:
    """The number of inputs and outputs a weight of this shape connects.

    The scale every variance-scaling initializer picks is a function of these two
    numbers, and reading them off the wrong axes is invisible: the weights still have
    a plausible magnitude, the model still trains, it just diverges at a learning rate
    a correctly initialized one handles.

    The library has two weight layouts, so the rank decides which is meant:

    - a `Linear` weight is `(in_features, out_features)`;
    - a `Conv2d` kernel is `(out_channels, in_channels, *kernel_size)`, where each
      output unit reads `in_channels * prod(kernel_size)` values — not
      `out_channels`, and not one value per input channel.

    Taking `shape[0]` for both makes a convolution's fan-in its *output* channel count
    and drops the receptive field entirely. For `Conv2d(16, 32, 3)` that is a standard
    deviation 2.1x too wide.

    Examples
    --------
    >>> fans((784, 256))            # Linear(784, 256)
    (784, 256)
    >>> fans((32, 16, 3, 3))        # Conv2d(16, 32, kernel_size=3)
    (144, 288)
    """
    if len(shape) < 2:
        size = int(np.prod(shape)) if shape else 1
        return size, size
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive_field = int(np.prod(shape[2:]))
    return shape[1] * receptive_field, shape[0] * receptive_field


def zeros(shape: Shape) -> Tensor:
    return Tensor(np.zeros(shape))


def ones(shape: Shape) -> Tensor:
    return Tensor(np.ones(shape))


def constant(shape: Shape, value: Number) -> Tensor:
    return Tensor(np.full(shape, value))


def random_uniform(
    shape: Shape,
    low: float = 0.0,
    high: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tensor:
    return Tensor(default_rng(rng).uniform(low, high, shape))


def random_normal(
    shape: Shape,
    mean: float = 0.0,
    std: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tensor:
    return Tensor(default_rng(rng).normal(mean, std, shape))


def xavier_uniform(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    fan_in, fan_out = fans(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return Tensor(default_rng(rng).uniform(-limit, limit, shape))


def xavier_normal(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    fan_in, fan_out = fans(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return Tensor(default_rng(rng).normal(0.0, std, shape))


def he_uniform(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    fan_in, _ = fans(shape)
    limit = np.sqrt(6.0 / fan_in)
    return Tensor(default_rng(rng).uniform(-limit, limit, shape))


def he_normal(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    fan_in, _ = fans(shape)
    std = np.sqrt(2.0 / fan_in)
    return Tensor(default_rng(rng).normal(0.0, std, shape))


def lecun_uniform(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    fan_in, _ = fans(shape)
    limit = np.sqrt(3.0 / fan_in)
    return Tensor(default_rng(rng).uniform(-limit, limit, shape))


def lecun_normal(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    fan_in, _ = fans(shape)
    std = np.sqrt(1.0 / fan_in)
    return Tensor(default_rng(rng).normal(0.0, std, shape))

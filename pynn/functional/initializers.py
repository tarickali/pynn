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


def zeros(shape: Shape) -> Tensor:
    return Tensor(np.zeros(shape))


def ones(shape: Shape) -> Tensor:
    return Tensor(np.ones(shape))


def constant(shape: Shape, value: Number) -> Tensor:
    return Tensor(np.full(shape, value))


def random_uniform(
    shape: Shape,
    low: Number = 0.0,
    high: Number = 1.0,
    rng: np.random.Generator | None = None,
) -> Tensor:
    return Tensor(default_rng(rng).uniform(low, high, shape))


def random_normal(
    shape: Shape,
    mean: Number = 0.0,
    std: Number = 1.0,
    rng: np.random.Generator | None = None,
) -> Tensor:
    return Tensor(default_rng(rng).normal(mean, std, shape))


def xavier_uniform(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return Tensor(default_rng(rng).uniform(-limit, limit, shape))


def xavier_normal(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    std = np.sqrt(2.0 / (shape[0] + shape[1]))
    return Tensor(default_rng(rng).normal(0.0, std, shape))


def he_uniform(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    limit = np.sqrt(6.0 / shape[0])
    return Tensor(default_rng(rng).uniform(-limit, limit, shape))


def he_normal(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    std = np.sqrt(2.0 / shape[0])
    return Tensor(default_rng(rng).normal(0.0, std, shape))


def lecun_uniform(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    limit = np.sqrt(3.0 / shape[0])
    return Tensor(default_rng(rng).uniform(-limit, limit, shape))


def lecun_normal(shape: Shape, rng: np.random.Generator | None = None) -> Tensor:
    std = np.sqrt(1.0 / shape[0])
    return Tensor(default_rng(rng).normal(0.0, std, shape))

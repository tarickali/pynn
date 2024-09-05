import numpy as np
from pynn.core.types import Number, Shape
from pynn.core import Tensor

__all__ = [
    "zeros",
    "ones",
    "constant",
    "random_uniform",
    "random_normal",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_uniform",
    "lecun_normal",
]


def zeros(shape: Shape) -> Tensor:
    return Tensor(np.zeros(shape))


def ones(shape: Shape) -> Tensor:
    return Tensor(np.ones(shape))


def constant(shape: Shape, value: Number) -> Tensor:
    return Tensor(np.full(shape, value))


def random_uniform(shape: Shape, low: Number = 0.0, high: Number = 1.0) -> Tensor:
    return Tensor(np.random.uniform(low, high, shape))


def random_normal(shape: Shape, mean: Number = 0.0, std: Number = 1.0) -> Tensor:
    return Tensor(np.random.normal(mean, std, shape))


def xavier_uniform(shape: Shape) -> Tensor:
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return Tensor(np.random.uniform(-limit, limit, shape))


def xavier_normal(shape: Shape) -> Tensor:
    std = np.sqrt(2.0 / (shape[0] + shape[1]))
    return Tensor(np.random.normal(0.0, std, shape))


def he_uniform(shape: Shape) -> Tensor:
    limit = np.sqrt(6.0 / shape[0])
    return Tensor(np.random.uniform(-limit, limit, shape))


def he_normal(shape: Shape) -> Tensor:
    std = np.sqrt(2.0 / shape[0])
    return Tensor(np.random.normal(0.0, std, shape))


def lecun_uniform(shape: Shape) -> Tensor:
    limit = np.sqrt(3.0 / shape[0])
    return Tensor(np.random.uniform(-limit, limit, shape))


def lecun_normal(shape: Shape) -> Tensor:
    std = np.sqrt(1.0 / shape[0])
    return Tensor(np.random.normal(0.0, std, shape))

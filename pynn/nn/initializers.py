from pynn.core.types import Shape
from pynn.core import Tensor, Initializer
from pynn.functional.initializers import *

__all__ = [
    "Zeros",
    "Ones",
    "Constant",
    "RandomUniform",
    "RandomNormal",
    "XavierUniform",
    "XavierNormal",
    "HeUniform",
    "HeNormal",
    "LecunUniform",
    "LecunNormal",
]


class Zeros(Initializer):
    """Zeros Initializer

    Initializes a Tensor with zeros.

    """

    def init(self, shape: Shape) -> Tensor:
        return zeros(shape)


class Ones(Initializer):
    """Ones Initializer

    Initializes a Tensor with ones.

    """

    def init(self, shape: Shape) -> Tensor:
        return ones(shape)


class Constant(Initializer):
    """Constant Initializer

    Initializes a Tensor with a constant value.

    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def init(self, shape: Shape) -> Tensor:
        return constant(shape, self.value)


class RandomUniform(Initializer):
    """RandomUniform Initializer

    Initializes a Tensor with values sampled from `U(low, high)` where
    `low` and `high` are the given low and high range values for the
    distribution, respectively.

    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def init(self, shape: Shape) -> Tensor:
        return random_uniform(shape, self.low, self.high)


class RandomNormal(Initializer):
    """RandomNormal Initializer

    Initializes a Tensor with values sampled from `N(mu, std)` where
    `mu` and `std` are the given mean and standard deviation, respectively.

    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def init(self, shape: Shape) -> Tensor:
        return random_normal(shape, self.mean, self.std)


class XavierUniform(Initializer):
    """XavierUniform Initializer

    Initializes a Tensor with values sampled from `U(-limit, limit)` with
    `limit = sqrt(6.0 / (fan_in + fan_out))` where `fan_in` and `fan_out`
    are the number of input and output units to the Tensor, respectively.

    """

    def init(self, shape: Shape) -> Tensor:
        return xavier_uniform(shape)


class XavierNormal(Initializer):
    """XavierNormal Initializer

    Initializes a Tensor with values sampled from `N(0.0, std)` with
    `std = sqrt(2.0 / (fan_in + fan_out))` where `fan_in` and `fan_out`
    are the number of input and output units to the Tensor, respectively.

    """

    def init(self, shape: Shape) -> Tensor:
        return xavier_normal(shape)


class HeUniform(Initializer):
    """HeUniform Initalizer

    Initializes a Tensor with values sampled from `U(-limit, limit)` with
    `limit = sqrt(6.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape) -> Tensor:
        return he_uniform(shape)


class HeNormal(Initializer):
    """HeNormal Initalizer

    Initializes a Tensor with values sampled from `N(0.0, std)` with
    `std = sqrt(2.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape) -> Tensor:
        return he_normal(shape)


class LecunUniform(Initializer):
    """LecunUniform Initializer

    Initializes a Tensor with values sampled from `U(-limit, limit)` with
    `limit = sqrt(3.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape) -> Tensor:
        return lecun_uniform(shape)


class LecunNormal(Initializer):
    """LecunNormal Initializer

    Initializes a Tensor with values sampled from `N(0.0, std)` with
    `std = sqrt(1.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape) -> Tensor:
        return lecun_normal(shape)

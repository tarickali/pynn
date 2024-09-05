from pynn.core import Tensor, Activation
from pynn.functional.activations import *

__all__ = [
    "Identity",
    "Affine",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "ELU",
    "SELU",
    "SoftPlus",
    "Softmax",
]


class Identity(Activation):
    """Identity Activation

    Computes the elementwise function `f(x) = x`.

    """

    def func(self, x: Tensor) -> Tensor:
        return identity(x)


class Affine(Activation):
    """Affine Activation.

    Parameterized by `slope` [float] and `intercept` [float], computes the
    function `f(x) = slope * x + intercept`.

    Parameters
    ----------
    slope : float
    intercept : float

    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def func(self, x: Tensor) -> Tensor:
        return affine(x, self.slope, self.intercept)


class ReLU(Activation):
    """ReLU Activation

    Parameterized by `alpha` [float], computes the function
    ```
    f(x) = {
        x : x >= 0,
        alpha * x : x < 0
    }
    ```.

    """

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: Tensor) -> Tensor:
        return relu(x, self.alpha)


class Sigmoid(Activation):
    """Sigmoid Activation

    Computes the function `f(x) = 1 / (1 + exp(-x))`.

    """

    def func(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Activation):
    """Tanh Activation

    Computes the function `f(x) = tanh(x)`.

    """

    def func(self, x: Tensor) -> Tensor:
        return tanh(x)


class ELU(Activation):
    """ELU Activation

    Parameterized by `alpha` [float], computes the function
    ```
    f(x) = {
        x : x >= 0,
        alpha * (exp(x) - 1) : x < 0
    }
    ```.

    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: Tensor) -> Tensor:
        return elu(x, self.alpha)


class SELU(Activation):
    """SELU Activation

    Computes the function
    ```
    f(x) = {
        SCALE * x : x >= 0,
        SCALE * ALPHA * (exp(x) - 1) : x < 0
    }
    ```
    where
    SCALE = 1.0507009873554804934193349852946,
    ALPHA = 1.6732632423543772848170429916717

    """

    def func(self, x: Tensor) -> Tensor:
        return selu(x)


class SoftPlus(Activation):
    """SoftPlus Activation

    Computes the function `f(x) = log(1 + exp(x))`.

    """

    def func(self, x: Tensor) -> Tensor:
        return softplus(x)


class Softmax(Activation):
    """Softmax Activation

    Computes the function `f(x) = exp(x) / sum(exp(x))`.

    NOTE: Returns the all ones matrix with shape of input z,
    since the categorical cross-entropy loss function L computes
    the appropriate gradient of L with respect to z.

    NOTE: The true gradient of softmax with respect to z is a Jacobian,
    and the code is given below:
    s = softmax(z)
    jacob = np.diag(s.flatten()) - np.outer(s, s)

    NOTE: It is important to note that this choice limits the use of
    the softmax activation to only the last layer of a neural network.

    """

    def func(self, x: Tensor) -> Tensor:
        return softmax(x)

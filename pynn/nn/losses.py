from pynn.core import Tensor, Loss
from pynn.functional.losses import *

__all__ = [
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "MeanSquaredError",
    "MeanAbsoluteError",
]


class BinaryCrossentropy(Loss):
    """BinaryCrossentropy Loss

    Computes the crossentropy loss between binary arrays true and pred
    given by: `-mean(true * log(pred) + (1 - true) * log(1 - pred))`.

    NOTE: This `Loss` can be used when pred are unactivated (logits) or
    are activated.

    """

    def __init__(self, logits: bool = True) -> None:
        super().__init__()
        self.logits = logits

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return binary_crossentropy(true, pred, self.logits)


class CategoricalCrossentropy(Loss):
    """CategoricalCrossentropy Loss

    Computes the crossentropy loss between multiclass arrays true and pred
    given by: `-mean(true * log(pred))`.

    NOTE: This `Loss` can be used when pred are unactivated (logits) or
    are activated.

    """

    def __init__(self, logits: bool = True) -> None:
        super().__init__()
        self.logits = logits

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return categorical_crossentropy(true, pred, self.logits)


class MeanSquaredError(Loss):
    """MeanSquaredError Loss

    Computes the squared error between true and pred given by:
    `mean((true - pred)**2)`

    """

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return mean_squared_error(true, pred)


class MeanAbsoluteError(Loss):
    """MeanAbsoluteError Loss

    Computes the squared error between true and pred given by:
    `mean((true - pred)**2)`

    """

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return mean_absolute_error(true, pred)

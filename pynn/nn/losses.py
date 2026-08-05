from typing import Literal

from pynn.core import Loss, Tensor
from pynn.functional.losses import (
    binary_crossentropy,
    categorical_crossentropy,
    mean_absolute_error,
    mean_squared_error,
)

__all__ = [
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "MeanAbsoluteError",
    "MeanSquaredError",
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

    Computes the squared error between true and pred:
    - reduction='mean' (default): `mean((true - pred)**2)`
    - reduction='sum': `sum((true - pred)**2)`

    """

    def __init__(self, reduction: Literal["mean", "sum"] = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return mean_squared_error(true, pred, self.reduction)


class MeanAbsoluteError(Loss):
    """MeanAbsoluteError Loss

    Computes the mean absolute error between true and pred given by:
    `mean(|true - pred|)`.

    """

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return mean_absolute_error(true, pred)

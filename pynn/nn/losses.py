from pynn.core import Loss, Tensor
from pynn.core.types import Array
from pynn.functional.losses import (
    Reduction,
    binary_crossentropy,
    categorical_crossentropy,
    huber,
    mean_absolute_error,
    mean_squared_error,
    sparse_categorical_crossentropy,
)

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "CrossEntropyLoss",
    "HuberLoss",
    "L1Loss",
    "MSELoss",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "SmoothL1Loss",
    "SparseCategoricalCrossentropy",
]


class BinaryCrossentropy(Loss):
    """BinaryCrossentropy Loss

    Computes the crossentropy loss between binary arrays true and pred
    given by: `-mean(true * log(pred) + (1 - true) * log(1 - pred))`.

    NOTE: This `Loss` can be used when pred are unactivated (logits) or
    are activated.

    """

    def __init__(self, logits: bool = True, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.logits = logits
        self.reduction = reduction

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return binary_crossentropy(true, pred, self.logits, self.reduction)


class BCEWithLogitsLoss(BinaryCrossentropy):
    """Binary cross-entropy over logits, with the sigmoid fused in.

    Named for what it takes rather than what it computes, matching PyTorch. Prefer it
    over composing `Sigmoid` with `BCELoss`: the fused form never evaluates `log(0)`,
    and its gradient with respect to the logits is just `sigmoid(z) - y`.
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__(logits=True, reduction=reduction)


class BCELoss(BinaryCrossentropy):
    """Binary cross-entropy over probabilities.

    Takes values already through a sigmoid, matching PyTorch's `BCELoss`. If you have
    logits, use `BCEWithLogitsLoss` instead — it is the stabler path, not merely a
    convenience.
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__(logits=False, reduction=reduction)


class CategoricalCrossentropy(Loss):
    """CategoricalCrossentropy Loss

    Computes the crossentropy loss between multiclass arrays true and pred
    given by: `-mean(true * log(pred))`.

    NOTE: This `Loss` can be used when pred are unactivated (logits) or
    are activated.

    """

    def __init__(self, logits: bool = True, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.logits = logits
        self.reduction = reduction

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return categorical_crossentropy(true, pred, self.logits, self.reduction)


class SparseCategoricalCrossentropy(Loss):
    """Categorical cross-entropy taking integer labels rather than one-hot targets.

    Identical to `CategoricalCrossentropy` in value and gradient; it just skips the
    one-hot encoding, which for a large number of classes is most of the memory and
    most of the arithmetic.
    """

    def __init__(self, logits: bool = True, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.logits = logits
        self.reduction = reduction

    def compute(self, true: Tensor | Array, pred: Tensor) -> Tensor:
        return sparse_categorical_crossentropy(true, pred, self.logits, self.reduction)


class MeanSquaredError(Loss):
    """MeanSquaredError Loss

    Computes the squared error between true and pred:
    - reduction='mean' (default): `mean((true - pred)**2)`
    - reduction='sum': `sum((true - pred)**2)`
    - reduction='none': the per-element squared errors

    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return mean_squared_error(true, pred, self.reduction)


class MeanAbsoluteError(Loss):
    """MeanAbsoluteError Loss

    Computes the mean absolute error between true and pred given by:
    `mean(|true - pred|)`.

    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return mean_absolute_error(true, pred, self.reduction)


class HuberLoss(Loss):
    """Huber Loss

    Squared error where the residual is smaller than `delta`, absolute error beyond it.
    Squared error lets one outlier dominate a batch; absolute error does not, but its
    gradient never shrinks as the fit improves. This is the compromise.

    Parameters
    ----------
    delta : float, default 1.0
        Residual magnitude at which the loss switches from quadratic to linear.
    reduction : Reduction, default "mean"

    """

    def __init__(self, delta: float = 1.0, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def compute(self, true: Tensor, pred: Tensor) -> Tensor:
        return huber(true, pred, self.delta, self.reduction)


#: PyTorch's name for the same loss. Its `SmoothL1Loss` divides the quadratic region by
#: `beta` where `HuberLoss` multiplies the linear region by `delta`; the two differ by a
#: factor of `delta`, so this is an alias rather than a separate class only because the
#: default `delta=1.0` makes them identical.
SmoothL1Loss = HuberLoss

# PyTorch-style aliases for the losses whose names already match.
CrossEntropyLoss = CategoricalCrossentropy
MSELoss = MeanSquaredError
L1Loss = MeanAbsoluteError

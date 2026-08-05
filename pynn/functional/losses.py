from typing import Literal

import numpy as np

from pynn.core.constants import EPSILON
from pynn.core import Tensor
from pynn.core.numeric import stable_sigmoid
import pynn.core.math as pmath

__all__ = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "mean_squared_error",
    "mean_absolute_error",
]


def _check_same_shape(true: Tensor, pred: Tensor) -> None:
    if true.shape != pred.shape:
        raise ValueError(
            f"true and pred must have the same shape, got {true.shape} and {pred.shape}"
        )


def binary_crossentropy(true: Tensor, pred: Tensor, logits: bool = True) -> Tensor:
    """Binary cross-entropy, averaged over every element.

    Parameters
    ----------
    true : Tensor
        Targets in [0, 1], same shape as ``pred``.
    pred : Tensor
        Logits when ``logits`` is True, otherwise probabilities.
    logits : bool, default True
        When True, sigmoid is fused into this loss. The fused form is both more
        numerically stable than composing ``sigmoid`` with this loss (it never
        evaluates ``log(0)``) and cheaper, since ``dL/dlogits`` simplifies to
        ``(sigmoid(z) - y) / n``.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    _check_same_shape(true, pred)

    true_arr = true.data

    if logits:
        z = pred.data
        # log(1 + exp(-|z|)) + max(z, 0) - z*y is the overflow-free rearrangement of
        # -[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))].
        elementwise = np.maximum(z, 0) - z * true_arr + np.log1p(np.exp(-np.abs(z)))
        data = np.mean(elementwise)
        probabilities = stable_sigmoid(z)

        def gradient() -> np.ndarray:
            return (probabilities - true_arr) / true_arr.size
    else:
        p = np.clip(pred.data, EPSILON, 1.0 - EPSILON)
        data = -np.mean(true_arr * np.log(p) + (1 - true_arr) * np.log(1 - p))

        def gradient() -> np.ndarray:
            return (p - true_arr) / (p * (1 - p) * true_arr.size)

    output = Tensor(data)
    output.add_children((pred,))

    def reverse():
        pred.grad += gradient() * output.grad

    output.forward = "binary_crossentropy"
    output.reverse = reverse

    return output


def categorical_crossentropy(true: Tensor, pred: Tensor, logits: bool = True) -> Tensor:
    """Categorical cross-entropy, summed over classes and averaged over the batch.

    Parameters
    ----------
    true : Tensor
        One-hot (or otherwise normalized) targets of shape (batch, ..., classes).
    pred : Tensor
        Logits when ``logits`` is True, otherwise probabilities.
    logits : bool, default True
        When True, softmax is fused into this loss so that ``dL/dlogits`` is the
        familiar ``(softmax(z) - y) / batch``. Fusing matters here: applying that
        expression to a separate softmax node's output would send a logit-space
        gradient back through the softmax Jacobian a second time.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    _check_same_shape(true, pred)

    true_arr = true.data
    batch_size = pred.shape[0]

    if logits:
        z = pred.data
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        probabilities = e / np.sum(e, axis=-1, keepdims=True)
        data = (
            -np.sum(true_arr * np.log(np.maximum(probabilities, EPSILON))) / batch_size
        )

        def gradient() -> np.ndarray:
            return (probabilities - true_arr) / batch_size
    else:
        p = np.clip(pred.data, EPSILON, 1.0)
        data = -np.sum(true_arr * np.log(p)) / batch_size

        def gradient() -> np.ndarray:
            return -true_arr / (p * batch_size)

    output = Tensor(data)
    output.add_children((pred,))

    def reverse():
        pred.grad += gradient() * output.grad

    output.forward = "categorical_crossentropy"
    output.reverse = reverse

    return output


def mean_squared_error(
    true: Tensor, pred: Tensor, reduction: Literal["mean", "sum"] = "mean"
) -> Tensor:
    """MSE loss. reduction='mean' (default) or 'sum'."""
    _check_same_shape(true, pred)
    diff_sq = (true - pred) ** 2
    if reduction == "mean":
        return pmath.mean(diff_sq)
    if reduction == "sum":
        return pmath.sum(diff_sq)
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def mean_absolute_error(true: Tensor, pred: Tensor) -> Tensor:
    _check_same_shape(true, pred)
    return pmath.mean(pmath.abs(true - pred))

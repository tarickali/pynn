from typing import Literal

import numpy as np

from pynn.core.constants import EPSILON
from pynn.core import Tensor
from pynn.functional import sigmoid, softmax
import pynn.core.math as pmath

__all__ = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "mean_squared_error",
    "mean_absolute_error",
]


def binary_crossentropy(true: Tensor, pred: Tensor, logits: bool = True) -> Tensor:
    assert true.shape == pred.shape
    if logits:
        pred = sigmoid(pred)

    pred_arr = np.clip(pred.data, EPSILON, 1.0 - EPSILON)
    true_arr = true.data

    data = -np.mean(true_arr * np.log(pred_arr) + (1 - true_arr) * np.log(1 - pred_arr))
    output = Tensor(data)
    output.add_children((pred,))

    def reverse():
        grad = (pred_arr - true_arr) / (pred_arr * (1 - pred_arr))
        grad = grad / pred_arr.size
        pred.grad = grad * output.grad

    output.forward = "binary_crossentropy"
    output.reverse = reverse

    return output


def categorical_crossentropy(true: Tensor, pred: Tensor, logits: bool = True) -> Tensor:
    assert true.shape == pred.shape
    if logits:
        pred = softmax(pred)

    pred_arr = np.clip(pred.data, EPSILON, 1.0 - EPSILON)
    true_arr = true.data

    data = -np.sum(true_arr * np.log(pred_arr)) / pred_arr.shape[0]
    output = Tensor(data)
    output.add_children((pred,))

    def reverse():
        grad = (pred_arr - true_arr) / pred_arr.shape[0]
        pred.grad = grad * output.grad

    output.forward = "categorical_crossentropy"
    output.reverse = reverse

    return output


def mean_squared_error(
    true: Tensor, pred: Tensor, reduction: Literal["mean", "sum"] = "mean"
) -> Tensor:
    """MSE loss. reduction='mean' (default) or 'sum'.

    Mininet's SquaredError is 0.5 * mean_squared_error(..., reduction='sum').
    """
    assert true.shape == pred.shape
    diff_sq = (true - pred) ** 2
    if reduction == "mean":
        return pmath.mean(diff_sq)
    elif reduction == "sum":
        return pmath.sum(diff_sq)
    raise ValueError("reduction must be 'mean' or 'sum'")


def mean_absolute_error(true: Tensor, pred: Tensor) -> Tensor:
    assert true.shape == pred.shape
    return pmath.mean(pmath.abs(true - pred))

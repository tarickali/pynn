from collections.abc import Callable
from typing import Literal

import numpy as np

import pynn.core.math as pmath
from pynn.core import Tensor
from pynn.core.constants import EPSILON
from pynn.core.numeric import stable_sigmoid
from pynn.core.types import Array

__all__ = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "huber",
    "mean_absolute_error",
    "mean_squared_error",
    "sparse_categorical_crossentropy",
]

#: How a per-item loss becomes the number that gets differentiated. `"none"` returns the
#: per-item losses untouched, which is what a caller weighting examples needs — and what
#: makes `backward()` require an explicit gradient, since the result is not a scalar.
Reduction = Literal["mean", "sum", "none"]


def _check_same_shape(true: Tensor, pred: Tensor) -> None:
    if true.shape != pred.shape:
        raise ValueError(
            f"true and pred must have the same shape, got {true.shape} and {pred.shape}"
        )


def _reduce(
    elementwise: Array,
    reduction: Reduction,
    pred: Tensor,
    backward: Callable[[Array], Array],
    name: str,
) -> Tensor:
    """Reduce a per-item loss and record the node that undoes it.

    The losses that fuse an activation share this: each computes its per-item loss and a
    `backward` that turns an upstream gradient of that shape into a gradient with
    respect to `pred`. Reduction is then the only thing that differs between `"mean"`,
    `"sum"`, and `"none"`, which is what keeps the three modes from drifting apart —
    every loss reduces the same way or none of them do.

    Parameters
    ----------
    elementwise : Array
        The loss per item. For the cross-entropies an "item" is one example, since the
        loss already summed over classes; elsewhere it is one element.
    reduction : Reduction
        How to collapse `elementwise`.
    pred : Tensor
        The tensor being differentiated with respect to.
    backward : Callable[[Array], Array]
        Maps an upstream gradient shaped like `elementwise` to one shaped like `pred`.
    name : str
        Recorded on the output node for debugging.

    Raises
    ------
    ValueError
        If `reduction` is not one of the three modes.
    """
    if reduction == "mean":
        data: Array = np.mean(elementwise)
        scale = 1.0 / elementwise.size
    elif reduction == "sum":
        data = np.sum(elementwise)
        scale = 1.0
    elif reduction == "none":
        data = elementwise
        scale = 1.0
    else:
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}"
        )

    output = Tensor(data)
    output.add_children((pred,))

    def reverse() -> None:
        # For a scalar reduction the upstream gradient is one number that applies to
        # every item; broadcasting it back to the per-item shape lets one expression
        # serve all three modes.
        upstream = np.broadcast_to(output.grad, elementwise.shape)
        pred.grad += backward(upstream * scale)

    output.forward = name
    output.reverse = reverse

    return output


def binary_crossentropy(
    true: Tensor,
    pred: Tensor,
    logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Binary cross-entropy.

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
        ``sigmoid(z) - y``.
    reduction : Reduction, default "mean"
        Averaged, summed, or returned per element.

    Returns
    -------
    Tensor
        Scalar unless ``reduction="none"``, in which case it has ``pred``'s shape.
    """
    _check_same_shape(true, pred)

    true_arr = true.data

    if logits:
        z = pred.data
        # log(1 + exp(-|z|)) + max(z, 0) - z*y is the overflow-free rearrangement of
        # -[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))].
        elementwise = np.maximum(z, 0) - z * true_arr + np.log1p(np.exp(-np.abs(z)))
        probabilities = stable_sigmoid(z)

        def backward(upstream: Array) -> Array:
            return (probabilities - true_arr) * upstream
    else:
        p = np.clip(pred.data, EPSILON, 1.0 - EPSILON)
        elementwise = -(true_arr * np.log(p) + (1 - true_arr) * np.log(1 - p))

        def backward(upstream: Array) -> Array:
            return (p - true_arr) / (p * (1 - p)) * upstream

    return _reduce(elementwise, reduction, pred, backward, "binary_crossentropy")


def categorical_crossentropy(
    true: Tensor,
    pred: Tensor,
    logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Categorical cross-entropy over one-hot targets.

    Parameters
    ----------
    true : Tensor
        One-hot (or otherwise normalized) targets of shape (batch, ..., classes).
    pred : Tensor
        Logits when ``logits`` is True, otherwise probabilities.
    logits : bool, default True
        When True, softmax is fused into this loss so that ``dL/dlogits`` is the
        familiar ``softmax(z) - y``. Fusing matters here: applying that expression to a
        separate softmax node's output would send a logit-space gradient back through
        the softmax Jacobian a second time.
    reduction : Reduction, default "mean"
        Averaged over examples, summed, or returned per example. Note that the loss is
        always summed over *classes* first — that sum is the definition, not a
        reduction — so ``"none"`` yields one value per example.

    Returns
    -------
    Tensor
        Scalar unless ``reduction="none"``, in which case it has one entry per example.
    """
    _check_same_shape(true, pred)

    true_arr = true.data

    if logits:
        z = pred.data
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        probabilities = e / np.sum(e, axis=-1, keepdims=True)
        per_example = -np.sum(
            true_arr * np.log(np.maximum(probabilities, EPSILON)), axis=-1
        )

        def backward(upstream: Array) -> Array:
            return (probabilities - true_arr) * upstream[..., np.newaxis]
    else:
        p = np.clip(pred.data, EPSILON, 1.0)
        per_example = -np.sum(true_arr * np.log(p), axis=-1)

        def backward(upstream: Array) -> Array:
            return -true_arr / p * upstream[..., np.newaxis]

    return _reduce(per_example, reduction, pred, backward, "categorical_crossentropy")


def sparse_categorical_crossentropy(
    true: Tensor | Array,
    pred: Tensor,
    logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Categorical cross-entropy over integer class labels.

    The same loss as `categorical_crossentropy`, taking the labels directly rather than
    making the caller one-hot them first. For a large vocabulary that matters: one-hot
    targets for 50,000 classes are a 50,000-wide array of zeros per example, and every
    operation on them is a multiply by zero.

    Parameters
    ----------
    true : Tensor | Array
        Integer class indices, one per example. Not differentiated.
    pred : Tensor
        Logits of shape (batch, classes) when ``logits`` is True, otherwise
        probabilities.
    logits : bool, default True
        Whether ``pred`` holds logits, in which case softmax is fused in.
    reduction : Reduction, default "mean"
        Averaged over examples, summed, or returned per example.

    Returns
    -------
    Tensor
        Scalar unless ``reduction="none"``, in which case it has one entry per example.

    Raises
    ------
    ValueError
        If the number of labels does not match the number of rows in ``pred``, or a
        label is outside the range of classes.
    """
    labels = np.asarray(true.data if isinstance(true, Tensor) else true)
    labels = labels.reshape(-1).astype(np.intp)

    if pred.ndim != 2:
        raise ValueError(
            f"sparse_categorical_crossentropy expects pred of shape (batch, classes), "
            f"got {pred.shape}"
        )
    batch, classes = pred.shape[0], pred.shape[1]
    if labels.shape[0] != batch:
        raise ValueError(
            f"expected one label per example: pred has {batch} rows but {labels.size} "
            "labels were given"
        )
    if labels.size and (labels.min() < 0 or labels.max() >= classes):
        raise ValueError(
            f"labels must be in [0, {classes}), got range "
            f"[{labels.min()}, {labels.max()}]"
        )

    rows = np.arange(batch)

    if logits:
        z = pred.data
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        probabilities = e / np.sum(e, axis=-1, keepdims=True)
        per_example = -np.log(np.maximum(probabilities[rows, labels], EPSILON))

        def backward(upstream: Array) -> Array:
            # softmax(z) - onehot(y), written without ever forming the one-hot.
            gradient = probabilities * upstream[:, np.newaxis]
            gradient[rows, labels] -= upstream
            return gradient
    else:
        p = np.clip(pred.data, EPSILON, 1.0)
        per_example = -np.log(p[rows, labels])

        def backward(upstream: Array) -> Array:
            gradient = np.zeros_like(p)
            gradient[rows, labels] = -upstream / p[rows, labels]
            return gradient

    return _reduce(
        per_example, reduction, pred, backward, "sparse_categorical_crossentropy"
    )


def huber(
    true: Tensor,
    pred: Tensor,
    delta: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Huber loss: squared error near zero, absolute error past ``delta``.

    Squared error weights an outlier by its square, so one bad label can dominate a
    batch; absolute error does not, but its gradient is discontinuous at zero and never
    shrinks as the fit improves. Huber is the compromise — quadratic where the residual
    is small, linear where it is large, and continuous in value and slope at the join.

    Parameters
    ----------
    true, pred : Tensor
        Targets and predictions, same shape.
    delta : float
        Residual magnitude at which the loss switches from quadratic to linear.
    reduction : Reduction, default "mean"
        Averaged, summed, or returned per element.

    Returns
    -------
    Tensor
        Scalar unless ``reduction="none"``, in which case it has ``pred``'s shape.

    Raises
    ------
    ValueError
        If `delta` is not positive.
    """
    _check_same_shape(true, pred)
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")

    residual = pred.data - true.data
    magnitude = np.abs(residual)
    quadratic = magnitude <= delta
    elementwise = np.where(
        quadratic, 0.5 * residual**2, delta * (magnitude - 0.5 * delta)
    )

    def backward(upstream: Array) -> Array:
        # The two pieces agree at |r| = delta, which is what makes the slope continuous
        # there: r and delta*sign(r) are the same number at the join.
        return np.where(quadratic, residual, delta * np.sign(residual)) * upstream

    return _reduce(elementwise, reduction, pred, backward, "huber")


def mean_squared_error(
    true: Tensor, pred: Tensor, reduction: Reduction = "mean"
) -> Tensor:
    """Squared error between ``true`` and ``pred``.

    Built from graph primitives rather than fused, since squaring a difference has no
    stability problem to solve and the composed form differentiates itself.
    """
    _check_same_shape(true, pred)
    squared = (true - pred) ** 2
    if reduction == "mean":
        return pmath.mean(squared)
    if reduction == "sum":
        return pmath.sum(squared)
    if reduction == "none":
        return squared
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def mean_absolute_error(
    true: Tensor, pred: Tensor, reduction: Reduction = "mean"
) -> Tensor:
    """Absolute error between ``true`` and ``pred``."""
    _check_same_shape(true, pred)
    absolute = pmath.abs(true - pred)
    if reduction == "mean":
        return pmath.mean(absolute)
    if reduction == "sum":
        return pmath.sum(absolute)
    if reduction == "none":
        return absolute
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")

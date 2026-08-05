"""Numerically careful array kernels.

These operate on raw arrays and build no graph nodes, so they sit below the autodiff
layer and are shared by `pynn.functional.activations` and `pynn.functional.losses`.

Properties asserted here are checked by `pynn.verify.stability`.
"""

import numpy as np

from pynn.core.types import Array

__all__ = ["stable_sigmoid"]


def stable_sigmoid(array: Array) -> Array:
    """Logistic sigmoid without overflow for large-magnitude inputs.

    ``1 / (1 + exp(-x))`` overflows for very negative ``x``. Evaluating ``exp(-x)`` on
    the positive half and ``exp(x)`` on the negative half keeps every exponent <= 0.

    Parameters
    ----------
    array : Array
        Input of any shape.

    Returns
    -------
    Array
        ``1 / (1 + exp(-array))``, as float64, finite for every finite input.

    Examples
    --------
    >>> stable_sigmoid(np.array([-800.0, 0.0, 800.0])).tolist()
    [0.0, 0.5, 1.0]
    """

    positive = array >= 0
    result = np.empty(array.shape, dtype=np.float64)

    exp_negative = np.exp(-array[positive])
    result[positive] = 1.0 / (1.0 + exp_negative)

    exp_positive = np.exp(array[~positive])
    result[~positive] = exp_positive / (1.0 + exp_positive)

    return result

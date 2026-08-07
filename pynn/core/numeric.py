"""Numerically careful array kernels.

These operate on raw arrays and build no graph nodes, so they sit below the autodiff
layer and are shared by `pynn.functional.activations` and `pynn.functional.losses`.

Properties asserted here are checked by `pynn.verify.stability`.
"""

import math

import numpy as np

from pynn.core.types import Array

__all__ = ["erf", "stable_sigmoid"]

#: `math.erf` lifted to arrays. NumPy has no erf and SciPy is not a dependency — it
#: reaches this environment only through scikit-learn, which is an optional extra, so
#: relying on it would make a base install fail. `frompyfunc` returns an object array,
#: hence the cast back to float64.
_erf = np.frompyfunc(math.erf, 1, 1)


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


def erf(array: Array) -> Array:
    """The error function, elementwise.

    Correctly rounded, since it defers to `math.erf` per element — and therefore an
    order of magnitude slower than a vectorized kernel would be. `gelu` uses it only on
    its exact path; the tanh approximation it defaults to needs nothing but `np.tanh`.

    Parameters
    ----------
    array : Array
        Input of any shape.

    Returns
    -------
    Array
        ``erf(array)`` as float64.

    Examples
    --------
    >>> erf(np.array([-1.0, 0.0, 1.0])).round(6).tolist()
    [-0.842701, 0.0, 0.842701]
    """
    return np.asarray(_erf(array), dtype=np.float64)

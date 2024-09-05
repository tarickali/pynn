from numba import njit

from ..types import Array, Number

__all__ = [
    "add",
    "subtract",
    "multiply",
    "matrix_multiply",
    "true_division",
    "power",
    "negate",
    "transpose",
    "equal",
    "not_equal",
    "greater_than_equal",
    "greater_than",
    "less_than_equal",
    "less_than",
]


@njit
def add(x: Array, y: Array) -> Array:
    return x + y


@njit
def subtract(x: Array, y: Array) -> Array:
    return x - y


@njit
def multiply(x: Array, y: Array) -> Array:
    return x * y


# NOTE: Generic matrix multiplication is not support with numba
def matrix_multiply(x: Array, y: Array) -> Array:
    return x @ y


@njit
def true_division(x: Array, y: Array) -> Array:
    return x / y


@njit
def power(x: Array, y: Number) -> Array:
    return x**y


@njit
def negate(x: Array) -> Array:
    return -x


@njit
def transpose(x: Array) -> Array:
    return x.T


@njit
def equal(x: Array, y: Array) -> Array:
    return x == y


@njit
def not_equal(x: Array, y: Array) -> Array:
    return x != y


@njit
def greater_than_equal(x: Array, y: Array) -> Array:
    return x >= y


@njit
def greater_than(x: Array, y: Array) -> Array:
    return x > y


@njit
def less_than_equal(x: Array, y: Array) -> Array:
    return x <= y


@njit
def less_than(x: Array, y: Array) -> Array:
    return x < y

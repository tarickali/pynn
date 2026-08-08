"""Elementwise and matrix primitives, one thin wrapper per NumPy operation.

They exist so that `Tensor`'s operators read as names rather than symbols, and so the
forward computation for each is in one place next to nothing else. Every body is a
single NumPy call, which is already a vectorized C kernel — there is no Python-level
loop here for a JIT to remove, and nothing to gain by compiling them.
"""

from .types import Array, Number

__all__ = [
    "add",
    "equal",
    "greater_than",
    "greater_than_equal",
    "less_than",
    "less_than_equal",
    "matrix_multiply",
    "multiply",
    "negate",
    "not_equal",
    "power",
    "subtract",
    "transpose",
    "true_division",
]


def add(x: Array, y: Array) -> Array:
    return x + y


def subtract(x: Array, y: Array) -> Array:
    return x - y


def multiply(x: Array, y: Array) -> Array:
    return x * y


def matrix_multiply(x: Array, y: Array) -> Array:
    return x @ y


def true_division(x: Array, y: Array) -> Array:
    return x / y


def power(x: Array, y: Number) -> Array:
    return x**y


def negate(x: Array) -> Array:
    return -x


def transpose(x: Array) -> Array:
    return x.T


def equal(x: Array, y: Array) -> Array:
    return x == y


def not_equal(x: Array, y: Array) -> Array:
    return x != y


def greater_than_equal(x: Array, y: Array) -> Array:
    return x >= y


def greater_than(x: Array, y: Array) -> Array:
    return x > y


def less_than_equal(x: Array, y: Array) -> Array:
    return x <= y


def less_than(x: Array, y: Array) -> Array:
    return x < y

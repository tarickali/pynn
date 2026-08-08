"""Core abstractions: the autodiff Tensor, and the base classes built on top of it.

Re-exported explicitly rather than by star import so that the public surface is readable
here, and so type checkers and editors can resolve `pynn.core.X` to its definition.
"""

from .activation import Activation
from .constants import EPSILON, MAXINT, MININT, PI, E
from .grad_mode import enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from .initializer import Initializer
from .loss import Loss
from .module import Module
from .optimizer import Optimizer
from .shape import concat, split, stack
from .tensor import Tensor
from .types import Array, ArrayLike, DataType, List, Number, Shape

__all__ = [
    "EPSILON",
    "MAXINT",
    "MININT",
    "PI",
    "Activation",
    "Array",
    "ArrayLike",
    "DataType",
    "E",
    "Initializer",
    "List",
    "Loss",
    "Module",
    "Number",
    "Optimizer",
    "Shape",
    "Tensor",
    "concat",
    "enable_grad",
    "is_grad_enabled",
    "no_grad",
    "set_grad_enabled",
    "split",
    "stack",
]

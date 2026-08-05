import sys

import numpy as np

__all__ = ["EPSILON", "MAXINT", "MININT", "PI", "E"]

EPSILON = np.finfo(float).eps
E = np.e
PI = np.pi
MAXINT = sys.maxsize
MININT = -sys.maxsize - 1

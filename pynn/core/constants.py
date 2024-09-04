import numpy as np
import sys

__all__ = ["EPSILON", "E", "PI", "MAXINT", "MININT"]

EPSILON = np.finfo(float).eps
E = np.e
PI = np.pi
MAXINT = sys.maxsize
MININT = -sys.maxsize - 1

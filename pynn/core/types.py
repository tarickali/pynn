import numpy as np

__all__ = ["Array", "List", "Number", "ArrayLike", "DataType", "Shape"]

Array = np.ndarray
List = list
Number = np.number | int | float | bool
ArrayLike = Array | List | Number
DataType = np.dtype | int | float | bool
Shape = tuple[None | np.int64, ...]

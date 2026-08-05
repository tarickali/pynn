import numpy as np
import numpy.typing as npt

__all__ = ["Array", "ArrayLike", "DataType", "List", "Number", "Shape"]

Array = np.ndarray
List = list
Number = np.number | int | float | bool
ArrayLike = Array | List | Number
#: Anything numpy accepts as a `dtype=` argument, including `np.float64` (a *type*,
#: not a `np.dtype` instance), a string like `"float32"`, or `float`.
DataType = npt.DTypeLike
#: A concrete array shape. Never contains `None`; layers that infer a dimension track
#: it as `Shape | None` until `build()` resolves it.
Shape = tuple[int, ...]

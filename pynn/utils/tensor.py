import numpy as np

from pynn.core import Tensor

__all__ = ["get_data_and_grad"]


def get_data_and_grad(param: Tensor) -> tuple[np.ndarray, np.ndarray]:
    return param.data, param.grad

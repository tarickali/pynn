from abc import ABC, abstractmethod

from pynn.core import Tensor

__all__ = ["Loss"]


class Loss(ABC):
    @abstractmethod
    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return self.compute(y_true, y_pred)

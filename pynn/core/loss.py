from abc import ABC, abstractmethod

from pynn.core import Tensor


class Loss(ABC):
    @abstractmethod
    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return self.compute(y_true, y_pred)

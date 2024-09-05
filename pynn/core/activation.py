from abc import ABC, abstractmethod
from pynn.core import Tensor

__all__ = ["Activation"]


class Activation(ABC):
    @abstractmethod
    def compute(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.compute(x)

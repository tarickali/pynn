from abc import ABC, abstractmethod

from pynn.core.types import Shape
from pynn.core.tensor import Tensor

__all__ = ["Initializer"]


class Initializer(ABC):
    @abstractmethod
    def init(self, shape: Shape) -> Tensor:
        raise NotImplementedError

    def __call__(self, shape: Shape) -> Tensor:
        return self.init(shape)

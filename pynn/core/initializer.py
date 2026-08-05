from abc import ABC, abstractmethod

from pynn.core.tensor import Tensor
from pynn.core.types import Shape

__all__ = ["Initializer"]


class Initializer(ABC):
    @abstractmethod
    def init(self, shape: Shape) -> Tensor:
        raise NotImplementedError

    def __call__(self, shape: Shape) -> Tensor:
        return self.init(shape)

from typing import Any
from abc import ABC, abstractmethod

from pynn.core import Tensor

__all__ = ["Model"]


class Model(ABC):
    """Base class for all models.

    A Model is a container of modules that supports forward pass,
    gradient zeroing, and parameter access. Backward pass is performed
    via autograd on the loss Tensor (loss.backward()).
    """

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Compute the forward pass of the model on input X."""
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear the gradients for each parameter in the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> list[dict[str, Any]]:
        """Return all parameters (list of param dicts, one per module)."""
        raise NotImplementedError

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)

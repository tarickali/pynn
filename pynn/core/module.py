from typing import Any
from abc import ABC, abstractmethod

from pynn.core.types import Shape
from pynn.core import Tensor

__all__ = ["Module"]


class Module(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.parameters: dict[str, Tensor] = {}
        self.trainable: bool = True
        self.initialized: bool = False

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Computes the forward pass of the Module on input X.

        Parameters
        ----------
        X : Tensor

        Returns
        -------
        Tensor

        """

        raise NotImplementedError

    def build(self, input_shape: int | Shape) -> None:
        """Build the Module parameters based on the given input shape."""

        return None

    def zero_grad(self) -> None:
        """Clear the gradients for each parameter in the Module."""

        for param in self.parameters:
            self.parameters[param].zero_grad()

    def freeze(self) -> None:
        """Set the Module to be untrainable."""

        for param in self.parameters:
            self.parameters[param].trainable = False
        self.trainable = False

    def unfreeze(self) -> None:
        """Set the Module to be trainable."""

        for param in self.parameters:
            self.parameters[param].trainable = True
        self.trainable = True

    def summary(self) -> dict[str, Any]:
        """Get a summary of the Module.

        Returns
        -------
        dict[str, Any]

        """

        return {
            "name": self.name,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        """Get the hyperparameters of the Module.

        Returns
        -------
        dict[str, Any]

        """

        raise NotImplementedError

from abc import ABC, abstractmethod

from pynn.core.tensor import Tensor

__all__ = ["Optimizer"]


class Optimizer(ABC):
    def __init__(
        self, parameters: list[dict[str, Tensor]], learning_rate: float
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.time = 0

    def increment(self) -> None:
        self.time += 1

    def reset(self) -> None:
        self.time = 0

    def trainable_parameters(self) -> list[dict[str, Tensor]]:
        """The parameter groups with frozen parameters removed.

        `update` implementations must iterate this rather than `self.parameters`, or
        `Module.freeze()` has no effect: a frozen parameter still receives a gradient
        during the backward pass, so an optimizer reading `self.parameters` directly
        will happily step it.

        One group is returned per group in `self.parameters`, empty groups included, so
        that the result stays index-aligned with per-group optimizer state.
        """
        return [
            {name: p for name, p in group.items() if p.trainable}
            for group in self.parameters
        ]

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError

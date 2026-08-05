from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pynn.core.module import Module
from pynn.core.tensor import Tensor

__all__ = ["Optimizer"]

#: What an optimizer accepts: a whole model, or explicit parameter groups.
ParameterSource = Module | Sequence[dict[str, Tensor]]


def as_parameter_groups(parameters: ParameterSource) -> list[dict[str, Tensor]]:
    """Normalize an optimizer's `parameters` argument to a list of groups.

    A `Module` contributes one group per module in its tree, holding the modules' live
    parameter dictionaries. Keeping the references live rather than copying is what
    makes lazily built layers work: the optimizer is usually constructed before the
    first forward pass, when those dictionaries are still empty.
    """

    if isinstance(parameters, Module):
        return parameters.parameter_groups()
    if isinstance(parameters, dict):
        raise TypeError(
            "expected a Module or a sequence of parameter groups, got a single dict. "
            "Pass the model itself, or [layer.parameters] for one layer."
        )
    groups = list(parameters)
    for group in groups:
        if not isinstance(group, dict):
            raise TypeError(
                "each parameter group must be a {name: Tensor} dict, got "
                f"{type(group).__name__}"
            )
    return groups


class Optimizer(ABC):
    def __init__(self, parameters: ParameterSource, learning_rate: float) -> None:
        super().__init__()
        self.parameters = as_parameter_groups(parameters)
        self.learning_rate = learning_rate
        self.time = 0

    def increment(self) -> None:
        self.time += 1

    def reset(self) -> None:
        self.time = 0

    def zero_grad(self) -> None:
        """Clear the gradients of every parameter this optimizer steps.

        Frozen parameters are cleared too: they still accumulate gradients during the
        backward pass, and leaving those to pile up across steps would make them wrong
        the moment the parameter is unfrozen.
        """
        for group in self.parameters:
            for parameter in group.values():
                parameter.zero_grad()

    def step(self) -> None:
        """Apply one update. Alias for `update`, matching PyTorch's spelling."""
        self.update()

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

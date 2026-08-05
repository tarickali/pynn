from typing import Any

from pynn.core import Model, Module, Tensor

__all__ = ["Sequential"]


class Sequential(Model):
    """A sequential container of modules.

    Modules are applied in order in the forward pass. Gradients are
    computed via autograd by calling backward() on the loss Tensor.
    """

    def __init__(self, modules: list[Module] | None = None) -> None:
        super().__init__()
        self.modules = list(modules) if modules is not None else []

    def forward(self, X: Tensor) -> Tensor:
        for module in self.modules:
            X = module(X)
        return X

    def zero_grad(self) -> None:
        for module in self.modules:
            module.zero_grad()

    def append(self, module: Module) -> None:
        self.modules.append(module)

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [m.parameters for m in self.modules]

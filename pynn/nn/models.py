from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pynn.core import Module, Tensor

__all__ = ["Sequential"]


class Sequential(Module):
    """A container that applies its modules in order.

    Being a `Module` rather than a separate container type is what lets it nest:
    `Sequential([Sequential([...]), Linear(...)])` reports its parameters through the
    same recursive walk as any other module, so the optimizer sees one flat list of
    groups no matter how the layers are grouped. As a `Model` with a `list[dict]`
    `parameters`, nesting type-checked and ran the forward pass, then failed inside
    the optimizer.
    """

    def __init__(
        self, modules: Iterable[Module] | None = None, name: str = "Sequential"
    ) -> None:
        super().__init__()
        self.name = name
        for module in modules or ():
            self.append(module)

    def forward(self, X: Tensor) -> Tensor:
        for module in self._modules.values():
            X = module(X)
        return X

    def append(self, module: Module) -> Module:
        """Add `module` to the end of the sequence, and return it."""
        return self.register_module(str(len(self._modules)), module)

    @property
    def modules(self) -> list[Module]:
        """The child modules, in order."""
        return list(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        return self.modules[index]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"modules": [module.name for module in self]}

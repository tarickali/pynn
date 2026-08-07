"""Containers that hold modules without deciding how they are applied.

`Sequential` covers the common case — apply these in order — but a model whose layers
are wired by hand still needs somewhere to keep them that the module tree can see. A
plain Python list cannot be that place: nothing registers it, so its modules run in the
forward pass, receive gradients, and are never handed to an optimizer.

`Module.__setattr__` raises on a plain list or dict of modules and names these two
instead, so the failure is loud rather than a model that trains one layer and silently
leaves the rest at their initial weights.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from pynn.core import Module, Tensor

__all__ = ["ModuleDict", "ModuleList"]


class _Container(Module):
    """Shared behaviour: a container has children but no forward pass of its own."""

    def forward(self, X: Tensor) -> Tensor:
        raise TypeError(
            f"{type(self).__name__} has no forward pass. It holds modules for an "
            "enclosing module to wire up itself; use Sequential to apply them in order."
        )

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"modules": [module.name for module in self._modules.values()]}


class ModuleList(_Container):
    """An indexable list of modules, registered with the module tree.

    Use it wherever a plain list would hold layers::

        class Trunk(Module):
            def __init__(self, depth: int) -> None:
                super().__init__()
                self.blocks = ModuleList(Linear(64, activation="relu")
                                         for _ in range(depth))

            def forward(self, X):
                for block in self.blocks:
                    X = block(X) + X          # the wiring is yours
                return X

    Indices are positions, not names: inserting or deleting renumbers the ones after
    it, exactly as a list does. That means the dotted paths in `state_dict` follow
    position too, so a checkpoint taken before an insert does not load after one.
    """

    def __init__(
        self, modules: Iterable[Module] | None = None, name: str = "ModuleList"
    ) -> None:
        super().__init__()
        self.name = name
        self.extend(modules or ())

    def append(self, module: Module) -> Module:
        """Add `module` to the end, and return it."""
        return self.register_module(str(len(self._modules)), module)

    def extend(self, modules: Iterable[Module]) -> ModuleList:
        """Add each of `modules` to the end, and return self."""
        for module in modules:
            self.append(module)
        return self

    def insert(self, index: int, module: Module) -> Module:
        """Insert `module` at `index`, shifting the rest along."""
        if not isinstance(module, Module):
            raise TypeError(f"expected a Module, got {type(module).__name__}")
        modules = list(self._modules.values())
        modules.insert(index, module)
        self._renumber(modules)
        return module

    def _renumber(self, modules: list[Module]) -> None:
        """Rebuild the registry so keys are 0..n-1 in order."""
        self._modules.clear()
        for position, module in enumerate(modules):
            self._modules[str(position)] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(list(self._modules.values()))

    def __contains__(self, module: object) -> bool:
        return any(child is module for child in self._modules.values())

    def __getitem__(self, index: int | slice) -> Module | ModuleList:
        modules = list(self._modules.values())
        if isinstance(index, slice):
            # A slice of a list is a list; a slice of a ModuleList is a ModuleList, so
            # that the result is still something a Module can hold.
            return ModuleList(modules[index])
        return modules[index]

    def __setitem__(self, index: int, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"expected a Module, got {type(module).__name__}")
        modules = list(self._modules.values())
        modules[index] = module
        self._renumber(modules)

    def __delitem__(self, index: int | slice) -> None:
        modules = list(self._modules.values())
        del modules[index]
        self._renumber(modules)

    def __iadd__(self, modules: Iterable[Module]) -> ModuleList:
        return self.extend(modules)


class ModuleDict(_Container):
    """A keyed collection of modules, registered with the module tree.

    Use it when the choice of module is data rather than position — one head per task,
    one embedding per feature::

        self.heads = ModuleDict({"value": Linear(1), "policy": Linear(n_actions)})
        ...
        return self.heads["policy"](trunk)

    Keys become part of the dotted paths in `state_dict`, so they may not contain a
    `.`; iteration yields keys, matching `dict`.
    """

    def __init__(
        self, modules: Mapping[str, Module] | None = None, name: str = "ModuleDict"
    ) -> None:
        super().__init__()
        self.name = name
        self.update(modules or {})

    def update(self, modules: Mapping[str, Module]) -> ModuleDict:
        """Add or replace each entry of `modules`, and return self."""
        for key, module in modules.items():
            self[key] = module
        return self

    def pop(self, key: str) -> Module:
        """Remove `key` and return the module that was under it."""
        return self._modules.pop(key)

    def clear(self) -> None:
        self._modules.clear()

    def keys(self) -> list[str]:
        return list(self._modules)

    def values(self) -> list[Module]:
        return list(self._modules.values())

    def items(self) -> list[tuple[str, Module]]:
        return list(self._modules.items())

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._modules))

    def __contains__(self, key: object) -> bool:
        return key in self._modules

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"expected a Module, got {type(module).__name__}")
        if "." in key:
            raise ValueError(
                f"module key {key!r} may not contain '.': keys become path segments "
                "in state_dict, and a dot there would be indistinguishable from the "
                "separator between a module and its parameter"
            )
        self._modules[key] = module

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"modules": {key: module.name for key, module in self._modules.items()}}

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pynn.core.activation import Activation
from pynn.core.tensor import Tensor
from pynn.core.types import Array, Shape

__all__ = ["Module"]


def _reject_plain_container(name: str, value: Any) -> None:
    """Refuse a plain list, tuple, or dict that holds Modules.

    Only a `Module` is registered on assignment, so a plain container of them is
    invisible to the tree: its layers run in the forward pass, receive gradients, and
    are never handed to an optimizer. Nothing raises, and the model trains everything
    except those layers, which sit at their initial weights.

    That is the exact failure `ModuleList` and `ModuleDict` exist to prevent, so the
    assignment is refused rather than silently accepted.
    """
    if isinstance(value, list | tuple):
        holds_modules = any(isinstance(item, Module) for item in value)
        wrapper = "ModuleList"
    elif isinstance(value, dict):
        holds_modules = any(isinstance(item, Module) for item in value.values())
        wrapper = "ModuleDict"
    else:
        return

    if holds_modules:
        raise TypeError(
            f"cannot assign a plain {type(value).__name__} of Modules to {name!r}: it "
            "would not be registered, so those modules would receive gradients but "
            f"never be updated by an optimizer. Wrap it in pynn.nn.{wrapper}."
        )


def _hint(value: Any) -> str:
    """A suffix naming the fix, when the thing handed to a container has one.

    `Sequential([ReLU()])` is the natural thing to write and does not work: the
    stateless activations are `Activation` objects rather than Modules, so a container
    cannot hold one. "expected a Module, got ReLU" is true and leaves the reader no
    better off, hence this.
    """
    if isinstance(value, Activation):
        return (
            ". The stateless activations are not Modules; wrap it in "
            'pynn.nn.Activation, e.g. pynn.nn.Activation("relu"), or use '
            "pynn.nn.Identity() for the identity"
        )
    return ""


class Module(ABC):
    """Base class for every layer and container.

    A Module owns two things: its own parameters, in `parameters`, and its child
    modules, which are registered automatically when one is assigned to an attribute.
    Every collective operation — `named_parameters`, `parameter_groups`, `state_dict`,
    `zero_grad`, `train`, `freeze` — walks that tree recursively, which is what lets
    containers nest and what lets an optimizer take a whole model regardless of how
    deeply its layers are grouped.

    Subclasses implement `forward` and `hyperparameters`, and create their parameters
    in `build`, which is called on the first forward pass with the input shape.
    """

    def __init__(self) -> None:
        super().__init__()
        # First, because __setattr__ consults it on every subsequent assignment.
        self._modules: dict[str, Module] = {}
        self.name: str = self.__class__.__name__
        self.parameters: dict[str, Tensor] = {}
        #: State that belongs to the module but is not optimized — batch
        #: normalization's running statistics. Saved and loaded with the parameters,
        #: because a model whose running statistics are lost evaluates differently
        #: after a round trip, but never handed to an optimizer.
        self._buffers: dict[str, Array] = {}
        self.trainable: bool = True
        self.training: bool = True
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

    def build(self, input_shape: Shape) -> None:
        """Build the Module parameters based on the given input shape."""
        return

    # ------------------------------------------------------------------------ #
    # The module tree
    # ------------------------------------------------------------------------ #
    def __setattr__(self, name: str, value: Any) -> None:
        """Register child modules as they are assigned.

        `self.encoder = Sequential(...)` in a custom Module has to put `encoder` into
        the tree, or its parameters are invisible to the optimizer and to
        `state_dict` — a failure that shows up as a layer that silently never trains.
        """
        if isinstance(value, Module):
            modules = self.__dict__.get("_modules")
            if modules is None:
                raise AttributeError(
                    f"cannot assign the child module {name!r} before "
                    f"{type(self).__name__}.__init__() calls super().__init__()"
                )
            modules[name] = value
        else:
            _reject_plain_container(name, value)
            if name in self.__dict__.get("_modules", {}):
                # Overwriting a child with a non-module takes it out of the tree.
                del self._modules[name]
        object.__setattr__(self, name, value)

    def register_module(self, name: str, module: Module) -> Module:
        """Register `module` as a child under `name` and return it.

        Containers that hold their children in a list rather than in named attributes
        use this; attribute assignment registers automatically.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"expected a Module, got {type(module).__name__}{_hint(module)}"
            )
        self._modules[name] = module
        return module

    def named_children(self) -> list[tuple[str, Module]]:
        """The direct child modules, in registration order."""
        return list(self._modules.items())

    def named_modules(self, prefix: str = "") -> list[tuple[str, Module]]:
        """This module and every descendant, depth first, under dotted names.

        The root is included under `prefix`, which is empty by default, so a bare
        layer's own parameters keep their unqualified names.
        """
        found = [(prefix, self)]
        for name, child in self._modules.items():
            found.extend(child.named_modules(f"{prefix}.{name}" if prefix else name))
        return found

    def register_parameter(self, name: str, value: Array | Tensor) -> Tensor:
        """Register a parameter under `name` and return it.

        `build` implementations should use this rather than assigning into
        `self.parameters` directly, so that a parameter created by a lazy build inherits
        the Module's frozen state. A Module frozen before its first forward pass has no
        parameters to mark yet, and would otherwise come back trainable once built.
        """
        tensor = value if isinstance(value, Tensor) else Tensor(value)
        tensor.trainable = self.trainable
        self.parameters[name] = tensor
        return tensor

    def register_buffer(self, name: str, value: Array) -> Array:
        """Register non-optimized state under `name` and return it.

        Buffers are updated in place by the layer that owns them, so the array handed
        back stays the live one across a `load_state_dict`.
        """
        self._buffers[name] = np.asarray(value)
        return self._buffers[name]

    def named_buffers(self) -> dict[str, Array]:
        """Every buffer in the tree, keyed by dotted path."""
        return {
            f"{prefix}.{name}" if prefix else name: buffer
            for prefix, module in self.named_modules()
            for name, buffer in module._buffers.items()
        }

    def named_parameters(self) -> dict[str, Tensor]:
        """Every parameter in the tree, keyed by dotted path.

        `Sequential([Linear(4, 3)])` reports `{"0.W": ..., "0.b": ...}`; a bare
        `Linear` reports `{"W": ..., "b": ...}`.
        """
        return {
            f"{prefix}.{name}" if prefix else name: parameter
            for prefix, module in self.named_modules()
            for name, parameter in module.parameters.items()
        }

    def parameter_groups(self) -> list[dict[str, Tensor]]:
        """One group per module in the tree, in the form the optimizers consume.

        The dictionaries are the modules' live `parameters`, not copies, so a layer
        that builds its parameters on the first forward pass is still stepped by an
        optimizer constructed before that happened. Modules with no parameters of
        their own contribute an empty group, which keeps the list index-aligned with
        the optimizers' per-group state as those dictionaries fill in.
        """
        return [module.parameters for _, module in self.named_modules()]

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Total number of scalar parameters in the tree."""
        return sum(
            parameter.size
            for parameter in self.named_parameters().values()
            if parameter.trainable or not trainable_only
        )

    # ------------------------------------------------------------------------ #
    # Collective state
    # ------------------------------------------------------------------------ #
    def zero_grad(self) -> None:
        """Clear the gradients of every parameter in the tree."""
        for parameter in self.named_parameters().values():
            parameter.zero_grad()

    def train(self, mode: bool = True) -> Module:
        """Put the tree into training mode, and return self.

        Dropout and the normalization layers behave differently between fitting and
        inference; this is the switch they read. Propagating to descendants is the
        point — a model is put in eval mode at the top and every layer has to follow.
        """
        for _, module in self.named_modules():
            module.training = mode
        return self

    def eval(self) -> Module:
        """Put the tree into evaluation mode, and return self."""
        return self.train(False)

    def freeze(self) -> None:
        """Exclude this Module's parameters from optimizer updates.

        Frozen parameters still receive gradients during the backward pass; it is the
        optimizer that skips them, via `Optimizer.trainable_parameters`.
        """
        self._set_trainable(False)

    def unfreeze(self) -> None:
        """Return this Module's parameters to being updated by the optimizer."""
        self._set_trainable(True)

    def _set_trainable(self, trainable: bool) -> None:
        # Recorded on each Module as well as on its parameters, so that a build
        # triggered after this point propagates it to the parameters it creates.
        for _, module in self.named_modules():
            module.trainable = trainable
            for parameter in module.parameters.values():
                parameter.trainable = trainable

    # ------------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Array]:
        """Copies of every parameter and buffer, keyed by dotted path.

        Buffers are included because a batch-normalized model whose running statistics
        were dropped evaluates differently after a round trip — the weights alone are
        not the whole model.

        Copies rather than views, so that a checkpoint taken mid-training is a snapshot
        and not a live reference to values that keep moving.
        """
        state = {
            name: parameter.data.copy()
            for name, parameter in self.named_parameters().items()
        }
        state.update(
            (name, buffer.copy()) for name, buffer in self.named_buffers().items()
        )
        return state

    def load_state_dict(self, state: Mapping[str, Array], strict: bool = True) -> None:
        """Copy `state` into this module's parameters and buffers in place.

        Parameters are created by `build` on the first forward pass, so a model that
        has not run one yet has nothing to load into. Run a forward pass (or call
        `build`) before loading.

        Parameters
        ----------
        state : Mapping[str, Array]
            Arrays keyed as `state_dict` produces them.
        strict : bool, default True
            Require the keys to match exactly. With `strict=False`, missing keys keep
            their current values and unexpected keys are ignored.

        Raises
        ------
        KeyError
            If `strict` and the keys do not match.
        ValueError
            If a matching key holds an array of the wrong shape. Checked even when
            `strict` is False, since a silent shape mismatch loads a different model.
        """
        parameters = self.named_parameters()
        buffers = self.named_buffers()
        known = set(parameters) | set(buffers)
        missing = sorted(known - set(state))
        unexpected = sorted(set(state) - known)
        if strict and (missing or unexpected):
            raise KeyError(
                f"state does not match this module: missing {missing}, "
                f"unexpected {unexpected}"
            )

        for name, value in state.items():
            target = parameters.get(name)
            current = target.data if target is not None else buffers.get(name)
            if current is None:
                continue
            array = np.asarray(value)
            if array.shape != current.shape:
                raise ValueError(
                    f"{name!r} has shape {current.shape} but the state holds shape "
                    f"{array.shape}"
                )
            if target is not None:
                target.data = array.astype(target.dtype, copy=True)
            else:
                # In place, so that a layer holding a reference to its own buffer sees
                # the loaded values.
                current[...] = array

    def save(self, path: str | Path) -> None:
        """Write `state_dict` to `path` as an uncompressed `.npz` archive.

        Written through an open handle rather than by passing the path to
        `numpy.savez`, which would append `.npz` and leave `load` looking in the
        wrong place.
        """
        with open(path, "wb") as handle:
            # numpy types `savez` with a keyword-only `allow_pickle: bool`, so mypy
            # checks the unpacked arrays against it. The arrays are the payload.
            np.savez(handle, **self.state_dict())  # type: ignore[arg-type]

    def load(self, path: str | Path, strict: bool = True) -> None:
        """Load parameters from an archive written by `save`."""
        with np.load(path) as archive:
            self.load_state_dict(
                {name: archive[name] for name in archive.files}, strict=strict
            )

    # ------------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """Get a summary of the Module and its children.

        Parameter shapes and counts rather than the parameters themselves, so that
        printing a summary does not dump every weight matrix.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "name": self.name,
            "parameters": {
                name: parameter.shape for name, parameter in self.parameters.items()
            },
            "num_parameters": self.num_parameters(),
            "hyperparameters": self.hyperparameters,
            "modules": [child.summary() for child in self._modules.values()],
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass.

        Forwards whatever it is given rather than a single Tensor, because a recurrent
        cell takes a hidden state alongside its input and returns one alongside its
        output. Every other layer here is Tensor to Tensor.
        """
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.name}({self.num_parameters()} parameters)"

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        """Get the hyperparameters of the Module.

        Returns
        -------
        dict[str, Any]
        """
        raise NotImplementedError

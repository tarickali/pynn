"""The optimizer base class, and the two pieces every update rule starts from.

Every `update` in `pynn/optim/` is written **in place**: the per-parameter state lives
in `self.cache` and is mutated with `*=` / `+=` / `out=` rather than rebound, and the
step is applied as `data -= ...` rather than `param.data = param.data - ...`. The
arithmetic is unchanged; what goes away is the allocation. Profiling an MLP training
step put `SGD.update` at 32% of it, and almost all of that was allocating a fresh
full-size array per line — seven of them per parameter per step for momentum SGD,
several of which were computing `+ 0.0 * data` or `* 1.0`.

Two consequences worth knowing about:

- **`param.data` is mutated, not replaced.** A caller holding the array — `param.data`
  itself, or `Tensor.numpy()`, which returns it — sees the update, the way it would in
  PyTorch. `state_dict()` and `detach()` copy, so neither is affected. The one ordering
  this changes is stepping *between* a forward pass and its own backward pass, where a
  reverse closure that captured a parameter's array would now see post-step values;
  most of the library already read parameters through the Tensor at reverse time
  (`docs/DESIGN.md` §14), so this makes the behaviour uniform rather than introducing
  it.
- **`effective_gradient` returns a read-only array.** It aliases `param.grad` when
  there is nothing to do, which is the common case and the whole point of it; writing
  through the result would corrupt the gradient the caller still owns, so NumPy raises
  instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from pynn.core.module import Module
from pynn.core.tensor import Tensor
from pynn.core.types import Array

# Only `Optimizer` is re-exported from `pynn.core`. The three names below are the
# optimizer-authoring surface and are imported by path, the way `ParameterSource`
# already is.
__all__ = ["Optimizer"]

#: What an optimizer accepts: a whole model, or explicit parameter groups.
ParameterSource = Module | Sequence[dict[str, Tensor]]


def effective_gradient(
    grad: Array, data: Array, weight_decay: float, maximize: bool
) -> Array:
    """The gradient an update rule consumes, before any per-optimizer state.

    Two adjustments, in the order PyTorch applies them: negate it to ascend the
    objective rather than descend it, then fold in coupled weight decay. Both are
    skipped when they would be the identity — `grad + 0.0 * data` is two full-size
    allocations to compute `grad`, and it ran on every parameter of every step.

    Parameters
    ----------
    grad : Array
        The parameter's gradient. Never modified.
    data : Array
        The parameter's current value, for the decay term.
    weight_decay : float
        Coupled L2 penalty, folded into the gradient. AdamW deliberately does not use
        this — decoupling it from the adaptive rescaling is the point of that
        optimizer.
    maximize : bool
        Ascend rather than descend.

    Returns
    -------
    Array
        A **read-only** array, because when neither adjustment applies it is a view of
        `grad` rather than a copy. An update rule that needs a mutable array of its
        own must say so with `np.copy`.
    """

    if maximize:
        g = -grad
        if weight_decay:
            g += weight_decay * data
    elif weight_decay:
        g = grad + weight_decay * data
    else:
        # Nothing to do. A view rather than the array itself, so that marking it
        # read-only guards this function's callers without touching `param.grad`.
        g = grad.view()

    g.flags.writeable = False
    return g


def state_buffer(state: dict[str, Array], key: str, like: Array) -> Array:
    """The per-parameter buffer `state[key]`, zero-filled on the step that needs it.

    Returned rather than looked up again on each line: the buffers are mutated in
    place, so binding one to a local name once is both faster and how the update rule
    reads on the page.

    Parameters
    ----------
    state : dict[str, Array]
        One of the optimizer's per-buffer dictionaries, e.g. `cache["velocity"]`.
    key : str
        The parameter's name within its group.
    like : Array
        Shape and dtype for a buffer that does not exist yet. Layers build their
        parameters lazily, so this is the first step that has seen this parameter.
    """
    buffer = state.get(key)
    if buffer is None:
        buffer = np.zeros_like(like)
        state[key] = buffer
    return buffer


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

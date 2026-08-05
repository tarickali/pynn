"""Global control over whether operations are recorded on the autodiff tape.

Every operation normally records the node it produced: its children, and the closure
that pushes a gradient back to them. During inference that record is pure overhead —
built, never differentiated, and thrown away — and it is worse than overhead when the
outputs are kept, because each closure holds the forward pass's intermediate arrays
alive. Evaluating on a validation set and collecting the predictions therefore retains
the whole graph for every batch.

`no_grad` turns recording off for a block::

    model.eval()
    with no_grad():
        predictions = [model(Tensor(batch)) for batch in batches]

Tensors produced inside the block have no children and no reverse function, so they
report `requires_grad == False` and `backward()` on one raises rather than silently
returning a zero gradient.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["enable_grad", "is_grad_enabled", "no_grad", "set_grad_enabled"]

_grad_enabled = True


def is_grad_enabled() -> bool:
    """Whether operations are currently being recorded on the tape."""
    return _grad_enabled


@contextmanager
def set_grad_enabled(mode: bool) -> Iterator[None]:
    """Record operations, or not, for the duration of the block.

    Restores the previous setting on exit rather than turning recording back on, so
    that nesting works in either order.
    """
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = mode
    try:
        yield
    finally:
        _grad_enabled = previous


@contextmanager
def no_grad() -> Iterator[None]:
    """Stop recording operations for the duration of the block.

    Usable as a context manager or as a decorator::

        with no_grad():
            logits = model(X)

        @no_grad()
        def evaluate(model, X):
            return model(X)
    """
    with set_grad_enabled(False):
        yield


@contextmanager
def enable_grad() -> Iterator[None]:
    """Resume recording operations, even inside a `no_grad` block."""
    with set_grad_enabled(True):
        yield

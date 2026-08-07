"""Gradient clipping."""

from __future__ import annotations

import numpy as np

from pynn.core.optimizer import ParameterSource, as_parameter_groups

__all__ = ["clip_grad_norm", "clip_grad_value"]


def clip_grad_norm(
    parameters: ParameterSource, max_norm: float, norm_type: float = 2.0
) -> float:
    """Rescale gradients in place so their combined norm is at most `max_norm`.

    The norm is taken over *every* parameter at once, not per tensor, and the whole set
    is scaled by one factor. That is the point: scaling each tensor separately would
    change their relative sizes, which is to say it would change the direction of the
    step, and the direction is the part the gradient got right. Clipping is meant to
    shorten the step, not turn it.

    Under it are the runs that diverge on one bad batch — a long sequence, a rare label,
    a `log` that got close to zero — where a single enormous update undoes an epoch of
    progress. Clipping bounds the damage without touching anything else.

    Parameters
    ----------
    parameters : ParameterSource
        A Module, or explicit parameter groups. Frozen parameters are included: they
        still hold gradients, and the total norm is a property of the gradients.
    max_norm : float
        Ceiling for the total norm.
    norm_type : float, default 2.0
        Order of the norm. `float("inf")` gives the maximum absolute gradient.

    Returns
    -------
    float
        The total norm *before* clipping, which is the number worth logging — it says
        whether clipping is doing anything, and a norm that climbs across epochs is the
        signal that a run is about to come apart.

    Raises
    ------
    ValueError
        If `max_norm` is not positive.

    Examples
    --------
    >>> loss.backward()                                  # doctest: +SKIP
    >>> clip_grad_norm(model, max_norm=1.0)              # doctest: +SKIP
    >>> optimizer.step()                                 # doctest: +SKIP
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    gradients = [
        parameter.grad
        for group in as_parameter_groups(parameters)
        for parameter in group.values()
    ]
    if not gradients:
        return 0.0

    if norm_type == float("inf"):
        total = max(float(np.abs(gradient).max()) for gradient in gradients)
    else:
        total = float(
            sum(float(np.sum(np.abs(gradient) ** norm_type)) for gradient in gradients)
            ** (1.0 / norm_type)
        )

    if total > max_norm:
        # A small epsilon keeps the scale finite if the norm is exactly at the limit
        # after rounding, matching PyTorch.
        scale = max_norm / (total + 1e-6)
        for gradient in gradients:
            gradient *= scale

    return total


def clip_grad_value(parameters: ParameterSource, clip_value: float) -> None:
    """Clamp every gradient element in place to ``[-clip_value, clip_value]``.

    Cruder than `clip_grad_norm` and occasionally what you want: it bounds each
    element independently, so it *does* change the step direction, but it cannot be
    defeated by a single component large enough to dominate the norm.

    Parameters
    ----------
    parameters : ParameterSource
        A Module, or explicit parameter groups.
    clip_value : float
        Symmetric bound on each element.

    Raises
    ------
    ValueError
        If `clip_value` is not positive.
    """
    if clip_value <= 0:
        raise ValueError(f"clip_value must be positive, got {clip_value}")

    for group in as_parameter_groups(parameters):
        for parameter in group.values():
            np.clip(parameter.grad, -clip_value, clip_value, out=parameter.grad)

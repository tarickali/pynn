"""Numerical stability checks.

Every function here asserts that a computation stays finite for inputs far outside the
range where a naive formulation would work. The naive forms overflow around
``|x| > 709``, the point where ``exp(x)`` leaves float64 range, so the checks probe well
past that: an activation that returns ``inf`` produces ``nan`` gradients one step later,
which is the kind of failure that shows up only on real data.

Checks run under ``np.errstate(over="raise", invalid="raise")`` where relevant, so a
silent ``RuntimeWarning: overflow encountered in exp`` counts as a failure rather than
being lost in the log.
"""

from __future__ import annotations

import numpy as np

import pynn.core.math as pmath
import pynn.functional as F
from pynn.core import Tensor
from pynn.core.numeric import stable_sigmoid
from pynn.functional.losses import binary_crossentropy, categorical_crossentropy
from pynn.verify.report import CheckReport

__all__ = ["check_stability"]

#: Magnitudes chosen to sit far beyond the float64 overflow point of exp (~709).
EXTREME = np.array([-1000.0, -800.0, -50.0, 0.0, 50.0, 800.0, 1000.0])


def _finite(array: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(array)))


def _strict():
    """Turn floating-point warnings into errors, so a silent overflow fails the check.

    Returns a fresh context manager on every call; an `np.errstate` instance cannot be
    entered twice.
    """
    return np.errstate(over="raise", invalid="raise", divide="raise")


def check_stability() -> CheckReport:
    """Verify that activations and losses stay finite at extreme input magnitudes.

    Returns
    -------
    CheckReport
        One result per property checked.
    """

    report = CheckReport(name="stability")
    # --- stable_sigmoid --------------------------------------------------- #
    try:
        with _strict():
            values = stable_sigmoid(EXTREME)
        report.add(
            "stable_sigmoid is finite at |x| up to 1000",
            _finite(values) and bool(np.all((values >= 0.0) & (values <= 1.0))),
            f"range [{values.min():.3g}, {values.max():.3g}]",
        )
    except FloatingPointError as error:
        report.add("stable_sigmoid is finite at |x| up to 1000", False, str(error))

    # --- sigmoid ---------------------------------------------------------- #
    try:
        with _strict():
            values = F.sigmoid(Tensor(EXTREME)).data
        report.add(
            "sigmoid does not overflow",
            _finite(values),
            f"sigmoid(-800)={values[1]:.1f}, sigmoid(800)={values[5]:.1f}",
        )
    except FloatingPointError as error:
        report.add("sigmoid does not overflow", False, str(error))

    # --- softplus --------------------------------------------------------- #
    # log(1 + exp(x)) computed naively returns inf for x > ~709. softplus(x) -> x
    # for large x, so the correct answer is simply 800.0.
    try:
        with _strict():
            values = F.softplus(Tensor(EXTREME)).data
        report.add(
            "softplus does not overflow",
            _finite(values) and np.isclose(values[5], 800.0),
            f"softplus(800)={values[5]:.1f} (expected 800.0)",
        )
    except FloatingPointError as error:
        report.add("softplus does not overflow", False, str(error))

    report.add(
        "softplus is non-negative",
        bool(np.all(F.softplus(Tensor(EXTREME)).data >= 0.0)),
    )

    # --- elu and selu ----------------------------------------------------- #
    for name, activation in [("elu", F.elu), ("selu", F.selu)]:
        try:
            with _strict():
                values = activation(Tensor(EXTREME)).data
            report.add(f"{name} does not overflow", _finite(values))
        except FloatingPointError as error:
            report.add(f"{name} does not overflow", False, str(error))

    # --- softmax ---------------------------------------------------------- #
    try:
        with _strict():
            probabilities = F.softmax(Tensor(EXTREME.reshape(1, -1))).data
        report.add(
            "softmax is finite at extreme logits",
            _finite(probabilities),
        )
        report.add(
            "softmax rows sum to 1 at extreme logits",
            bool(np.allclose(probabilities.sum(axis=-1), 1.0)),
            f"row sum {probabilities.sum():.6f}",
        )
    except FloatingPointError as error:
        report.add("softmax is finite at extreme logits", False, str(error))

    # --- log -------------------------------------------------------------- #
    # log(0) is -inf mathematically; the implementation clamps its input so that a
    # zero probability cannot poison an entire loss.
    values = pmath.log(Tensor(np.array([0.0, 1e-300, 1.0]))).data
    report.add(
        "log(0) is finite (input is clamped)",
        _finite(values),
        f"log(0)={values[0]:.3g}",
    )

    tensor = Tensor(np.array([0.0, 1.0]))
    pmath.sum(pmath.log(tensor)).backward()
    report.add("log gradient at 0 is finite", _finite(tensor.grad))

    # --- losses at extreme logits ----------------------------------------- #
    # A confidently wrong prediction is the worst case: the loss is large but must
    # stay finite, and so must the gradient it sends back.
    logits = Tensor(np.array([[-800.0], [800.0]]))
    targets = Tensor(np.array([[1.0], [0.0]]))
    loss = binary_crossentropy(targets, logits, logits=True)
    loss.backward()
    report.add(
        "binary crossentropy is finite for confidently wrong logits",
        _finite(loss.data),
        f"loss={float(loss.data):.1f}",
    )
    report.add(
        "binary crossentropy gradient is finite for confidently wrong logits",
        _finite(logits.grad),
    )

    class_logits = Tensor(np.array([[-800.0, 800.0, 0.0]]))
    class_targets = Tensor(np.array([[1.0, 0.0, 0.0]]))
    class_loss = categorical_crossentropy(class_targets, class_logits, logits=True)
    class_loss.backward()
    report.add(
        "categorical crossentropy is finite for confidently wrong logits",
        _finite(class_loss.data),
        f"loss={float(class_loss.data):.1f}",
    )
    report.add(
        "categorical crossentropy gradient is finite for confidently wrong logits",
        _finite(class_logits.grad),
    )

    # --- gradients at extreme magnitudes ---------------------------------- #
    for name, activation in [
        ("sigmoid", F.sigmoid),
        ("softplus", F.softplus),
        ("tanh", F.tanh),
        ("elu", F.elu),
        ("selu", F.selu),
        ("softmax", F.softmax),
    ]:
        tensor = Tensor(EXTREME.reshape(1, -1))
        pmath.sum(activation(tensor)).backward()
        report.add(f"{name} gradient is finite at extreme inputs", _finite(tensor.grad))

    return report

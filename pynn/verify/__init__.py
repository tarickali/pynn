"""Self-verification suite for PyNN.

An autodiff library can be wrong while looking healthy: training loss still falls when a
gradient is scaled by the batch size, and an activation that overflows to ``inf`` only
poisons the run once the inputs get large enough. This package makes those failures
observable.

Three suites, runnable individually or together:

- `check_all_gradients` compares analytic gradients against central differences for
  every operator, activation, loss, and module the library ships, plus graph shapes
  where a tensor has multiple consumers.
- `check_stability` asserts activations and losses stay finite far past the point where
  ``exp`` overflows in float64.
- `check_invariants` asserts behavioral properties of the tape, the optimizers (against
  closed-form transcriptions of their published update rules), and the public factories.

Run everything from the command line::

    python -m pynn.verify            # failures only
    python -m pynn.verify --verbose  # every check

Or from Python::

    import pynn.verify

    report = pynn.verify.run_all()
    assert report.passed, report

`check_gradients` is also useful on its own for verifying a custom layer::

    from pynn.core import Tensor
    from pynn.verify import check_gradients

    result = check_gradients(lambda ts: my_loss(ts[0]), [Tensor(x)])
    assert result.passed, result

The `tests/` suite covers the same ground for CI. This package exists so the checks can
be run against an installed copy of PyNN without pytest or the test files present.
"""

from pynn.verify.gradients import (
    DEFAULT_SEED,
    GradCheckResult,
    GradientCase,
    InputCheck,
    analytic_gradient,
    check_all_gradients,
    check_case,
    check_gradients,
    gradient_cases,
    numerical_gradient,
)
from pynn.verify.invariants import (
    check_api,
    check_autodiff,
    check_invariants,
    check_optimizers,
)
from pynn.verify.report import CheckReport, CheckResult
from pynn.verify.stability import check_stability

__all__ = [
    "DEFAULT_SEED",
    "CheckReport",
    "CheckResult",
    "GradCheckResult",
    "GradientCase",
    "InputCheck",
    "analytic_gradient",
    "check_all_gradients",
    "check_api",
    "check_autodiff",
    "check_case",
    "check_gradients",
    "check_invariants",
    "check_optimizers",
    "check_stability",
    "gradient_cases",
    "numerical_gradient",
    "run_all",
]


def run_all() -> CheckReport:
    """Run the gradient, stability, and invariant suites.

    Returns
    -------
    CheckReport
        Every result, flattened. Use ``report.passed`` for a single verdict and
        ``report.format(verbose=True)`` for the full listing.
    """

    report = CheckReport(name="pynn.verify")
    for suite in (check_all_gradients(), check_stability(), check_invariants()):
        report.extend(suite)
    return report

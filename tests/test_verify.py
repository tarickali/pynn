"""Run the shipped self-verification suites under pytest.

`pynn.verify` exists so the checks can be run against an installed copy of PyNN without
the test files present. Driving it from pytest as well keeps it from rotting: a suite
that stops being run stops being true.

The gradient suite is not here — `tests/test_gradcheck.py` parametrizes over its cases
individually, which gives better failure messages than one test for the whole sweep.
"""

import pytest

from pynn.verify import check_invariants, check_stability

SUITES = {
    "stability": check_stability,
    "invariants": check_invariants,
}


@pytest.mark.parametrize("name", sorted(SUITES))
def test_verify_suite_passes(name: str) -> None:
    report = SUITES[name]()
    assert report.results, f"{name} ran no checks"
    assert report.passed, "\n" + report.format(verbose=False)


def test_run_all_covers_every_suite() -> None:
    """`python -m pynn.verify` must run the gradient suite too, even though pytest
    reaches it through test_gradcheck.py instead."""
    from pynn.verify import run_all

    report = run_all()
    assert report.passed, "\n" + report.format(verbose=False)
    assert len(report.results) > 180, f"run_all only ran {len(report.results)} checks"

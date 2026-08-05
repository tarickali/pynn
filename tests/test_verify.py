"""Run the shipped self-verification suites under pytest.

`pynn.verify` exists so the checks can be run against an installed copy of PyNN without
the test files present. Driving it from pytest as well keeps it from rotting: a suite
that stops being run stops being true.
"""

import pytest

from pynn.verify import check_all_gradients, check_invariants, check_stability

SUITES = {
    "gradients": check_all_gradients,
    "stability": check_stability,
    "invariants": check_invariants,
}


@pytest.mark.parametrize("name", sorted(SUITES))
def test_verify_suite_passes(name: str) -> None:
    report = SUITES[name]()
    assert report.results, f"{name} ran no checks"
    assert report.passed, "\n" + report.format(verbose=False)

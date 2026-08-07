"""Run the shipped self-verification suites under pytest.

`pynn.verify` exists so the checks can be run against an installed copy of PyNN without
the test files present. Driving it from pytest as well keeps it from rotting: a suite
that stops being run stops being true.

The gradient suite is not here — `tests/test_gradcheck.py` parametrizes over its cases
individually, which gives better failure messages than one test for the whole sweep.
"""

import pytest

from pynn.verify import CheckReport, CheckResult, check_invariants, check_stability
from pynn.verify.__main__ import main

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


# --------------------------------------------------------------------------- #
# Reporting
#
# A report that renders a failure as a pass, or a command that exits 0 on a failed
# suite, would make every check above vacuous.
# --------------------------------------------------------------------------- #


def test_a_report_summarizes_its_failures() -> None:
    report = CheckReport(name="demo")
    report.add("first", True, "0.0")
    report.add("second", False, "off by 3")

    assert not report.passed
    assert [result.name for result in report.failures] == ["second"]
    assert report.summary() == "demo: 1/2 passed (1 FAILED)"


def test_an_all_passing_report_summarizes_as_ok() -> None:
    report = CheckReport(name="demo")
    report.add("only", True)

    assert report.passed
    assert report.summary() == "demo: 1/1 passed (OK)"


def test_format_lists_only_failures_unless_verbose() -> None:
    report = CheckReport(name="demo")
    report.add("kept quiet", True)
    report.add("shown", False, "why")

    assert "kept quiet" not in report.format()
    assert "kept quiet" in report.format(verbose=True)
    assert "[FAIL] shown — why" in str(report)


def test_extend_merges_another_report() -> None:
    report = CheckReport(name="outer")
    other = CheckReport(name="inner")
    other.add("inherited", True)

    report.extend(other)

    assert [result.name for result in report.results] == ["inherited"]


def test_a_result_renders_its_pass_mark() -> None:
    assert str(CheckResult("named", True)) == "[pass] named"
    assert str(CheckResult("named", True, "detail")) == "[pass] named — detail"


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def test_main_runs_a_named_suite(capsys) -> None:
    assert main(["stability"]) == 0

    output = capsys.readouterr().out
    assert "stability:" in output
    assert "pynn.verify:" in output


def test_main_verbose_lists_individual_checks(capsys) -> None:
    assert main(["stability", "--verbose"]) == 0
    assert "[pass]" in capsys.readouterr().out


def test_main_rejects_an_unknown_suite() -> None:
    with pytest.raises(SystemExit):
        main(["not-a-suite"])


def test_main_with_no_arguments_runs_every_suite(capsys) -> None:
    """`python -m pynn.verify` with no arguments is what CI runs and what the README
    documents, and it is the one invocation `nargs="*"` combined with `choices=` got
    wrong: argparse validated the empty default against the choice list and exited 2
    with "invalid choice: []" on Python 3.10 and 3.11.
    """
    assert main([]) == 0

    output = capsys.readouterr().out
    for suite in ("gradients", "stability", "invariants"):
        assert f"{suite}:" in output

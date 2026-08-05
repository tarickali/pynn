"""Command-line entry point: ``python -m pynn.verify``."""

from __future__ import annotations

import argparse
import sys

from pynn.verify import check_all_gradients, check_invariants, check_stability
from pynn.verify.report import CheckReport

SUITES = {
    "gradients": check_all_gradients,
    "stability": check_stability,
    "invariants": check_invariants,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pynn.verify",
        description=(
            "Verify a PyNN installation: gradients, numerical stability, invariants."
        ),
    )
    parser.add_argument(
        "suites",
        nargs="*",
        choices=sorted(SUITES),
        help="Suites to run. Defaults to all of them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List every check, not just the failures.",
    )
    args = parser.parse_args(argv)

    selected = args.suites or sorted(SUITES)
    overall = CheckReport(name="pynn.verify")

    for name in selected:
        report = SUITES[name]()
        print(report.format(verbose=args.verbose))
        overall.extend(report)

    print()
    print(overall.summary())
    return 0 if overall.passed else 1


if __name__ == "__main__":
    sys.exit(main())

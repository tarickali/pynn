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
        # Deliberately no `choices=`: combined with `nargs="*"`, argparse validates the
        # empty default against the choice list on Python 3.10 and 3.11, so running
        # this with no arguments at all — the common case — exits 2 with
        # "invalid choice: []". Fixed in 3.12; validated by hand below so that the
        # command behaves the same on every supported version.
        metavar="{" + ",".join(sorted(SUITES)) + "}",
        help="Suites to run. Defaults to all of them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List every check, not just the failures.",
    )
    args = parser.parse_args(argv)

    unknown = [name for name in args.suites if name not in SUITES]
    if unknown:
        parser.error(
            f"unknown suite {unknown[0]!r} (choose from {', '.join(sorted(SUITES))})"
        )

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

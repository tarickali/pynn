"""Result types shared by the verification checks."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CheckReport", "CheckResult"]


@dataclass
class CheckResult:
    """Outcome of a single named check.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"softplus does not overflow"``.
    passed : bool
        Whether the property held.
    detail : str
        Measured values or the reason for failure. Shown for failures, and for
        passes when running in verbose mode.
    """

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        suffix = f" — {self.detail}" if self.detail else ""
        return f"[{mark}] {self.name}{suffix}"


@dataclass
class CheckReport:
    """A group of check results, produced by one verification suite."""

    name: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if not result.passed]

    def add(self, name: str, passed: bool, detail: str = "") -> CheckResult:
        result = CheckResult(name=name, passed=passed, detail=detail)
        self.results.append(result)
        return result

    def extend(self, other: CheckReport) -> None:
        self.results.extend(other.results)

    def summary(self) -> str:
        total = len(self.results)
        failed = len(self.failures)
        status = "OK" if self.passed else f"{failed} FAILED"
        return f"{self.name}: {total - failed}/{total} passed ({status})"

    def format(self, verbose: bool = False) -> str:
        lines = [self.summary()]
        shown = self.results if verbose else self.failures
        lines.extend(f"  {result}" for result in shown)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()

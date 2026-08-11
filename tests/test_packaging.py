"""The packaging metadata, asserted rather than assumed.

Packaging fails quietly. A version bumped in one place and not the other produces a
wheel that reports the wrong number; a missing `py.typed` produces a package that is
fully annotated and reads as untyped to every consumer. Neither breaks an import, so
neither is visible from inside the library — the first person to notice is someone who
installed it.
"""

from pathlib import Path

import pytest

import pynn

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def read_pyproject() -> dict:
    """Parse `pyproject.toml`, or skip on the one interpreter that cannot.

    `tomllib` is standard library from 3.11. Skipping on 3.10 rather than taking a
    dependency on `tomli` is safe here: the file being checked is static
    configuration, so the other four interpreters in the CI matrix assert exactly the
    same thing about exactly the same bytes.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
    return dict(tomllib.loads(PYPROJECT.read_text()))


def test_version_matches_the_installed_distribution() -> None:
    """`pynn.__version__` and the installed metadata must agree.

    They come from different places — the attribute is read out of the source tree,
    the metadata out of the `.dist-info` setuptools wrote at install time — so this
    fails when the version has been bumped without reinstalling, and when the
    `[tool.setuptools.dynamic]` wiring that connects them has been broken.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("pynn")
    except PackageNotFoundError:  # pragma: no cover - CI always installs the package
        pytest.skip("pynn is not installed; run `pip install -e .`")

    assert pynn.__version__ == installed, (
        f"pynn.__version__ is {pynn.__version__} but the installed distribution "
        f"reports {installed}; reinstall with `pip install -e .`"
    )


def test_pyproject_reads_the_version_from_the_package() -> None:
    """There must be no second version literal for the first one to drift from."""
    project = read_pyproject()["project"]

    assert project["dynamic"] == ["version"]
    assert "version" not in project


def test_the_type_marker_exists() -> None:
    """PEP 561's marker, next to the package it marks."""
    assert (Path(pynn.__file__).parent / "py.typed").is_file()


def test_the_type_marker_is_declared_as_package_data() -> None:
    """The marker existing in the tree is not the same as the marker shipping.

    setuptools includes `.py` files on its own and nothing else, so a `py.typed` that
    is not named here is in the repository and absent from every wheel built from it.
    """
    package_data = read_pyproject()["tool"]["setuptools"]["package-data"]

    assert "py.typed" in package_data["pynn"]

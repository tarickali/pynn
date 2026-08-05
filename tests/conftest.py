"""Shared fixtures.

A seeded `numpy.random.Generator` rather than the legacy global `numpy.random`: an
unseeded test that fails once and then passes on re-run tells you nothing, and the
legacy global functions cannot be seeded independently per test.
"""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh generator, identically seeded for every test that asks for one."""
    return np.random.default_rng(20240605)

"""Initializer distributions and reproducibility.

Layers construct their initializers through the string/factory interface and cannot be
handed a generator, so `set_seed` is the only hook that makes a whole model's
initialization reproducible. A run that cannot be repeated cannot be debugged.
"""

import numpy as np
import pytest

from pynn.functional.initializers import (
    constant,
    he_normal,
    he_uniform,
    lecun_normal,
    lecun_uniform,
    ones,
    random_normal,
    random_uniform,
    set_seed,
    xavier_normal,
    xavier_uniform,
    zeros,
)

RANDOM = [
    he_normal,
    he_uniform,
    lecun_normal,
    lecun_uniform,
    random_normal,
    random_uniform,
    xavier_normal,
    xavier_uniform,
]


def test_deterministic_initializers():
    assert np.array_equal(zeros((2, 3)).data, np.zeros((2, 3)))
    assert np.array_equal(ones((2, 3)).data, np.ones((2, 3)))
    assert np.array_equal(constant((2, 3), 4.5).data, np.full((2, 3), 4.5))


@pytest.mark.parametrize("initializer", RANDOM, ids=lambda f: f.__name__)
def test_set_seed_makes_initialization_reproducible(initializer):
    set_seed(0)
    first = initializer((4, 3)).data
    set_seed(0)
    second = initializer((4, 3)).data

    assert np.array_equal(first, second)


@pytest.mark.parametrize("initializer", RANDOM, ids=lambda f: f.__name__)
def test_different_seeds_give_different_weights(initializer):
    set_seed(0)
    first = initializer((4, 3)).data
    set_seed(1)
    second = initializer((4, 3)).data

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("initializer", RANDOM, ids=lambda f: f.__name__)
def test_an_explicit_generator_bypasses_the_module_generator(initializer):
    set_seed(0)
    first = initializer((4, 3), rng=np.random.default_rng(99)).data
    set_seed(123)
    second = initializer((4, 3), rng=np.random.default_rng(99)).data

    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "initializer,fan_scale",
    [
        (xavier_normal, lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out))),
        (he_normal, lambda fan_in, _: np.sqrt(2.0 / fan_in)),
        (lecun_normal, lambda fan_in, _: np.sqrt(1.0 / fan_in)),
    ],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_normal_initializers_use_the_documented_fan_scaling(initializer, fan_scale):
    """A wrong fan term is invisible in a shape test and lethal in a deep network."""
    fan_in, fan_out = 400, 200
    values = initializer((fan_in, fan_out), rng=np.random.default_rng(0)).data

    assert values.std() == pytest.approx(fan_scale(fan_in, fan_out), rel=0.05)
    assert values.mean() == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize(
    "initializer,limit",
    [
        (xavier_uniform, lambda fan_in, fan_out: np.sqrt(6.0 / (fan_in + fan_out))),
        (he_uniform, lambda fan_in, _: np.sqrt(6.0 / fan_in)),
        (lecun_uniform, lambda fan_in, _: np.sqrt(3.0 / fan_in)),
    ],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_uniform_initializers_respect_their_limits(initializer, limit):
    fan_in, fan_out = 400, 200
    values = initializer((fan_in, fan_out), rng=np.random.default_rng(0)).data
    bound = limit(fan_in, fan_out)

    assert values.min() >= -bound
    assert values.max() <= bound
    assert values.max() == pytest.approx(bound, rel=0.02)


def test_random_uniform_and_normal_honor_their_parameters():
    values = random_uniform((5000,), low=-2.0, high=-1.0, rng=np.random.default_rng(0))
    assert values.data.min() >= -2.0
    assert values.data.max() <= -1.0

    values = random_normal((5000,), mean=3.0, std=0.5, rng=np.random.default_rng(0))
    assert values.data.mean() == pytest.approx(3.0, abs=0.05)
    assert values.data.std() == pytest.approx(0.5, rel=0.05)

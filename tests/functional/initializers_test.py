"""Initializer distributions and reproducibility.

Layers construct their initializers through the string/factory interface and cannot be
handed a generator, so `set_seed` is the only hook that makes a whole model's
initialization reproducible. A run that cannot be repeated cannot be debugged.
"""

import numpy as np
import pytest

from pynn.functional.initializers import (
    constant,
    fans,
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


# --------------------------------------------------------------------------- #
# Fan computation
#
# The scale every variance-scaling initializer picks is a function of the fan-in and
# fan-out. Reading them off the wrong axes leaves the weights at a plausible
# magnitude, so nothing looks wrong until a deep stack diverges at a learning rate a
# correctly initialized one handles.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((784, 256), (784, 256)),  # Linear(784, 256): (in_features, out_features)
        ((4, 3), (4, 3)),
        ((32, 16, 3, 3), (144, 288)),  # Conv2d(16, 32, 3): in_ch * kh * kw
        ((16, 1, 3, 3), (9, 144)),  # Conv2d(1, 16, 3)
        ((8, 4, 5, 5), (100, 200)),
        ((10,), (10, 10)),  # a bias vector has no meaningful direction
    ],
    ids=str,
)
def test_fans_reads_the_right_axes(shape, expected):
    assert fans(shape) == expected


def test_a_conv_kernel_is_scaled_by_its_receptive_field():
    """`sqrt(2 / shape[0])` would use the *output* channel count and ignore kh*kw.

    For this kernel that is 0.25 instead of 0.118 — a factor of 2.1, which compounds
    through a stack until the logits are large enough to stall training.
    """
    kernel = he_normal((32, 16, 3, 3), rng=np.random.default_rng(0)).data

    assert kernel.std() == pytest.approx(np.sqrt(2.0 / (16 * 3 * 3)), rel=0.05)
    assert kernel.std() != pytest.approx(np.sqrt(2.0 / 32), rel=0.05)


@pytest.mark.parametrize("initializer", RANDOM, ids=lambda f: f.__name__)
def test_conv_and_equivalent_dense_shapes_get_the_same_scale(initializer):
    """A kernel with fan-in 144 must be scaled like a dense weight with fan-in 144."""
    conv = initializer((32, 16, 3, 3), rng=np.random.default_rng(0)).data
    dense = initializer((144, 288), rng=np.random.default_rng(0)).data

    assert conv.std() == pytest.approx(dense.std(), rel=0.05)


def test_activation_scale_is_preserved_through_a_conv_stack():
    """He initialization exists to hold the activation scale steady through depth."""
    from pynn.core import Tensor
    from pynn.nn import Conv2d, Sequential

    set_seed(0)
    model = Sequential(
        [
            Conv2d(
                1,
                16,
                3,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                include_bias=False,
            ),
            Conv2d(
                16,
                16,
                3,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                include_bias=False,
            ),
            Conv2d(
                16,
                16,
                3,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                include_bias=False,
            ),
        ]
    )
    x = Tensor(np.random.default_rng(0).standard_normal((16, 1, 12, 12)))

    activations = x
    scales = []
    for layer in model:
        activations = layer(activations)
        scales.append(activations.data.std())

    # ReLU halves the variance and He's factor of 2 restores it, so the scale should
    # hold rather than compound. Reading fan-in off the wrong axis grew it ~2x a layer.
    assert scales[-1] / scales[0] == pytest.approx(1.0, abs=0.45)


def test_random_uniform_and_normal_honor_their_parameters():
    values = random_uniform((5000,), low=-2.0, high=-1.0, rng=np.random.default_rng(0))
    assert values.data.min() >= -2.0
    assert values.data.max() <= -1.0

    values = random_normal((5000,), mean=3.0, std=0.5, rng=np.random.default_rng(0))
    assert values.data.mean() == pytest.approx(3.0, abs=0.05)
    assert values.data.std() == pytest.approx(0.5, rel=0.05)

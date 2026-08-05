"""Layer-level behavior: parameter shapes, lazy building, and shape contracts."""

import numpy as np
import pytest

import pynn.core.math as pmath
from pynn.core import Tensor
from pynn.nn import Activation, Conv2d, Flatten, Linear

# --------------------------------------------------------------------------- #
# Conv2d bias
#
# A convolution has one bias per output channel, shared across every spatial
# position. Allocating one bias per output *position* instead still trains and
# still passes a gradient check, so these are shape and invariance assertions
# rather than numerical ones.
# --------------------------------------------------------------------------- #


def build_conv(height: int = 8, width: int | None = None, **kwargs) -> Conv2d:
    width = height if width is None else width
    layer = Conv2d(**kwargs)
    layer(Tensor(np.zeros((1, kwargs["in_channels"], height, width))))
    return layer


def test_conv2d_bias_is_one_per_output_channel():
    layer = build_conv(in_channels=2, out_channels=3, kernel_size=3)

    assert layer.parameters["B"].shape == (3, 1, 1)
    assert layer.parameters["B"].data.size == 3


@pytest.mark.parametrize("size", [8, 16, 32])
def test_conv2d_parameter_count_is_independent_of_input_size(size):
    """A per-position bias would tie the parameter count to the input resolution."""
    layer = build_conv(
        size, in_channels=2, out_channels=3, kernel_size=3, padding="same"
    )

    total = sum(p.data.size for p in layer.parameters.values())
    assert total == 3 * 2 * 3 * 3 + 3


def test_conv2d_bias_is_shared_across_spatial_positions(rng):
    """With a zero kernel, every output position must equal the same channel bias."""
    layer = build_conv(
        6, in_channels=2, out_channels=3, kernel_size=3, kernel_initializer="zeros"
    )
    bias = np.array([0.5, -1.5, 2.0]).reshape(3, 1, 1)
    layer.parameters["B"].data = bias

    output = layer(Tensor(rng.standard_normal((4, 2, 6, 6))))
    expected = np.broadcast_to(bias, output.data.shape)
    assert np.allclose(output.data, expected)


def test_conv2d_bias_gradient_sums_over_batch_and_positions(rng):
    """dL/dB[c] sums the incoming gradient over batch and every spatial position."""
    layer = build_conv(6, in_channels=2, out_channels=3, kernel_size=3)
    X = Tensor(rng.standard_normal((4, 2, 6, 6)))

    output = layer(X)
    pmath.sum(output).backward()

    batch, _, out_h, out_w = output.shape
    expected = np.full((3, 1, 1), float(batch * out_h * out_w))
    assert np.allclose(layer.parameters["B"].grad, expected)


def test_conv2d_is_translation_equivariant(rng):
    """Shifting the input shifts the output. A per-position bias breaks this."""
    layer = build_conv(9, in_channels=1, out_channels=2, kernel_size=3)
    layer.parameters["B"].data = np.array([0.5, -1.5]).reshape(2, 1, 1)

    X = np.zeros((1, 1, 9, 9))
    X[0, 0, 2:5, 2:5] = rng.standard_normal((3, 3))

    shifted = np.zeros_like(X)
    shifted[0, 0, 4:7, 2:5] = X[0, 0, 2:5, 2:5]

    base = layer(Tensor(X)).data
    moved = layer(Tensor(shifted)).data

    # A shift of 2 rows in the input is a shift of 2 rows in the output at stride 1.
    assert np.allclose(base[:, :, :-2, :], moved[:, :, 2:, :])


# --------------------------------------------------------------------------- #
# Freezing
# --------------------------------------------------------------------------- #


def test_freeze_marks_existing_parameters():
    layer = Linear(4, 3)
    layer(Tensor(np.zeros((2, 4))))

    layer.freeze()
    assert not layer.trainable
    assert all(not p.trainable for p in layer.parameters.values())

    layer.unfreeze()
    assert layer.trainable
    assert all(p.trainable for p in layer.parameters.values())


def test_freeze_survives_a_later_lazy_build():
    """Freezing before the first forward pass has no parameters to mark yet."""
    layer = Linear(4, 3)
    layer.freeze()
    assert layer.parameters == {}

    layer(Tensor(np.zeros((2, 4))))

    assert layer.parameters, "build produced no parameters"
    assert all(not p.trainable for p in layer.parameters.values())


def test_register_parameter_inherits_module_trainability():
    layer = Flatten()
    layer.freeze()

    param = layer.register_parameter("w", np.zeros(3))
    assert not param.trainable
    assert layer.parameters["w"] is param


# --------------------------------------------------------------------------- #
# Shape contracts
# --------------------------------------------------------------------------- #


def test_linear_infers_in_features_from_the_first_input():
    layer = Linear(3)
    assert layer.in_features is None

    layer(Tensor(np.zeros((5, 4))))
    assert layer.in_features == 4
    assert layer.parameters["W"].shape == (4, 3)


def test_linear_rejects_a_mismatched_in_features():
    layer = Linear(4, 3)
    with pytest.raises(AssertionError):
        layer(Tensor(np.zeros((5, 7))))


def test_linear_rejects_too_many_dimensions():
    with pytest.raises(TypeError, match="1 or 2 positional"):
        Linear(1, 2, 3)


@pytest.mark.parametrize(
    "padding,expected",
    [
        (0, (2, 3, 6, 6)),
        (1, (2, 3, 8, 8)),
        ("same", (2, 3, 8, 8)),
        ("valid", (2, 3, 6, 6)),
    ],
    ids=str,
)
def test_conv2d_output_shape(padding, expected):
    layer = Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=padding)
    assert layer(Tensor(np.zeros((2, 2, 8, 8)))).shape == expected


def test_conv2d_rejects_an_unknown_padding_string():
    layer = Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding="reflect")
    with pytest.raises(ValueError, match="padding must be"):
        layer(Tensor(np.zeros((1, 1, 5, 5))))


def test_flatten_keeps_the_batch_axis():
    assert Flatten()(Tensor(np.zeros((4, 2, 3, 5)))).shape == (4, 30)


def test_activation_layer_has_no_parameters():
    layer = Activation("relu")
    layer(Tensor(np.zeros((2, 3))))
    assert layer.parameters == {}

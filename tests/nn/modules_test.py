"""Layer-level behavior: parameter shapes, lazy building, and shape contracts."""

import numpy as np
import pytest

import pynn.core.math as pmath
import pynn.nn.activations as activations
from pynn.core import Tensor
from pynn.nn import (
    Activation,
    Conv2d,
    Flatten,
    Identity,
    Linear,
    ReLU,
    Sequential,
    Unflatten,
)

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
    with pytest.raises(ValueError, match="expects 4 input features, got 7"):
        layer(Tensor(np.zeros((5, 7))))


# --------------------------------------------------------------------------- #
# Shape errors name the fix
#
# These were bare `assert`s, which state the condition and nothing else and are
# removed entirely by `python -O` — so a model run under optimization got a
# `matmul` shape error several frames away instead, or silently wrong parameters
# from a custom initializer. Every one of them is now a raise, and the tests below
# exist so that the messages are covered rather than merely present.
# --------------------------------------------------------------------------- #


def test_linear_rejects_a_non_matrix_input_and_names_flatten():
    with pytest.raises(ValueError, match=r"rank 2.*Flatten"):
        Linear(3)(Tensor(np.zeros((5, 2, 3))))


def test_linear_rechecks_the_width_on_every_call():
    """A layer built against one width and handed another used to die in `matmul`."""
    layer = Linear(3)
    layer(Tensor(np.zeros((5, 4))))  # infers in_features = 4

    with pytest.raises(ValueError, match=r"shape \(batch, 4\), got \(5, 6\)"):
        layer(Tensor(np.zeros((5, 6))))


def test_linear_reports_an_initializer_that_returns_the_wrong_shape():
    layer = Linear(4, 3)
    layer.weight_init = lambda shape: np.zeros((2, 2))

    with pytest.raises(ValueError, match="returned a weight of shape"):
        layer(Tensor(np.zeros((5, 4))))


def test_linear_reports_a_bias_initializer_that_returns_the_wrong_shape():
    layer = Linear(4, 3)
    layer.bias_init = lambda shape: np.zeros((7,))

    with pytest.raises(ValueError, match="returned a bias of shape"):
        layer(Tensor(np.zeros((5, 4))))


def test_linear_reports_an_activation_that_changes_the_shape():
    layer = Linear(4, 3)
    layer(Tensor(np.zeros((5, 4))))  # build first, so the activation is what fails
    layer.act_fn = lambda Z: Z[:, :2]

    with pytest.raises(ValueError, match="activation changed the shape"):
        layer(Tensor(np.zeros((5, 4))))


def test_conv2d_rejects_an_input_that_is_not_an_image():
    with pytest.raises(ValueError, match="rank 4"):
        Conv2d(in_channels=1, out_channels=2, kernel_size=3)(Tensor(np.zeros((5, 4))))


def test_conv2d_rejects_a_mismatched_channel_count():
    layer = Conv2d(in_channels=3, out_channels=2, kernel_size=3)
    with pytest.raises(ValueError, match="built for 3 input channels, got 1"):
        layer(Tensor(np.zeros((2, 1, 8, 8))))


def test_conv2d_rechecks_the_spatial_shape_on_every_call():
    """`padding='same'` is resolved at build time against those exact dimensions."""
    layer = Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding="same")
    layer(Tensor(np.zeros((2, 1, 8, 8))))

    with pytest.raises(
        ValueError, match=r"built for an input of shape \(batch, 1, 8, 8"
    ):
        layer(Tensor(np.zeros((2, 1, 12, 12))))


def test_conv2d_reports_an_initializer_that_returns_the_wrong_shape():
    layer = Conv2d(in_channels=1, out_channels=2, kernel_size=3)
    layer.kernel_init = lambda shape: np.zeros((1, 1, 1, 1))

    with pytest.raises(ValueError, match="returned a kernel of shape"):
        layer(Tensor(np.zeros((2, 1, 8, 8))))


def test_conv2d_reports_a_bias_initializer_that_returns_the_wrong_shape():
    layer = Conv2d(in_channels=1, out_channels=2, kernel_size=3)
    layer.bias_init = lambda shape: np.zeros((5, 1, 1))

    with pytest.raises(ValueError, match="returned a bias of shape"):
        layer(Tensor(np.zeros((2, 1, 8, 8))))


def test_the_shape_guards_survive_python_dash_o():
    """The whole point: `assert` is removed by `-O` and these must not be.

    Run in a subprocess, because `-O` is an interpreter flag rather than a runtime
    switch — there is no way to assert this from inside a normal test session.
    """
    import subprocess
    import sys

    program = (
        "import numpy as np;"
        "from pynn.core import Tensor;"
        "from pynn.nn import Linear;"
        "layer = Linear(4, 3);"
        "layer(Tensor(np.zeros((5, 7))))"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program], capture_output=True, text=True
    )

    assert result.returncode != 0
    assert "expects 4 input features, got 7" in result.stderr


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


# --------------------------------------------------------------------------- #
# Unflatten and Identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spelling", [(2, 3, 5), ((2, 3, 5),)], ids=["varargs", "tuple"]
)
def test_unflatten_splits_the_flattened_axis(spelling):
    assert Unflatten(*spelling)(Tensor(np.zeros((4, 30)))).shape == (4, 2, 3, 5)


def test_unflatten_infers_one_axis():
    assert Unflatten(2, -1)(Tensor(np.zeros((4, 30)))).shape == (4, 2, 15)


def test_unflatten_reads_the_batch_size_from_the_input():
    """The reason it is not a Reshape layer taking the whole target shape.

    A shape with the batch size baked in is right for every batch of an epoch except
    the last, shorter one.
    """
    layer = Unflatten(2, 3, 5)

    assert layer(Tensor(np.zeros((4, 30)))).shape == (4, 2, 3, 5)
    assert layer(Tensor(np.zeros((1, 30)))).shape == (1, 2, 3, 5)


def test_unflatten_inverts_flatten():
    X = Tensor(np.arange(120.0).reshape(4, 2, 3, 5))

    round_tripped = Unflatten(2, 3, 5)(Flatten()(X))

    assert round_tripped.shape == X.shape
    assert np.array_equal(round_tripped.data, X.data)


def test_unflatten_refuses_a_shape_that_does_not_fit():
    with pytest.raises(ValueError, match="cannot unflatten"):
        Unflatten(2, 3, 7)(Tensor(np.zeros((4, 30))))


def test_unflatten_reports_its_hyperparameters():
    assert Unflatten(2, 3, 5).hyperparameters == {"shape": (2, 3, 5)}


def test_identity_returns_its_input():
    X = Tensor(np.arange(6.0).reshape(2, 3))
    output = Identity()(X)

    assert np.array_equal(output.data, X.data)
    assert Identity().parameters == {}


def test_identity_is_a_module_so_a_container_can_hold_it():
    """The gap it fills: the stateless `Activation` of the same name cannot."""
    model = Sequential([Linear(3, 4), Identity(), Linear(4, 2)])

    assert model(Tensor(np.zeros((2, 3)))).shape == (2, 2)
    assert len(model) == 3


def test_a_stateless_activation_in_a_container_says_what_to_write_instead():
    with pytest.raises(TypeError, match=r"pynn\.nn\.Activation"):
        Sequential([ReLU()])

    with pytest.raises(TypeError, match=r"pynn\.nn\.Identity"):
        Sequential([activations.Identity()])


# --------------------------------------------------------------------------- #
# Hyperparameters
#
# `hyperparameters` is what `summary()` renders and what a future `state_dict` would
# need to rebuild a layer, so it has to report what the layer was actually
# constructed with rather than a hard-coded default.
# --------------------------------------------------------------------------- #


def test_linear_reports_its_hyperparameters():
    layer = Linear(4, 3, activation="relu", include_bias=False)

    assert layer.hyperparameters == {
        "in_features": 4,
        "out_features": 3,
        "activation": "relu",
        "weight_initializer": "xavier_normal",
        "bias_initializer": "zeros",
        "include_bias": False,
    }


def test_linear_hyperparameters_pick_up_an_inferred_in_features():
    layer = Linear(3)
    assert layer.hyperparameters["in_features"] is None

    layer(Tensor(np.zeros((2, 6))))
    assert layer.hyperparameters["in_features"] == 6


def test_conv2d_reports_its_hyperparameters():
    layer = Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1)
    hyperparameters = layer.hyperparameters

    assert hyperparameters["in_channels"] == 2
    assert hyperparameters["out_channels"] == 3
    assert hyperparameters["kernel_size"] == (3, 3)
    assert hyperparameters["stride"] == (2, 2)
    # The original spec, not the resolved (int, int), so 'same' survives a round trip.
    assert hyperparameters["padding"] == 1
    assert hyperparameters["input_shape"] is None

    layer(Tensor(np.zeros((1, 2, 8, 8))))
    assert layer.hyperparameters["input_shape"] == (2, 8, 8)


def test_flatten_and_activation_report_their_hyperparameters():
    assert Flatten().hyperparameters == {}
    assert Activation("tanh").hyperparameters == {"activation": "tanh"}


def test_summary_names_the_layer_and_its_hyperparameters():
    layer = Linear(4, 3, name="encoder")
    summary = layer.summary()

    assert summary["name"] == "encoder"
    assert summary["hyperparameters"]["out_features"] == 3

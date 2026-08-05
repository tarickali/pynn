"""Dropout, the normalization layers, and pooling.

Gradients are covered by the verify sweep, which checks each of these against central
differences both alone and with its input reused. What is here instead is the behavior
a gradient check cannot see: that dropout is off at evaluation, that batch
normalization stops using the current batch's statistics once in eval mode, that
running statistics survive a checkpoint, and that pooling routes its gradient to the
positions it actually read.
"""

import numpy as np
import pytest

import pynn.core.math as pmath
from pynn.core import Tensor
from pynn.core.random import set_seed
from pynn.functional.modules import avg_pool2d, dropout, max_pool2d
from pynn.nn import (
    AvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2d,
    Sequential,
)


@pytest.fixture
def images(rng) -> Tensor:
    return Tensor(rng.standard_normal((4, 3, 8, 8)))


# --------------------------------------------------------------------------- #
# Dropout
# --------------------------------------------------------------------------- #


def test_dropout_zeros_roughly_the_requested_fraction(rng):
    x = Tensor(np.ones((200, 200)))

    output = dropout(x, p=0.3, training=True, rng=0)

    assert (output.data == 0.0).mean() == pytest.approx(0.3, abs=0.02)


def test_dropout_rescales_the_survivors(rng):
    """Inverted dropout: the expectation is preserved, so eval needs no correction."""
    x = Tensor(np.ones((400, 400)))

    output = dropout(x, p=0.4, training=True, rng=0)

    assert sorted(np.unique(output.data)) == pytest.approx([0.0, 1 / 0.6])
    assert output.data.mean() == pytest.approx(1.0, abs=0.01)


def test_dropout_is_the_identity_at_evaluation(rng):
    x = Tensor(rng.standard_normal((10, 10)))
    assert np.array_equal(dropout(x, p=0.5, training=False).data, x.data)


def test_dropout_with_p_zero_is_the_identity(rng):
    x = Tensor(rng.standard_normal((10, 10)))
    assert np.array_equal(dropout(x, p=0.0).data, x.data)


def test_dropout_with_p_one_zeros_everything(rng):
    x = Tensor(rng.standard_normal((10, 10)))

    output = dropout(x, p=1.0, training=True)
    pmath.sum(output).backward()

    assert np.all(output.data == 0.0)
    assert np.all(x.grad == 0.0)


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_an_out_of_range_probability_is_rejected(p):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        dropout(Tensor(np.zeros((2, 2))), p=p)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Dropout(p)


def test_dropout_routes_the_gradient_through_the_survivors(rng):
    x = Tensor(rng.standard_normal((6, 6)))

    output = dropout(x, p=0.5, training=True, rng=1)
    pmath.sum(output).backward()

    dropped = output.data == 0.0
    assert np.all(x.grad[dropped] == 0.0)
    assert np.allclose(x.grad[~dropped], 2.0)


def test_the_dropout_layer_follows_the_module_mode(rng):
    layer = Dropout(0.5)
    x = Tensor(np.ones((100, 100)))

    assert (layer(x).data == 0.0).any()

    layer.eval()
    assert np.array_equal(layer(x).data, x.data)

    layer.train()
    assert (layer(x).data == 0.0).any()


def test_dropout_masks_differ_between_calls():
    layer = Dropout(0.5)
    x = Tensor(np.ones((50, 50)))

    assert not np.array_equal(layer(x).data, layer(x).data)


def test_set_seed_makes_dropout_reproducible():
    x = Tensor(np.ones((50, 50)))

    set_seed(7)
    first = Dropout(0.5)(x).data
    set_seed(7)
    second = Dropout(0.5)(x).data

    assert np.array_equal(first, second)


def test_dropout_has_no_parameters():
    assert Dropout(0.5).num_parameters() == 0


# --------------------------------------------------------------------------- #
# LayerNorm
# --------------------------------------------------------------------------- #


def test_layer_norm_standardizes_each_example(rng):
    x = Tensor(rng.standard_normal((5, 8)) * 3.0 + 10.0)

    output = LayerNorm()(x)

    assert np.allclose(output.data.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(output.data.std(axis=-1), 1.0, atol=1e-3)


def test_layer_norm_is_independent_of_the_rest_of_the_batch(rng):
    """The property that distinguishes it from batch normalization."""
    layer = LayerNorm(6)
    example = rng.standard_normal((1, 6))

    alone = layer(Tensor(example)).data
    batched = layer(Tensor(np.vstack([example, rng.standard_normal((7, 6)) * 50.0])))

    assert np.allclose(alone, batched.data[:1])


def test_layer_norm_infers_the_normalized_shape():
    layer = LayerNorm()
    layer(Tensor(np.zeros((3, 7))))

    assert layer.normalized_shape == (7,)
    assert layer.parameters["gamma"].shape == (7,)


def test_layer_norm_normalizes_over_several_trailing_axes(rng):
    x = Tensor(rng.standard_normal((4, 3, 5)))

    output = LayerNorm((3, 5))(x)

    assert np.allclose(output.data.mean(axis=(1, 2)), 0.0, atol=1e-6)


def test_layer_norm_applies_its_scale_and_shift(rng):
    layer = LayerNorm(4)
    x = Tensor(rng.standard_normal((3, 4)))
    layer(x)
    layer.parameters["gamma"].data = np.full(4, 2.0)
    layer.parameters["beta"].data = np.full(4, 5.0)

    assert np.allclose(layer(x).data.mean(axis=-1), 5.0)
    assert np.allclose(layer(x).data.std(axis=-1), 2.0, atol=1e-3)


def test_layer_norm_without_affine_has_no_parameters():
    layer = LayerNorm(4, elementwise_affine=False)
    layer(Tensor(np.zeros((2, 4))))

    assert layer.num_parameters() == 0


def test_layer_norm_behaves_identically_in_both_modes(rng):
    layer = LayerNorm(5)
    x = Tensor(rng.standard_normal((3, 5)))

    training = layer(x).data
    layer.eval()

    assert np.array_equal(layer(x).data, training)


def test_layer_norm_rejects_a_mismatched_shape():
    layer = LayerNorm(5)
    with pytest.raises(ValueError, match="does not end with"):
        layer(Tensor(np.zeros((3, 4))))


# --------------------------------------------------------------------------- #
# BatchNorm
# --------------------------------------------------------------------------- #


def test_batch_norm_standardizes_each_feature(rng):
    x = Tensor(rng.standard_normal((64, 4)) * np.array([1.0, 5.0, 0.2, 3.0]) + 7.0)

    output = BatchNorm1d()(x)

    assert np.allclose(output.data.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(output.data.std(axis=0), 1.0, atol=1e-3)


def test_batch_norm_2d_standardizes_over_batch_and_space(rng):
    x = Tensor(rng.standard_normal((8, 3, 5, 5)) * 4.0 + 2.0)

    output = BatchNorm2d()(x)

    assert np.allclose(output.data.mean(axis=(0, 2, 3)), 0.0, atol=1e-6)
    assert np.allclose(output.data.std(axis=(0, 2, 3)), 1.0, atol=1e-3)


def test_batch_norm_tracks_running_statistics(rng):
    layer = BatchNorm1d(3)
    x = Tensor(rng.standard_normal((32, 3)) + 10.0)

    layer(x)
    after_one = layer.named_buffers()["running_mean"].copy()

    # PyTorch's convention: new = (1 - momentum) * old + momentum * batch.
    assert np.allclose(after_one, 0.1 * x.data.mean(axis=0))

    for _ in range(200):
        layer(x)
    assert np.allclose(layer.named_buffers()["running_mean"], x.data.mean(axis=0))


def test_batch_norm_uses_the_running_statistics_at_evaluation(rng):
    layer = BatchNorm1d(2)
    training_batch = Tensor(rng.standard_normal((64, 2)) * 3.0 + 5.0)
    for _ in range(300):
        layer(training_batch)

    layer.eval()
    single = Tensor(training_batch.data[:1])
    output = layer(single).data

    running_mean = layer.named_buffers()["running_mean"]
    running_var = layer.named_buffers()["running_var"]
    expected = (single.data - running_mean) / np.sqrt(running_var + 1e-5)
    assert np.allclose(output, expected)


def test_batch_norm_at_evaluation_does_not_move_the_running_statistics(rng):
    layer = BatchNorm1d(2)
    layer(Tensor(rng.standard_normal((16, 2))))
    layer.eval()
    before = layer.named_buffers()["running_mean"].copy()

    layer(Tensor(rng.standard_normal((16, 2)) + 100.0))

    assert np.array_equal(layer.named_buffers()["running_mean"], before)


def test_batch_norm_output_depends_on_the_batch_in_training(rng):
    """The property that makes eval mode necessary in the first place."""
    layer = BatchNorm1d(3)
    example = rng.standard_normal((1, 3))

    with_small = layer(Tensor(np.vstack([example, rng.standard_normal((7, 3))]))).data
    with_large = layer(
        Tensor(np.vstack([example, rng.standard_normal((7, 3)) * 50.0]))
    ).data

    assert not np.allclose(with_small[:1], with_large[:1])


def test_batch_norm_running_statistics_survive_a_checkpoint(tmp_path, rng):
    """Weights alone are not the whole model when a layer carries buffers."""
    source = Sequential([Linear(4, 3), BatchNorm1d()])
    X = Tensor(rng.standard_normal((16, 4)) * 5.0)
    for _ in range(20):
        source(X)
    source.eval()

    path = tmp_path / "bn.npz"
    source.save(path)

    target = Sequential([Linear(4, 3), BatchNorm1d()])
    target(X)
    target.eval()
    target.load(path)

    assert np.allclose(source(X).data, target(X).data)
    assert np.any(target.named_buffers()["1.running_mean"] != 0.0)


def test_batch_norm_buffers_are_not_optimizer_parameters():
    layer = BatchNorm1d(3)
    layer(Tensor(np.zeros((4, 3))))

    assert sorted(layer.named_parameters()) == ["beta", "gamma"]
    assert sorted(layer.named_buffers()) == ["running_mean", "running_var"]
    assert layer.num_parameters() == 6


def test_batch_norm_without_affine_has_no_parameters():
    layer = BatchNorm1d(3, affine=False)
    layer(Tensor(np.zeros((4, 3))))

    assert layer.num_parameters() == 0
    assert sorted(layer.named_buffers()) == ["running_mean", "running_var"]


def test_batch_norm_without_running_stats_uses_batch_statistics_in_eval(rng):
    layer = BatchNorm1d(3, track_running_stats=False)
    x = Tensor(rng.standard_normal((16, 3)) + 4.0)
    layer.eval()

    assert layer.named_buffers() == {}
    assert np.allclose(layer(x).data.mean(axis=0), 0.0, atol=1e-6)


def test_batch_norm_rejects_a_mismatched_feature_count():
    layer = BatchNorm1d(5)
    with pytest.raises(ValueError, match="expected 5 features"):
        layer(Tensor(np.zeros((4, 3))))


@pytest.mark.parametrize(
    "layer,shape",
    [(BatchNorm1d(), (4, 3, 2, 2)), (BatchNorm2d(), (4, 3))],
    ids=["1d-given-4d", "2d-given-2d"],
)
def test_batch_norm_rejects_the_wrong_rank(layer, shape):
    with pytest.raises(ValueError, match="expects an input of rank"):
        layer(Tensor(np.zeros(shape)))


def test_batch_norm_1d_accepts_a_length_axis(rng):
    output = BatchNorm1d()(Tensor(rng.standard_normal((8, 3, 5))))

    assert output.shape == (8, 3, 5)
    assert np.allclose(output.data.mean(axis=(0, 2)), 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #


def test_max_pool_takes_the_window_maximum():
    x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4))

    output = max_pool2d(x, (2, 2))

    assert output.shape == (1, 1, 2, 2)
    assert np.array_equal(output.data[0, 0], [[5.0, 7.0], [13.0, 15.0]])


def test_avg_pool_takes_the_window_mean():
    x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4))

    output = avg_pool2d(x, (2, 2))

    assert np.array_equal(output.data[0, 0], [[2.5, 4.5], [10.5, 12.5]])


def test_pooling_defaults_to_non_overlapping_windows(images):
    assert MaxPool2d(2)(images).shape == (4, 3, 4, 4)
    assert AvgPool2d(2)(images).shape == (4, 3, 4, 4)


@pytest.mark.parametrize(
    "kernel,stride,padding,expected",
    [
        (2, None, 0, (4, 3, 4, 4)),
        (2, 1, 0, (4, 3, 7, 7)),
        (3, 2, 1, (4, 3, 4, 4)),
        (8, None, 0, (4, 3, 1, 1)),
    ],
)
def test_pooling_output_shapes(images, kernel, stride, padding, expected):
    assert MaxPool2d(kernel, stride, padding)(images).shape == expected
    assert AvgPool2d(kernel, stride, padding)(images).shape == expected


def test_max_pool_routes_the_gradient_to_the_argmax():
    x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4))

    pmath.sum(max_pool2d(x, (2, 2))).backward()

    expected = np.zeros((4, 4))
    expected[[1, 1, 3, 3], [1, 3, 1, 3]] = 1.0
    assert np.array_equal(x.grad[0, 0], expected)


def test_avg_pool_spreads_the_gradient_evenly():
    x = Tensor(np.zeros((1, 1, 4, 4)))

    pmath.sum(avg_pool2d(x, (2, 2))).backward()

    assert np.allclose(x.grad, 0.25)


def test_overlapping_max_pool_windows_accumulate():
    """A position that wins two windows must receive both gradients."""
    x = Tensor(np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]))

    pmath.sum(max_pool2d(x, (2, 2), (1, 1))).backward()

    assert x.grad[0, 0, 1, 1] == 4.0


def test_max_pool_padding_never_wins_a_window():
    """Padding is filled with -inf; a zero would beat every negative input."""
    x = Tensor(np.full((1, 1, 2, 2), -5.0))

    output = max_pool2d(x, (2, 2), (1, 1), (1, 1))

    assert np.all(output.data == -5.0)


def test_avg_pool_padding_counts_toward_the_denominator():
    """PyTorch's default count_include_pad=True."""
    x = Tensor(np.ones((1, 1, 2, 2)))

    output = max_pool2d(x, (2, 2), (2, 2), (1, 1))
    averaged = avg_pool2d(x, (2, 2), (2, 2), (1, 1))

    assert output.shape == averaged.shape
    assert averaged.data[0, 0, 0, 0] == pytest.approx(0.25)


def test_pooling_has_no_parameters():
    assert MaxPool2d(2).num_parameters() == 0
    assert AvgPool2d(2).num_parameters() == 0


@pytest.mark.parametrize("pool", [max_pool2d, avg_pool2d], ids=lambda f: f.__name__)
def test_pooling_rejects_a_non_image_input(pool):
    with pytest.raises(ValueError, match="batch, channels, height, width"):
        pool(Tensor(np.zeros((4, 4))))


def test_pooling_hyperparameters_are_reported():
    assert MaxPool2d(3, 2, 1).hyperparameters == {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
    }


# --------------------------------------------------------------------------- #
# Composed
# --------------------------------------------------------------------------- #


def test_a_cnn_using_every_new_layer_trains(rng):
    from pynn.nn.losses import CategoricalCrossentropy
    from pynn.optim import SGD

    set_seed(0)
    model = Sequential(
        [
            Conv2d(1, 4, 3, padding="same", activation="relu"),
            BatchNorm2d(),
            MaxPool2d(2),
            Flatten(),
            LayerNorm(),
            Dropout(0.25),
            Linear(5),
        ]
    )
    X = Tensor(rng.standard_normal((8, 1, 8, 8)))
    y = Tensor(np.eye(5)[rng.integers(0, 5, 8)])
    loss_fn = CategoricalCrossentropy(logits=True)
    optimizer = SGD(model, learning_rate=0.05, momentum=0.9)

    first = float(loss_fn(y, model(X)).item())
    for _ in range(50):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.update()

    model.eval()
    assert float(loss_fn(y, model(X)).item()) < first


def test_eval_mode_makes_a_dropout_model_deterministic(rng):
    model = Sequential([Linear(4, 8, activation="relu"), Dropout(0.5), Linear(8, 2)])
    X = Tensor(rng.standard_normal((6, 4)))

    assert not np.array_equal(model(X).data, model(X).data)

    model.eval()
    assert np.array_equal(model(X).data, model(X).data)

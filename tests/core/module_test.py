"""The module tree: nesting, recursive parameter access, modes, and checkpointing.

`Sequential` used to subclass a separate `Model` type whose `parameters` was a
`list[dict]` while `Module.parameters` was a `dict`. Nesting one inside another
type-checked, ran the forward pass, and then failed inside the optimizer with
`AttributeError: 'list' object has no attribute 'items'`. Everything here is about the
tree holding together at arbitrary depth.
"""

from typing import Any

import numpy as np
import pytest

import pynn.core.math as pmath
from pynn.core import Module, Tensor
from pynn.nn import Conv2d, Flatten, Linear, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.optim import SGD


class Block(Module):
    """A custom Module holding children as attributes rather than in a container."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.first = Linear(width, activation="relu")
        self.second = Linear(width)

    def forward(self, X: Tensor) -> Tensor:
        return self.second(self.first(X)) + X

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {}


def mlp() -> Sequential:
    return Sequential([Linear(4, 3, activation="tanh"), Linear(3, 2)])


def nested() -> Sequential:
    return Sequential([Sequential([Linear(4, 3, activation="tanh")]), Linear(3, 2)])


# --------------------------------------------------------------------------- #
# Nesting
# --------------------------------------------------------------------------- #


def test_a_nested_sequential_runs_and_trains(rng):
    """The whole point: this used to raise inside `update()`."""
    model = nested()
    X = Tensor(rng.standard_normal((6, 4)))
    y = Tensor(rng.standard_normal((6, 2)))
    optimizer = SGD(model, learning_rate=0.1)
    loss_fn = MeanSquaredError()

    first = float(loss_fn(y, model(X)).item())
    for _ in range(20):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.step()

    assert float(loss_fn(y, model(X)).item()) < first


def test_nesting_does_not_change_the_parameters_reported():
    flat, deep = mlp(), nested()
    X = Tensor(np.zeros((2, 4)))
    flat(X)
    deep(X)

    assert flat.num_parameters() == deep.num_parameters()
    assert sorted(p.shape for p in flat.named_parameters().values()) == sorted(
        p.shape for p in deep.named_parameters().values()
    )


def test_named_parameters_use_dotted_paths():
    model = nested()
    model(Tensor(np.zeros((2, 4))))

    assert sorted(model.named_parameters()) == ["0.0.W", "0.0.b", "1.W", "1.b"]


def test_a_bare_layer_reports_unqualified_names():
    layer = Linear(4, 3)
    layer(Tensor(np.zeros((2, 4))))

    assert sorted(layer.named_parameters()) == ["W", "b"]


def test_named_modules_walks_depth_first():
    model = nested()

    assert [name for name, _ in model.named_modules()] == ["", "0", "0.0", "1"]


def test_named_children_returns_only_direct_children():
    model = nested()

    assert [name for name, _ in model.named_children()] == ["0", "1"]


def test_child_modules_assigned_to_attributes_are_registered():
    block = Block(4)

    assert [name for name, _ in block.named_children()] == ["first", "second"]

    block(Tensor(np.zeros((2, 4))))
    assert sorted(block.named_parameters()) == [
        "first.W",
        "first.b",
        "second.W",
        "second.b",
    ]


def test_a_custom_module_trains_inside_a_container(rng):
    model = Sequential([Linear(4, 5), Block(5), Linear(5, 1)])
    X = Tensor(rng.standard_normal((8, 4)))
    y = Tensor(rng.standard_normal((8, 1)))
    optimizer = SGD(model, learning_rate=0.05)
    loss_fn = MeanSquaredError()

    first = float(loss_fn(y, model(X)).item())
    for _ in range(30):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.update()

    assert float(loss_fn(y, model(X)).item()) < first
    assert model.num_parameters() > 0


def test_assigning_a_module_before_super_init_is_rejected():
    class Broken(Module):
        def __init__(self) -> None:
            self.layer = Linear(2, 2)  # no super().__init__() first
            super().__init__()

        def forward(self, X: Tensor) -> Tensor:
            return X

        @property
        def hyperparameters(self) -> dict[str, Any]:
            return {}

    with pytest.raises(AttributeError, match="super"):
        Broken()


def test_overwriting_a_child_with_a_non_module_removes_it():
    block = Block(4)
    block.first = None  # type: ignore[assignment]

    assert [name for name, _ in block.named_children()] == ["second"]


def test_register_module_rejects_a_non_module():
    with pytest.raises(TypeError, match="expected a Module"):
        Sequential().register_module("x", "not a module")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Sequential as a container
# --------------------------------------------------------------------------- #


def test_sequential_supports_length_indexing_and_iteration():
    model = mlp()

    assert len(model) == 2
    assert model[0] is model.modules[0]
    assert [m.name for m in model] == ["Linear", "Linear"]


def test_append_extends_the_sequence_and_the_tree():
    model = Sequential([Linear(4, 3)])
    added = model.append(Linear(3, 2))

    assert len(model) == 2
    assert model[1] is added

    model(Tensor(np.zeros((2, 4))))
    assert sorted(model.named_parameters()) == ["0.W", "0.b", "1.W", "1.b"]


def test_an_empty_sequential_is_the_identity():
    x = Tensor(np.arange(6.0).reshape(2, 3))
    assert np.array_equal(Sequential()(x).data, x.data)


def test_sequential_hyperparameters_list_its_layers():
    model = Sequential([Linear(4, 3), Flatten()])
    assert model.hyperparameters == {"modules": ["Linear", "Flatten"]}


# --------------------------------------------------------------------------- #
# Parameter groups and the optimizer
# --------------------------------------------------------------------------- #


def test_parameter_groups_are_live_dictionaries():
    """Lazily built layers depend on this: the optimizer holds the same dicts."""
    model = mlp()
    groups = model.parameter_groups()

    assert groups == [{}, {}, {}]  # container plus two unbuilt layers

    model(Tensor(np.zeros((2, 4))))
    assert [sorted(group) for group in groups] == [[], ["W", "b"], ["W", "b"]]


def test_one_group_per_module_at_any_depth():
    assert len(nested().parameter_groups()) == 4  # outer, inner, its Linear, Linear


def test_optimizer_rejects_a_bare_parameter_dict():
    """`SGD(model.parameters, ...)` used to be the documented call and is now a dict."""
    with pytest.raises(TypeError, match="got a single dict"):
        SGD(Linear(4, 3).parameters, learning_rate=0.1)  # type: ignore[arg-type]


def test_optimizer_rejects_a_sequence_of_non_dicts():
    with pytest.raises(TypeError, match="must be a"):
        SGD([Tensor(np.zeros(3))], learning_rate=0.1)  # type: ignore[list-item]


def test_optimizer_zero_grad_clears_the_whole_model(rng):
    model = mlp()
    optimizer = SGD(model, learning_rate=0.1)
    pmath.sum(model(Tensor(rng.standard_normal((3, 4))))).backward()

    assert any(np.any(p.grad != 0.0) for p in model.named_parameters().values())

    optimizer.zero_grad()
    assert all(np.all(p.grad == 0.0) for p in model.named_parameters().values())


def test_num_parameters_counts_the_tree():
    model = nested()
    model(Tensor(np.zeros((2, 4))))

    assert model.num_parameters() == (4 * 3 + 3) + (3 * 2 + 2)


def test_num_parameters_can_exclude_frozen_ones():
    model = mlp()
    model(Tensor(np.zeros((2, 4))))
    model[0].freeze()

    assert model.num_parameters(trainable_only=True) == 3 * 2 + 2
    assert model.num_parameters() == (4 * 3 + 3) + (3 * 2 + 2)


# --------------------------------------------------------------------------- #
# Collective state
# --------------------------------------------------------------------------- #


def test_zero_grad_reaches_every_depth(rng):
    model = nested()
    pmath.sum(model(Tensor(rng.standard_normal((3, 4))))).backward()

    assert any(np.any(p.grad != 0.0) for p in model.named_parameters().values())

    model.zero_grad()
    assert all(np.all(p.grad == 0.0) for p in model.named_parameters().values())


def test_freezing_a_container_freezes_its_descendants():
    model = nested()
    model(Tensor(np.zeros((2, 4))))

    model.freeze()
    assert all(not p.trainable for p in model.named_parameters().values())
    assert all(not m.trainable for _, m in model.named_modules())

    model.unfreeze()
    assert all(p.trainable for p in model.named_parameters().values())


def test_freezing_one_branch_leaves_the_others_trainable():
    model = nested()
    model(Tensor(np.zeros((2, 4))))
    model[0].freeze()

    assert not model.named_parameters()["0.0.W"].trainable
    assert model.named_parameters()["1.W"].trainable


def test_a_frozen_branch_is_not_updated_by_the_optimizer(rng):
    model = nested()
    model(Tensor(np.zeros((2, 4))))
    model[0].freeze()
    optimizer = SGD(model, learning_rate=0.5)

    before = model.state_dict()
    X = Tensor(rng.standard_normal((6, 4)))
    y = Tensor(rng.standard_normal((6, 2)))
    for _ in range(3):
        loss = MeanSquaredError()(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.update()

    after = model.state_dict()
    assert np.array_equal(before["0.0.W"], after["0.0.W"])
    assert not np.array_equal(before["1.W"], after["1.W"])


# --------------------------------------------------------------------------- #
# Training and evaluation modes
# --------------------------------------------------------------------------- #


def test_modules_start_in_training_mode():
    assert all(m.training for _, m in nested().named_modules())


def test_eval_and_train_propagate_to_every_descendant():
    model = Sequential([Block(4), Sequential([Linear(4, 2)])])

    model.eval()
    assert all(not m.training for _, m in model.named_modules())

    model.train()
    assert all(m.training for _, m in model.named_modules())


def test_train_takes_an_explicit_mode():
    model = nested()
    model.train(False)

    assert all(not m.training for _, m in model.named_modules())


def test_mode_switches_return_self_for_chaining():
    model = mlp()

    assert model.eval() is model
    assert model.train() is model


def test_a_branch_can_be_switched_independently():
    model = nested()
    model[0].eval()

    assert not model[0].training
    assert model[1].training
    assert model.training


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_summary_reports_shapes_rather_than_weights():
    """Printing a summary should not dump every weight matrix."""
    model = mlp()
    model(Tensor(np.zeros((2, 4))))
    summary = model.summary()

    assert summary["num_parameters"] == 23
    assert summary["modules"][0]["parameters"] == {"W": (4, 3), "b": (3,)}
    assert all(
        isinstance(shape, tuple)
        for shape in summary["modules"][0]["parameters"].values()
    )


def test_summary_nests_with_the_tree():
    summary = nested().summary()

    assert summary["modules"][0]["modules"][0]["name"] == "Linear"


def test_repr_names_the_module_and_its_size():
    model = mlp()
    model(Tensor(np.zeros((2, 4))))

    assert repr(model) == "Sequential(23 parameters)"


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def test_state_dict_round_trips_through_a_second_model(rng):
    X = Tensor(rng.standard_normal((5, 4)))
    source = nested()
    source(X)

    target = nested()
    target(X)
    assert not np.allclose(source(X).data, target(X).data)

    target.load_state_dict(source.state_dict())
    assert np.allclose(source(X).data, target(X).data)


def test_state_dict_is_a_snapshot_not_a_view(rng):
    model = mlp()
    model(Tensor(np.zeros((2, 4))))

    saved = model.state_dict()
    model.named_parameters()["0.W"].data += 1.0

    assert not np.array_equal(saved["0.W"], model.state_dict()["0.W"])


def test_save_and_load_round_trip(tmp_path, rng):
    X = Tensor(rng.standard_normal((5, 4)))
    source = nested()
    source(X)

    path = tmp_path / "checkpoint.npz"
    source.save(path)

    target = nested()
    target(X)
    target.load(path)

    assert np.allclose(source(X).data, target(X).data)


def test_save_writes_exactly_the_path_given(tmp_path):
    model = Linear(4, 3)
    model(Tensor(np.zeros((2, 4))))
    path = tmp_path / "weights.bin"

    model.save(path)

    assert path.exists()
    model.load(path)


def test_load_state_dict_rejects_a_mismatched_key_set():
    model = mlp()
    model(Tensor(np.zeros((2, 4))))
    state = model.state_dict()
    state["extra"] = np.zeros(3)

    with pytest.raises(KeyError, match="unexpected"):
        model.load_state_dict(state)


def test_load_state_dict_reports_missing_keys():
    model = mlp()
    model(Tensor(np.zeros((2, 4))))
    state = model.state_dict()
    del state["0.W"]

    with pytest.raises(KeyError, match="missing"):
        model.load_state_dict(state)


def test_non_strict_loading_ignores_a_partial_state():
    model = mlp()
    model(Tensor(np.zeros((2, 4))))
    untouched = model.state_dict()["1.W"].copy()

    model.load_state_dict({"0.W": np.ones((4, 3)), "nope": np.zeros(2)}, strict=False)

    assert np.array_equal(model.state_dict()["0.W"], np.ones((4, 3)))
    assert np.array_equal(model.state_dict()["1.W"], untouched)


@pytest.mark.parametrize("strict", [True, False])
def test_a_shape_mismatch_is_rejected_in_both_modes(strict):
    """Loading a differently shaped array would silently give a different model."""
    model = mlp()
    model(Tensor(np.zeros((2, 4))))
    state = model.state_dict()
    state["0.W"] = np.zeros((4, 9))

    with pytest.raises(ValueError, match="has shape"):
        model.load_state_dict(state, strict=strict)


def test_loading_into_an_unbuilt_model_reports_every_key_as_unexpected():
    """Parameters are created lazily, so there is nothing to load into yet."""
    source = mlp()
    source(Tensor(np.zeros((2, 4))))

    with pytest.raises(KeyError, match="unexpected"):
        mlp().load_state_dict(source.state_dict())


def test_a_convolutional_model_round_trips(tmp_path, rng):
    X = Tensor(rng.standard_normal((2, 1, 8, 8)))
    source = Sequential([Conv2d(1, 2, 3, activation="relu"), Flatten(), Linear(2)])
    source(X)

    path = tmp_path / "cnn.npz"
    source.save(path)

    target = Sequential([Conv2d(1, 2, 3, activation="relu"), Flatten(), Linear(2)])
    target(X)
    target.load(path)

    assert np.allclose(source(X).data, target(X).data)

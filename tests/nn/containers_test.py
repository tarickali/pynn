"""ModuleList and ModuleDict.

The failure these prevent is silent: a plain list of layers assigned to an attribute is
not registered, so its modules run in the forward pass, receive gradients, and are never
handed to an optimizer. Nothing raises, and the model trains everything except those
layers. Most of what is asserted here is that the tree actually sees the contents —
parameters, checkpoints, modes, and freezing all reach them.
"""

from typing import Any

import numpy as np
import pytest

from pynn.core import Module, Tensor
from pynn.nn import Linear, ModuleDict, ModuleList, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.optim import SGD


class Residual(Module):
    """A model whose wiring is its own, holding its layers in a ModuleList."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.blocks = ModuleList(Linear(width, activation="relu") for _ in range(depth))

    def forward(self, X: Tensor) -> Tensor:
        for block in self.blocks:
            X = block(X) + X
        return X

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {}


class TwoHeaded(Module):
    """A model that picks a head by name, holding them in a ModuleDict."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = Linear(8, activation="tanh")
        self.heads = ModuleDict({"value": Linear(1), "policy": Linear(3)})

    def forward(self, X: Tensor) -> Tensor:
        return self.heads["value"](self.trunk(X))

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {}


def layers(count: int = 3) -> list[Linear]:
    return [Linear(4) for _ in range(count)]


# --------------------------------------------------------------------------- #
# The failure they exist to prevent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,wrapper",
    [
        ([Linear(4), Linear(4)], "ModuleList"),
        ((Linear(4),), "ModuleList"),
        ({"a": Linear(4)}, "ModuleDict"),
    ],
    ids=["list", "tuple", "dict"],
)
def test_a_plain_container_of_modules_is_refused(value, wrapper):
    class Broken(Module):
        def forward(self, X: Tensor) -> Tensor:
            return X

        @property
        def hyperparameters(self) -> dict[str, Any]:
            return {}

    broken = Broken()
    with pytest.raises(TypeError, match=wrapper):
        broken.layers = value


def test_containers_without_modules_are_left_alone():
    """The guard must not object to ordinary data."""

    class Fine(Module):
        def __init__(self) -> None:
            super().__init__()
            self.shape = [4, 8]
            self.options = {"activation": "relu"}
            self.empty: list[Any] = []

        def forward(self, X: Tensor) -> Tensor:
            return X

        @property
        def hyperparameters(self) -> dict[str, Any]:
            return {}

    assert Fine().shape == [4, 8]


# --------------------------------------------------------------------------- #
# ModuleList
# --------------------------------------------------------------------------- #


def test_module_list_registers_its_contents():
    holder = ModuleList(layers(3))

    assert len(holder) == 3
    assert [name for name, _ in holder.named_children()] == ["0", "1", "2"]


def test_module_list_parameters_reach_the_tree():
    model = Residual(width=4, depth=3)
    model(Tensor(np.zeros((2, 4))))

    assert sorted(model.named_parameters()) == [
        "blocks.0.W",
        "blocks.0.b",
        "blocks.1.W",
        "blocks.1.b",
        "blocks.2.W",
        "blocks.2.b",
    ]
    assert model.num_parameters() == 3 * (4 * 4 + 4)


def test_a_model_holding_a_module_list_trains_every_block(rng):
    model = Residual(width=4, depth=3)
    X = Tensor(rng.standard_normal((8, 4)))
    y = Tensor(rng.standard_normal((8, 4)))
    model(X)

    before = model.state_dict()
    optimizer = SGD(model, learning_rate=0.05)
    loss_fn = MeanSquaredError()
    for _ in range(10):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.step()
    after = model.state_dict()

    for name in before:
        assert not np.array_equal(before[name], after[name]), f"{name} never moved"


def test_module_list_supports_append_extend_and_iadd():
    holder = ModuleList()
    holder.append(Linear(4))
    holder.extend([Linear(4), Linear(4)])
    holder += [Linear(4)]

    assert len(holder) == 4
    assert [name for name, _ in holder.named_children()] == ["0", "1", "2", "3"]


def test_module_list_indexing_and_iteration():
    contents = layers(3)
    holder = ModuleList(contents)

    assert holder[0] is contents[0]
    assert holder[-1] is contents[-1]
    assert list(holder) == contents
    assert contents[1] in holder
    assert Linear(4) not in holder


def test_a_slice_of_a_module_list_is_a_module_list():
    """So the result is still something a Module can hold."""
    holder = ModuleList(layers(4))
    part = holder[1:3]

    assert isinstance(part, ModuleList)
    assert len(part) == 2
    assert part[0] is holder[1]


def test_module_list_insert_and_delete_renumber():
    contents = layers(3)
    holder = ModuleList(contents)
    added = Linear(9)

    holder.insert(1, added)
    assert [name for name, _ in holder.named_children()] == ["0", "1", "2", "3"]
    assert holder[1] is added

    del holder[1]
    assert [name for name, _ in holder.named_children()] == ["0", "1", "2"]
    assert list(holder) == contents


def test_module_list_setitem_replaces_in_place():
    holder = ModuleList(layers(3))
    replacement = Linear(9)
    holder[1] = replacement

    assert holder[1] is replacement
    assert len(holder) == 3


@pytest.mark.parametrize("action", ["append", "insert", "setitem"])
def test_module_list_rejects_a_non_module(action):
    holder = ModuleList(layers(2))
    with pytest.raises(TypeError, match="expected a Module"):
        if action == "append":
            holder.append("not a module")  # type: ignore[arg-type]
        elif action == "insert":
            holder.insert(0, "not a module")  # type: ignore[arg-type]
        else:
            holder[0] = "not a module"  # type: ignore[assignment]


def test_module_list_has_no_forward():
    with pytest.raises(TypeError, match="no forward pass"):
        ModuleList(layers(2))(Tensor(np.zeros((2, 4))))


def test_module_list_nests():
    outer = ModuleList([ModuleList(layers(2)), Linear(4)])

    assert [name for name, _ in outer.named_modules()] == ["", "0", "0.0", "0.1", "1"]


# --------------------------------------------------------------------------- #
# ModuleDict
# --------------------------------------------------------------------------- #


def test_module_dict_registers_its_contents():
    model = TwoHeaded()
    model(Tensor(np.zeros((2, 6))))

    assert sorted(model.named_parameters()) == [
        "heads.value.W",
        "heads.value.b",
        "trunk.W",
        "trunk.b",
    ]


def test_module_dict_keys_values_items_and_iteration():
    a, b = Linear(1), Linear(2)
    holder = ModuleDict({"a": a, "b": b})

    assert holder.keys() == ["a", "b"]
    assert holder.values() == [a, b]
    assert holder.items() == [("a", a), ("b", b)]
    assert list(holder) == ["a", "b"], "iteration yields keys, matching dict"
    assert "a" in holder
    assert "z" not in holder
    assert len(holder) == 2


def test_module_dict_setitem_update_pop_and_delete():
    holder = ModuleDict()
    holder["a"] = Linear(1)
    holder.update({"b": Linear(2), "c": Linear(3)})
    assert holder.keys() == ["a", "b", "c"]

    popped = holder.pop("b")
    assert isinstance(popped, Linear)
    assert holder.keys() == ["a", "c"]

    del holder["a"]
    assert holder.keys() == ["c"]

    holder.clear()
    assert len(holder) == 0


def test_module_dict_rejects_a_dotted_key():
    """Keys become path segments in state_dict; a dot there is ambiguous."""
    with pytest.raises(ValueError, match="may not contain"):
        ModuleDict({"a.b": Linear(1)})


def test_module_dict_rejects_a_non_module():
    with pytest.raises(TypeError, match="expected a Module"):
        ModuleDict({"a": "not a module"})  # type: ignore[dict-item]


def test_module_dict_has_no_forward():
    with pytest.raises(TypeError, match="no forward pass"):
        ModuleDict({"a": Linear(4)})(Tensor(np.zeros((2, 4))))


def test_module_dict_selects_a_head_by_name(rng):
    model = TwoHeaded()
    X = Tensor(rng.standard_normal((5, 6)))

    assert model(X).shape == (5, 1)
    assert model.heads["policy"](model.trunk(X)).shape == (5, 3)


# --------------------------------------------------------------------------- #
# Collective state reaches container contents
# --------------------------------------------------------------------------- #


def test_modes_propagate_into_containers():
    model = TwoHeaded()

    model.eval()
    assert not model.heads["value"].training
    assert not model.heads.training

    model.train()
    assert model.heads["policy"].training


def test_freezing_reaches_container_contents():
    model = Residual(width=4, depth=2)
    model(Tensor(np.zeros((2, 4))))

    model.freeze()
    assert all(not p.trainable for p in model.named_parameters().values())

    model.blocks[0].unfreeze()
    assert model.named_parameters()["blocks.0.W"].trainable
    assert not model.named_parameters()["blocks.1.W"].trainable


def test_zero_grad_reaches_container_contents(rng):
    import pynn.core.math as pmath

    model = Residual(width=4, depth=2)
    pmath.sum(model(Tensor(rng.standard_normal((3, 4))))).backward()
    assert any(np.any(p.grad != 0.0) for p in model.named_parameters().values())

    model.zero_grad()
    assert all(np.all(p.grad == 0.0) for p in model.named_parameters().values())


def test_a_container_holding_model_round_trips_through_a_checkpoint(tmp_path, rng):
    X = Tensor(rng.standard_normal((5, 4)))
    source = Residual(width=4, depth=3)
    source(X)

    path = tmp_path / "residual.npz"
    source.save(path)

    target = Residual(width=4, depth=3)
    target(X)
    assert not np.allclose(source(X).data, target(X).data)

    target.load(path)
    assert np.allclose(source(X).data, target(X).data)


def test_containers_compose_with_sequential(rng):
    model = Sequential([Linear(4, 4), Residual(width=4, depth=2), Linear(4, 2)])
    X = Tensor(rng.standard_normal((6, 4)))
    y = Tensor(rng.standard_normal((6, 2)))
    optimizer = SGD(model, learning_rate=0.05)
    loss_fn = MeanSquaredError()

    first = float(loss_fn(y, model(X)).item())
    for _ in range(20):
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.step()

    assert float(loss_fn(y, model(X)).item()) < first
    assert "1.blocks.0.W" in model.named_parameters()

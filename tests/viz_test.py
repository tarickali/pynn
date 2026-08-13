"""The DOT dump of the tape.

The interesting cases are the ones where the graph is not a chain. A tensor with two
consumers has to appear once with an edge to each — a walk that recurses into children
without a visited set draws it twice, and the picture then disagrees with the graph
`backward` walks, which is the only thing that makes the picture worth having. The rest
is the cap: a large graph has to be cut somewhere, and it has to say so.
"""

import re

import numpy as np
import pytest

import pynn.core.math as pmath
import pynn.functional as F
from pynn.core import Tensor, no_grad
from pynn.nn import Linear, Sequential
from pynn.viz import to_dot


def nodes(dot: str) -> list[str]:
    return [line for line in dot.splitlines() if "[label=" in line]


def edges(dot: str) -> list[str]:
    return [line for line in dot.splitlines() if " -> " in line]


def fill_of(dot: str, label: str) -> str:
    """The fill colour of the one node carrying `label`."""
    (line,) = [node for node in nodes(dot) if f'label="{label}"' in node]
    found = re.search(r'fillcolor="([^"]+)"', line)
    assert found is not None, line
    return found.group(1)


# --------------------------------------------------------------------------- #
# Shape of the output
# --------------------------------------------------------------------------- #


def test_the_output_is_a_complete_dot_graph() -> None:
    dot = to_dot(Tensor(np.ones(2)))

    assert dot.startswith("digraph tape {\n")
    assert dot.endswith("}\n")


def test_a_lone_leaf_is_one_node_and_no_edges() -> None:
    dot = to_dot(Tensor(np.ones((2, 3))))

    assert len(nodes(dot)) == 1
    assert edges(dot) == []


def test_a_small_graph_draws_one_node_per_tensor() -> None:
    x, w = Tensor(np.ones((2, 3))), Tensor(np.ones((3,)))

    dot = to_dot(pmath.sum(x * w))

    assert len(nodes(dot)) == 4  # x, w, mul, sum
    assert len(edges(dot)) == 3  # x -> mul, w -> mul, mul -> sum


def test_a_node_is_labelled_with_its_operation_and_shape() -> None:
    dot = to_dot(F.relu(Tensor(np.ones((2, 3)))))

    assert r'label="relu\n(2, 3)"' in dot
    # A leaf has no operation to name, and the tape does not know what the caller
    # calls it, so the shape is the whole label.
    assert 'label="(2, 3)"' in dot


def test_the_same_graph_produces_the_same_text_twice() -> None:
    """Node ids are positions in the walk rather than `id()` values, so a committed
    figure changes only when the graph does."""
    first = Tensor(np.ones((2, 2)))
    second = Tensor(np.ones((2, 2)))

    assert to_dot(pmath.sum(F.relu(first))) == to_dot(pmath.sum(F.relu(second)))


# --------------------------------------------------------------------------- #
# Graph shapes that are not a chain
# --------------------------------------------------------------------------- #


def test_a_tensor_used_twice_appears_once_with_an_edge_to_each_consumer() -> None:
    """A residual connection: the same tensor feeds the activation and the sum."""
    x = Tensor(np.ones((2, 2)))

    dot = to_dot(pmath.sum(F.relu(x) + x))

    assert len(nodes(dot)) == 4  # x, relu, add, sum — x once, not twice
    assert dot.count('label="(2, 2)"') == 1
    assert len(edges(dot)) == 4  # x -> relu, x -> add, relu -> add, add -> sum


def test_a_tensor_consumed_twice_by_one_operation_gets_both_edges() -> None:
    """`x * x` records x as both children, and the picture should say so."""
    x = Tensor(np.ones(3))

    dot = to_dot(pmath.sum(x * x))

    assert len(nodes(dot)) == 3  # x, mul, sum
    assert len(edges(dot)) == 3
    assert len(set(edges(dot))) == 2  # the two x -> mul edges are the same line


def test_a_graph_built_under_no_grad_is_a_single_node() -> None:
    """Not a limitation of the drawing: `add_children` dropped the edges, so the
    operation's name is all the tape kept."""
    x = Tensor(np.ones((2, 3)))
    with no_grad():
        output = pmath.sum(x * 2.0)

    dot = to_dot(output)

    assert len(nodes(dot)) == 1
    assert edges(dot) == []
    assert r'label="sum\n(1, 1)"' in dot


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def test_parameters_are_named_and_drawn_apart_from_other_leaves() -> None:
    model = Sequential([Linear(3, 2)])
    X = Tensor(np.ones((2, 3)))

    dot = to_dot(pmath.sum(model(X)), model)

    assert r'label="0.W\n(3, 2)"' in dot
    assert r'label="0.b\n(2,)"' in dot
    # Three classes, three fills: a parameter, the input batch, and a computed node.
    # `trainable` is True on all of them, which is why it is not the test used.
    fills = {
        fill_of(dot, r"0.W\n(3, 2)"),
        fill_of(dot, "(2, 3)"),
        fill_of(dot, r"matmul\n(2, 2)"),
    }
    assert len(fills) == 3


def test_parameters_may_be_given_as_a_mapping() -> None:
    W = Tensor(np.ones((3, 2)))

    dot = to_dot(pmath.sum(Tensor(np.ones((2, 3))) @ W), {"W": W})

    assert r'label="W\n(3, 2)"' in dot


def test_a_name_holding_a_quote_is_escaped() -> None:
    """A ModuleDict key becomes a path segment and is only refused for holding a `.`,
    so an unescaped quote here would end the DOT string early."""
    W = Tensor(np.ones(2))

    dot = to_dot(pmath.sum(W), {'a"b': W})

    assert r'label="a\"b' in dot


# --------------------------------------------------------------------------- #
# The cap
# --------------------------------------------------------------------------- #


def chain(length: int) -> Tensor:
    """`exp` at the far end, then `length` additions on top of it."""
    output = pmath.exp(Tensor(np.ones(2)))
    for _ in range(length):
        output = output + 1.0
    return output


def test_a_capped_graph_is_cut_and_says_how_much_is_missing() -> None:
    # exp's input, exp, and then an add and a promoted 1.0 per step: 14 in total.
    dot = to_dot(chain(6), max_nodes=4)

    assert len(nodes(dot)) == 5  # the marker is extra, not part of the cap
    assert "+ 10 more nodes" in dot
    assert "cut by max_nodes=4" in dot
    assert any(edge.startswith("  truncated -> ") for edge in edges(dot))


def test_a_cap_keeps_the_nodes_nearest_the_output() -> None:
    """Breadth-first from the output, so what survives is the end anyone was looking
    at rather than one branch followed to its leaves."""
    dot = to_dot(chain(6), max_nodes=4)

    assert "exp" not in dot


def test_a_cap_larger_than_the_graph_draws_no_marker() -> None:
    dot = to_dot(chain(2), max_nodes=100)

    assert "truncated" not in dot
    assert len(nodes(dot)) == 6


@pytest.mark.parametrize("cap", [0, -1])
def test_a_cap_below_one_is_refused(cap: int) -> None:
    with pytest.raises(ValueError, match="max_nodes must be at least 1"):
        to_dot(Tensor(np.ones(2)), max_nodes=cap)


# --------------------------------------------------------------------------- #
# The Tensor facade
# --------------------------------------------------------------------------- #


def test_the_tensor_method_forwards_to_the_module_function() -> None:
    model = Sequential([Linear(3, 2)])
    loss = pmath.sum(model(Tensor(np.ones((2, 3)))))

    assert loss.to_dot(model) == to_dot(loss, model)
    assert loss.to_dot(model, max_nodes=3) == to_dot(loss, model, max_nodes=3)

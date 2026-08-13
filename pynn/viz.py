"""Graphviz DOT for the tape behind a Tensor.

`pynn.verify` checks the tape; this draws it. Both are read-only tools that ship with
the library rather than parts of it, which is why the emitter lives here instead of in
`pynn/core/tensor.py`: that file is the first one this project asks anyone to read, and
what a `Tensor` *is* does not include how to draw one. `Tensor.to_dot` is a facade over
`to_dot` below, because `loss.to_dot(model)` is what someone types in a REPL.

Emitting text is the whole job. Nothing here shells out to Graphviz or imports the
`graphviz` package, so the library keeps its single NumPy dependency and rendering stays
the caller's:

    with open("tape.dot", "w") as handle:
        handle.write(to_dot(loss, model))

then `dot -Tpng tape.dot -o tape.png`, or `graphviz.Source(to_dot(loss, model))` to
render inline in a notebook.

What the picture shows is what the tape holds, which is the point of having it. A node
is a `Tensor`, labelled with the operation that produced it and its shape; an edge runs
from a tensor to the operation that consumed it, so the graph reads downwards from the
inputs to the tensor that was asked about. Leaves — anything with no children — are
drawn squared off rather than rounded, and a leaf the caller identified as a parameter
gets its name and a third fill.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from pynn.core.module import Module
from pynn.core.tensor import Tensor

__all__ = ["DEFAULT_MAX_NODES", "to_dot"]

#: Nodes drawn before the graph is cut short. Generous for anything feedforward — a
#: two-layer MLP with a loss is 15 nodes — and far below the size at which a picture
#: stops being one: 300 steps of an unrolled `LSTMCell` is 5,408 nodes, which Graphviz
#: will lay out, slowly, into a page nobody can read.
DEFAULT_MAX_NODES = 200

# Three node classes, separated by outline as well as by fill so the distinction
# survives a greyscale print: rounded for something the tape computed, squared off for
# a leaf, and a third fill for a leaf the caller named as a parameter.
_OPERATION = 'style="rounded,filled", fillcolor="#dbe7f6", color="#37628f"'
_LEAF = 'fillcolor="#eceff1", color="#8a939b"'
_PARAMETER = 'fillcolor="#fbeecb", color="#a3801c"'
_TRUNCATED = 'shape=note, style="filled,dashed", fillcolor="#f8dede", color="#a13d3d"'


def to_dot(
    tensor: Tensor,
    parameters: Module | Mapping[str, Tensor] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> str:
    """Graphviz DOT for the graph that produced `tensor`.

    Parameters
    ----------
    tensor : Tensor
        Output to walk back from, usually a loss.
    parameters : Module | Mapping[str, Tensor] | None
        A model, or any mapping of names to Tensors — `named_parameters()` is exactly
        that shape. Its entries are labelled by name and drawn as parameters. Without
        it every leaf is drawn the same way, because the tape has no way to tell a
        weight from the input batch (see Notes).
    max_nodes : int, default `DEFAULT_MAX_NODES`
        Most nodes to draw. The graph is walked breadth-first from `tensor`, so what a
        cap keeps is the part nearest the output. Anything cut is reported by a marker
        node, drawn dashed, with an edge into each node whose children it stands in
        for; the marker itself does not count against the cap.

    Returns
    -------
    str
        DOT source, ending in a newline. Node ids are positions in the walk rather
        than `id()` values, so the same graph produces the same text on every run and
        a committed figure only changes when the graph does.

    Raises
    ------
    ValueError
        If `max_nodes` is less than 1.

    Examples
    --------
    >>> import numpy as np
    >>> import pynn.core.math as pmath
    >>> from pynn.core import Tensor
    >>> x = Tensor(np.ones((2, 3)))
    >>> dot = to_dot(pmath.sum(x * 2.0))
    >>> dot.count(" -> ")   # x and the promoted 2.0 into mul, mul into sum
    3

    Rendering is the caller's, and needs Graphviz installed::

        with open("tape.dot", "w") as handle:
            handle.write(to_dot(loss, model))

    then ``dot -Tpng tape.dot -o tape.png``; ``-Grankdir=LR`` lays a deep graph out
    sideways, which is usually what a long chain wants. In a notebook, skip the file
    and let the ``graphviz`` package render inline::

        from graphviz import Source
        Source(to_dot(loss, model))

    Notes
    -----
    Nodes are identified by `id()`, matching the visited set in `Tensor.backward`: a
    tensor consumed twice appears once, with an edge to each consumer, and a tensor
    used twice by the *same* operation (``x * x``) gets two edges to it. Every
    reachable tensor is held alive by `tensor` for the duration of the walk, so an id
    cannot be recycled underneath one.

    `Tensor.trainable` looks like the way to spot a parameter without being told, and
    is not one: it is True on every Tensor, including the input batch and every
    constant a Python scalar was promoted into. What it distinguishes is a *frozen*
    parameter from a live one, which is a different question — hence `parameters`.

    A tensor produced under `no_grad` draws as a single node. That is not a limitation
    of this function but what the tape holds: `add_children` drops the edges, so the
    output keeps the name of the operation that made it and none of its history.
    """
    if max_nodes < 1:
        raise ValueError(f"max_nodes must be at least 1, got {max_nodes}")

    names = _parameter_names(parameters)
    order, position, reachable = _walk(tensor, max_nodes)

    lines = [
        "digraph tape {",
        "  rankdir=TB;",
        '  node [shape=box, style=filled, fontname="Helvetica", fontsize=10,'
        ' margin="0.14,0.07"];',
        '  edge [color="#5c6870", arrowsize=0.7];',
    ]

    for index, node in enumerate(order):
        name = names.get(id(node))
        if name is not None:
            style = _PARAMETER
        elif not node.children:
            style = _LEAF
        else:
            style = _OPERATION
        lines.append(f'  n{index} [label="{_label(node, name)}", {style}];')

    # Edges run child -> parent, the direction the values flowed, so the picture reads
    # from the inputs down to `tensor`. A drawn node whose children were cut is a
    # frontier node: it is where the marker below attaches, and a dict keeps that list
    # unique and in walk order.
    frontier: dict[int, None] = {}
    for index, node in enumerate(order):
        for child in node.children:
            child_index = position.get(id(child))
            if child_index is None:
                frontier[index] = None
            else:
                lines.append(f"  n{child_index} -> n{index};")

    cut = reachable - len(order)
    if cut:
        label = f"+ {cut} more nodes\\ncut by max_nodes={max_nodes}"
        lines.append(f'  truncated [label="{label}", {_TRUNCATED}];')
        lines.extend(f"  truncated -> n{index};" for index in frontier)

    lines.append("}")
    return "\n".join(lines) + "\n"


def _walk(root: Tensor, max_nodes: int) -> tuple[list[Tensor], dict[int, int], int]:
    """Tensors reachable from `root`, nearest first, and how many there are in total.

    Breadth-first, so that a graph too large to draw keeps the nodes closest to the
    output — the loss and the last few operations are the part anyone was looking at,
    where a depth-first cut would follow one branch to the leaves and drop the rest.

    The walk continues past `max_nodes` without recording anything, purely to count
    what was left out. That is the same traversal `backward` runs on every step, so it
    is cheap next to having built the graph, and it buys a truncation marker that says
    how much is missing rather than that something is.

    Returns
    -------
    tuple[list[Tensor], dict[int, int], int]
        The tensors to draw, a map from `id()` to each one's position in that list,
        and the number of tensors reachable in total.
    """
    order: list[Tensor] = []
    position: dict[int, int] = {}
    queue: deque[Tensor] = deque([root])
    queued = {id(root)}
    reachable = 0

    while queue:
        tensor = queue.popleft()
        reachable += 1
        if len(order) < max_nodes:
            position[id(tensor)] = len(order)
            order.append(tensor)
        for child in tensor.children:
            if id(child) not in queued:
                queued.add(id(child))
                queue.append(child)

    return order, position, reachable


def _parameter_names(
    parameters: Module | Mapping[str, Tensor] | None,
) -> dict[int, str]:
    """Map `id()` to name for the tensors the caller identified as parameters."""
    if parameters is None:
        return {}
    named = (
        parameters.named_parameters() if isinstance(parameters, Module) else parameters
    )
    return {id(tensor): name for name, tensor in named.items()}


def _label(tensor: Tensor, name: str | None) -> str:
    """A node's text: what it is, over its shape.

    A tensor produced under `no_grad` still carries the name of the operation that
    made it — `forward` is a plain attribute, and it is the edges and the reverse
    closure that recording drops — so it is labelled with that operation and drawn as
    a leaf. That is what the tape holds, and the picture should say so.
    """
    heading = name if name is not None else tensor.forward
    shape = str(tensor.shape)
    return f"{_escape(heading)}\\n{shape}" if heading else shape


def _escape(text: str) -> str:
    """Escape a label for a DOT quoted string.

    A `ModuleDict` key becomes a segment of its parameters' paths and is only refused
    for containing `.`, so a quote can reach here and would otherwise end the string
    early and produce a file Graphviz cannot parse.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')

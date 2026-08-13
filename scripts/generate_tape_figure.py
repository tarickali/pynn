"""Regenerate the computation-graph figure the README shows.

    python scripts/generate_tape_figure.py

Builds a two-layer MLP, runs a batch of two through it and takes a squared-error loss,
and writes the tape behind that loss to `docs/tape.dot` and `docs/tape.svg`.

Deliberately tiny. The figure is there so a reader can follow every node in it and see
that the tape is a real data structure rather than a claim in a design document; a
model large enough to be interesting produces a picture nobody reads. Fifteen nodes is
the whole of a forward pass and a loss.

Needs Graphviz on PATH for the SVG (`brew install graphviz`, `apt install graphviz`).
The library itself never shells out — `pynn.viz` emits text and stops there.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import numpy as np

from pynn.core import Tensor
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError

REPO = pathlib.Path(__file__).resolve().parent.parent
DOT = REPO / "docs" / "tape.dot"
SVG = REPO / "docs" / "tape.svg"


def build() -> str:
    """DOT for the graph one forward pass and one loss leave behind."""
    model = Sequential([Linear(3, 4, activation="relu"), Linear(4, 1)])
    X = Tensor(np.zeros((2, 3)))
    y = Tensor(np.zeros((2, 1)))

    loss = MeanSquaredError()(y, model(X))
    return loss.to_dot(model)


def main() -> int:
    dot = build()
    DOT.write_text(dot)
    print(f"wrote {DOT.relative_to(REPO)} — {dot.count('[label=')} nodes")

    if shutil.which("dot") is None:
        print("graphviz is not on PATH; install it and re-run to refresh the SVG")
        return 1

    # -Grankdir=LR here rather than in `to_dot`: which way a graph should be laid out
    # is a property of the page it is going on, not of the tape. A chain this long is
    # unreadably tall in Graphviz's top-down default.
    subprocess.run(
        ["dot", "-Grankdir=LR", "-Tsvg", str(DOT), "-o", str(SVG)], check=True
    )
    print(f"wrote {SVG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

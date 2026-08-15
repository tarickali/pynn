"""Peak resident memory and throughput of a training loop that unrolls a recurrence.

A finished graph is a reference cycle — every Tensor holds its reverse pass as a closure
over itself — so nothing reclaims it until either the caller frees it or CPython's
cyclic collector runs. This measures what that costs, and what each of the two answers
buys, on `examples/char_rnn.ipynb`'s configuration: batch 32, a 64-step unrolled
`LSTMCell`, roughly 150 MB of graph per step.

Run from the project root::

    python -m benchmarks.memory                    # every configuration
    python -m benchmarks.memory --steps 100        # shorter, and see the warning below
    python -m benchmarks.memory --only none free   # a subset

Two things about the method, both learned the hard way:

**A fresh process per configuration**, which is why this spawns children rather than
looping in one process. A heap that has already been grown and fragmented by an earlier
configuration does not shrink back for the next one.

**Peak RSS read from `ps`** by the parent, sampled while the child runs.
`resource.getrusage(...).ru_maxrss` reported near-identical peaks for configurations
whose true peaks differed by more than 2 GB.

And one about the numbers: **measure over at least 400 steps.** The same benchmark over
60 steps reports the opposite conclusion, because the cost of leaving the garbage alone
is the cost of allocating against a heap that is mostly garbage, and that only appears
once enough of it has piled up.
"""

from __future__ import annotations

import argparse
import gc
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pynn.core import Module, Tensor, stack
from pynn.core.random import set_seed
from pynn.core.types import Array
from pynn.nn import Embedding, Linear, LSTMCell
from pynn.nn.losses import SparseCategoricalCrossentropy
from pynn.optim import AdamW, clip_grad_norm

SEED = 0
DEFAULT_STEPS = 400
POLL_SECONDS = 0.05

# examples/char_rnn.ipynb's configuration, so the numbers speak to it directly.
WINDOW, BATCH = 64, 32
EMBEDDING_DIM, HIDDEN_SIZE = 32, 128
LEARNING_RATE, WEIGHT_DECAY, MAX_NORM = 3e-3, 0.01, 1.0
CORPUS = Path("examples/data/shakespeare/input.txt")
VOCAB_SIZE = 65  # Tiny Shakespeare's alphabet, and the fallback's


@dataclass(frozen=True)
class Configuration:
    """One way of dealing with the finished graph, and what to call it."""

    key: str
    label: str
    free: bool
    collect_every: int  # 0 for never


CONFIGURATIONS = [
    Configuration("none", "left to CPython", False, 0),
    Configuration("gc4", "gc.collect() every 4 steps", False, 4),
    Configuration("free", "loss.free_graph()", True, 0),
    Configuration("free+gc4", "free_graph() + gc.collect() every 4", True, 4),
]


class CharLSTM(Module):
    """Embedding -> LSTMCell unrolled over the window -> Linear head.

    A copy of the notebook's model rather than an import of it, because a notebook is
    not importable and the shape of the graph is the whole subject here.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embed = Embedding(vocab_size, embedding_dim)
        self.cell = LSTMCell(embedding_dim, hidden_size)
        self.head = Linear(hidden_size, vocab_size)

    def forward(self, tokens: Array) -> Tensor:
        batch, window = tokens.shape
        embedded = self.embed(tokens)

        state = None
        hidden = []
        for t in range(window):
            state = self.cell(embedded[:, t, :], state)
            hidden.append(state[0])

        sequence = stack(hidden, axis=1).reshape(batch * window, self.hidden_size)
        logits = self.head(sequence)
        return logits.reshape(batch, window, self.vocab_size)

    @property
    def hyperparameters(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
        }


def corpus_windows(steps: int) -> tuple[Array, Array]:
    """`(inputs, targets)` for `steps` batches of non-overlapping windows.

    Tiny Shakespeare if it has been downloaded, otherwise random tokens over a
    vocabulary of the same size. Only the shapes reach the allocator, so the fallback
    measures the same thing; it is not a fallback the notebook itself offers, since
    synthetic text would make its *outputs* meaningless.
    """
    needed = steps * BATCH * (WINDOW + 1)
    if CORPUS.exists():
        text = CORPUS.read_text(encoding="utf-8")
        vocabulary = sorted(set(text))
        stoi = {character: index for index, character in enumerate(vocabulary)}
        data = np.array([stoi[character] for character in text], dtype=np.int64)
        # Tile rather than truncate the run: 400 steps is more windows than the
        # corpus holds, and repeating it costs nothing this benchmark measures.
        data = np.tile(data, needed // len(data) + 1)
    else:
        print(
            f"{CORPUS} not found — using random tokens. "
            "python scripts/download_shakespeare.py for the real corpus.",
            file=sys.stderr,
        )
        data = np.random.default_rng(SEED).integers(0, VOCAB_SIZE, needed + 1)

    starts = np.arange(steps * BATCH) * WINDOW
    offsets = np.arange(WINDOW)
    positions = starts[:, np.newaxis] + offsets
    inputs = data[positions].reshape(steps, BATCH, WINDOW)
    targets = data[positions + 1].reshape(steps, BATCH, WINDOW)
    return inputs, targets


def train(configuration: Configuration, steps: int) -> float:
    """Run the loop under one configuration and return mean milliseconds per step.

    The mean rather than the median, deliberately: a collection pause *is* the cost
    being measured, and a median over a loop that collects every fourth step reports
    the three steps that did not.
    """
    inputs, targets = corpus_windows(steps)

    set_seed(SEED)
    model = CharLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE)
    model(inputs[0][:1])  # force the lazy build before timing anything

    loss_fn = SparseCategoricalCrossentropy(logits=True)
    optimizer = AdamW(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    started = time.perf_counter()
    for step in range(steps):
        logits = model(inputs[step])
        loss = loss_fn(targets[step].reshape(-1), logits.reshape(-1, VOCAB_SIZE))

        model.zero_grad()
        loss.backward()
        if configuration.free:
            loss.free_graph()
        clip_grad_norm(model, MAX_NORM)
        optimizer.step()

        if (
            configuration.collect_every
            and (step + 1) % configuration.collect_every == 0
        ):
            gc.collect()

    return (time.perf_counter() - started) / steps * 1000


def resident_kilobytes(pid: int) -> int:
    """This process's resident set size, from `ps`, or 0 once it has exited."""
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.strip()
    return int(output) if output else 0


@dataclass(frozen=True)
class Measurement:
    """What one configuration cost, and whether it survived."""

    configuration: Configuration
    megabytes: float
    milliseconds: float | None  # None if the run did not reach the end
    returncode: int

    @property
    def finished(self) -> bool:
        return self.milliseconds is not None


def measure(configuration: Configuration, steps: int) -> Measurement:
    """Run one configuration in a fresh process, sampling its RSS from outside.

    Sampling from the parent keeps the measured process exactly the training loop —
    no polling thread of its own competing for the interpreter it is being timed on.

    A run the kernel kills is reported rather than raised. That is the failure this
    benchmark exists to describe: on a machine with less free memory than the
    uncollected graph wants, the loop does not run slowly, it dies.
    """
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarks.memory",
            "--run",
            configuration.key,
            "--steps",
            str(steps),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    peak = 0
    while child.poll() is None:
        peak = max(peak, resident_kilobytes(child.pid))
        time.sleep(POLL_SECONDS)

    stdout, _ = child.communicate()
    milliseconds = float(stdout.strip()) if child.returncode == 0 else None
    return Measurement(configuration, peak / 1024, milliseconds, child.returncode)


def render(measurements: list[Measurement]) -> str:
    lines = [
        "| | ms/step | peak RSS |",
        "| --- | --- | --- |",
    ]
    for measurement in measurements:
        rate = (
            f"{measurement.milliseconds:.1f}"
            if measurement.milliseconds is not None
            else f"killed ({-measurement.returncode})"
        )
        reached = "" if measurement.finished else " reached, then"
        lines.append(
            f"| {measurement.configuration.label} | {rate} | "
            f"{measurement.megabytes:,.0f} MB{reached} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.memory",
        description="Peak RSS and throughput for a loop that unrolls a recurrence.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Training steps per configuration (default {DEFAULT_STEPS}). Fewer than "
        "a few hundred reports the opposite conclusion — see the module docstring.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        choices=[configuration.key for configuration in CONFIGURATIONS],
        help="Configurations to run (default: all).",
    )
    parser.add_argument(
        "--run",
        metavar="KEY",
        choices=[configuration.key for configuration in CONFIGURATIONS],
        help="Internal: run one configuration in this process and print ms/step.",
    )
    args = parser.parse_args(argv)

    by_key = {configuration.key: configuration for configuration in CONFIGURATIONS}

    if args.run is not None:
        print(f"{train(by_key[args.run], args.steps):.4f}")
        return 0

    selected = [by_key[key] for key in args.only] if args.only else CONFIGURATIONS
    print(
        f"Python {platform.python_version()} on {platform.system()} "
        f"{platform.machine()}, NumPy {np.__version__}"
    )
    print(
        f"batch {BATCH} x {WINDOW}-step LSTM, {args.steps} steps, "
        f"a fresh process each\n"
    )

    measurements = []
    for configuration in selected:
        print(f"  {configuration.label} ...", end="", flush=True)
        measurement = measure(configuration, args.steps)
        if measurement.finished:
            print(
                f" {measurement.milliseconds:.1f} ms/step, "
                f"{measurement.megabytes:,.0f} MB"
            )
        else:
            print(
                f" killed by signal {-measurement.returncode} at "
                f"{measurement.megabytes:,.0f} MB"
            )
        measurements.append(measurement)

    print()
    print(render(measurements))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

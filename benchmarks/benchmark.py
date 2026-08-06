"""Time PyNN against PyTorch on CPU for a fixed MLP and a fixed CNN.

Both libraries run the same architecture, the same batch size, and the same
forward / backward / step loop, on synthetic data of the same shape. The point is
not to win — PyTorch dispatches to a hand-tuned C++ kernel per operation, PyNN
dispatches to NumPy from Python — but to know the factor, and to know which
operations account for it.

Run from the project root::

    python -m benchmarks.benchmark
    python -m benchmarks.benchmark --steps 50 --markdown

PyTorch is optional: without it the PyNN timings are still reported, with the
comparison columns left out.
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pynn.core import Tensor
from pynn.core.random import set_seed
from pynn.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Sequential,
)
from pynn.nn.losses import CategoricalCrossentropy
from pynn.optim import SGD

SEED = 0


@dataclass
class Result:
    """Timings for one model on one library."""

    name: str
    library: str
    seconds_per_step: float
    parameters: int

    @property
    def examples_per_second(self) -> float:
        return self.batch_size / self.seconds_per_step

    batch_size: int = 0


def timed(step: Callable[[], None], steps: int, warmup: int) -> float:
    """Median seconds per call, after discarding warmup iterations.

    The median rather than the mean: one page fault or one scheduler preemption in a
    30-step run moves a mean by more than the difference this benchmark is measuring.
    """
    for _ in range(warmup):
        step()

    durations = []
    for _ in range(steps):
        start = time.perf_counter()
        step()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

MLP_BATCH, MLP_IN, MLP_HIDDEN, MLP_OUT = 128, 784, 256, 10
CNN_BATCH, CNN_CH, CNN_SIZE, CNN_OUT = 64, 1, 28, 10


def pynn_mlp() -> Sequential:
    return Sequential(
        [
            Linear(MLP_IN, MLP_HIDDEN, activation="relu"),
            Dropout(0.2),
            Linear(MLP_HIDDEN, MLP_HIDDEN, activation="relu"),
            Linear(MLP_HIDDEN, MLP_OUT),
        ]
    )


def pynn_cnn() -> Sequential:
    return Sequential(
        [
            Conv2d(CNN_CH, 16, 3, padding="same", activation="relu"),
            BatchNorm2d(),
            MaxPool2d(2),
            Conv2d(16, 32, 3, padding="same", activation="relu"),
            MaxPool2d(2),
            Flatten(),
            Linear(CNN_OUT),
        ]
    )


def torch_mlp():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(MLP_IN, MLP_HIDDEN),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
        nn.ReLU(),
        nn.Linear(MLP_HIDDEN, MLP_OUT),
    )


def torch_cnn():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(CNN_CH, 16, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * (CNN_SIZE // 4) ** 2, CNN_OUT),
    )


# --------------------------------------------------------------------------- #
# Harnesses
# --------------------------------------------------------------------------- #


def synthetic(shape: tuple[int, ...], classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal(shape)
    labels = rng.integers(0, classes, shape[0])
    return X, np.eye(classes)[labels]


def benchmark_pynn(
    name: str, build: Callable[[], Sequential], shape: tuple[int, ...], steps: int
) -> Result:
    set_seed(SEED)
    X_array, y_array = synthetic(shape, CNN_OUT)
    X, y = Tensor(X_array), Tensor(y_array)

    model = build()
    model(X)  # lazily build the parameters before timing
    optimizer = SGD(model, learning_rate=0.01, momentum=0.9)
    loss_fn = CategoricalCrossentropy(logits=True)

    def step() -> None:
        loss = loss_fn(y, model(X))
        model.zero_grad()
        loss.backward()
        optimizer.update()

    return Result(
        name, "pynn", timed(step, steps, warmup=3), model.num_parameters(), shape[0]
    )


def benchmark_torch(
    name: str, build: Callable[[], object], shape: tuple[int, ...], steps: int
) -> Result:
    import torch

    torch.manual_seed(SEED)
    # Both libraries run at their own defaults. Pinning PyTorch to one thread while
    # NumPy's BLAS keeps all of them would flatter PyNN, and the number a reader
    # cares about is what they would measure themselves.
    X_array, y_array = synthetic(shape, CNN_OUT)
    X = torch.tensor(X_array, dtype=torch.float64)
    y = torch.tensor(y_array.argmax(axis=1), dtype=torch.long)

    model = build().double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step() -> None:
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    parameters = sum(p.numel() for p in model.parameters())
    return Result(name, "torch", timed(step, steps, warmup=3), parameters, shape[0])


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def environment() -> str:
    lines = [
        f"Python {platform.python_version()} on {platform.system()} "
        f"{platform.machine()}",
        f"NumPy {np.__version__}",
    ]
    try:
        import torch

        threads = torch.get_num_threads()
        lines.append(f"PyTorch {torch.__version__} (CPU, {threads} threads)")
    except ImportError:
        lines.append("PyTorch not installed — comparison columns omitted")
    return "\n".join(lines)


def render(results: list[Result], markdown: bool) -> str:
    by_model: dict[str, dict[str, Result]] = {}
    for result in results:
        by_model.setdefault(result.name, {})[result.library] = result

    rows = []
    for name, libraries in by_model.items():
        pynn_result = libraries["pynn"]
        torch_result = libraries.get("torch")
        row = [
            name,
            f"{pynn_result.batch_size}",
            f"{pynn_result.parameters:,}",
            f"{pynn_result.seconds_per_step * 1000:.1f}",
            f"{torch_result.seconds_per_step * 1000:.1f}" if torch_result else "—",
            (
                f"{pynn_result.seconds_per_step / torch_result.seconds_per_step:.1f}x"
                if torch_result
                else "—"
            ),
            f"{pynn_result.examples_per_second:,.0f}",
        ]
        rows.append(row)

    header = [
        "Model",
        "Batch",
        "Parameters",
        "pynn ms/step",
        "torch ms/step",
        "Ratio",
        "pynn examples/s",
    ]
    if markdown:
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))
    ]

    def justify(row: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        )

    lines = [justify(header)]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(justify(row) for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.benchmark",
        description="Time PyNN against PyTorch on CPU.",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Timed steps per model (default 20)."
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Emit a markdown table for the README."
    )
    args = parser.parse_args(argv)

    models = [
        ("MLP 784-256-256-10", pynn_mlp, torch_mlp, (MLP_BATCH, MLP_IN)),
        (
            "CNN 2 conv + 2 pool",
            pynn_cnn,
            torch_cnn,
            (CNN_BATCH, CNN_CH, CNN_SIZE, CNN_SIZE),
        ),
    ]

    results: list[Result] = []
    for name, pynn_build, torch_build, shape in models:
        results.append(benchmark_pynn(name, pynn_build, shape, args.steps))
        # PyTorch is optional; without it the PyNN timings still stand on their own.
        with contextlib.suppress(ImportError):
            results.append(benchmark_torch(name, torch_build, shape, args.steps))

    print(environment())
    print()
    print(render(results, args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

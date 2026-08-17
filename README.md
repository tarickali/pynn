# PyNN

[![CI](https://github.com/tarickali/pynn/actions/workflows/ci.yml/badge.svg)](https://github.com/tarickali/pynn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tarickali/pynn/branch/main/graph/badge.svg)](https://codecov.io/gh/tarickali/pynn)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/tarickali/pynn/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

**PyNN** is a small, NumPy-based neural network library with automatic differentiation. It provides a PyTorch-like API for building and training feedforward and convolutional models from scratch, with no dependency on PyTorch or TensorFlow.

Every differentiable operation is checked against central-difference numerical gradients
— 226 checks, including branching graph topologies — and the design decisions behind the
tape are written up in [`docs/DESIGN.md`](docs/DESIGN.md).

![The computation graph of a two-layer MLP and a squared-error loss](docs/tape.svg)

*The tape a forward pass leaves behind, drawn by the library itself: every `Tensor` in a
two-layer MLP and its loss, and every edge `backward` walks. Rounded nodes were computed,
square ones are leaves — grey for the input batch and the targets, amber for parameters,
named from the model. One call: `loss.to_dot(model)`.*

---

## Features

- **Automatic differentiation** — Define-by-run style: build a computation graph as you run the forward pass; gradients are computed via reverse-mode differentiation.
- **Layers** — `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Dropout`, `LayerNorm`, `BatchNorm1d`/`BatchNorm2d`, `Embedding`, `RNNCell`, `LSTMCell`, `Flatten`, `Unflatten`, `Identity`, and a generic `Activation` wrapper. `Sequential` is itself a `Module`, so containers nest; `ModuleList` and `ModuleDict` hold layers whose wiring you decide.
- **Module tree** — recursive `named_parameters()`, `state_dict()`/`load_state_dict()`, `save`/`load`, `train()`/`eval()` mode propagation, and `freeze()`/`unfreeze()` that the optimizers honor. Assigning a plain list of layers to an attribute raises rather than silently leaving them untrained.
- **Autodiff controls** — `no_grad()` for inference that builds no graph, `Tensor.detach()`, `requires_grad` tracking, and `Tensor.free_graph()`, which hands a finished tape back at a point you choose instead of leaving a reference cycle to the collector — worth 3.0x in wall clock and 4.4x in peak memory on an unrolled LSTM.
- **Graph visualizer** — `Tensor.to_dot()` walks the tape and emits Graphviz DOT: one node per tensor, labelled with the operation that produced it and its shape, with parameters named from the model. It emits text and stops there, so nothing new is installed to produce it — rendering is `dot -Tsvg`, or `graphviz.Source(...)` in a notebook.
- **Differentiable indexing** — slicing, gathering, and boolean masks stay on the tape, alongside `concat` / `stack` / `split`, `where` / `masked_fill`, and `Tensor.reshape`. Backpropagation through time works because of it: `backward` uses an explicit stack, so a 300-step unrolled cell differentiates without touching the recursion limit.
- **Activations** — Identity, ReLU (and LeakyReLU), Sigmoid, Tanh, Softmax and LogSoftmax (with configurable axis), ELU, SELU, GELU (exact and tanh), SiLU/Swish, SoftPlus, Affine, and PReLU — a *learnable* activation, which is a `Module` so its slope reaches the optimizer.
- **Losses** — Binary and categorical cross-entropy (logits or probabilities, one-hot or integer labels), mean squared error, mean absolute error, Huber/SmoothL1, KL divergence, and hinge (plain and squared). Every loss takes `reduction='mean' | 'sum' | 'none'`.
- **Optimizers** — SGD (momentum, weight decay, Nesterov), Adam, AdamW (decoupled decay), NAdam (Nesterov momentum), RMSprop, Adagrad, Adadelta, with standard hyperparameters. Every update rule runs in place, which took the optimizer from 32% of an MLP training step to about 15%. Plus `StepLR` / `ExponentialLR` / `CosineAnnealingLR` / `OneCycleLR` / `ReduceLROnPlateau` schedules and `clip_grad_norm`.
- **Initializers** — Zeros, ones, constant, random uniform/normal, Xavier (Glorot), He, and LeCun variants (uniform and normal). Fan-in and fan-out are read from the weight layout, so a `Conv2d` kernel is scaled by its receptive field rather than by its output-channel count.
- **Utilities** — `one_hot`, shuffled batch iteration (`get_batches`), `im2col` / `col2im`, and `set_seed` for a reproducible run.
- **Optional acceleration** — the `numba` extra compiles the convolution backward pass for a ~1.27x speedup on a CNN; results are identical with or without it.
- **Typed** — fully annotated and checked by `mypy`, with a PEP 561 `py.typed` marker in the wheel, so a consuming project's type checker reads the annotations rather than treating the package as untyped.
- **Verified gradients** — every differentiable operation is checked against central-difference numerical gradients, including broadcasting and non-linear graph topologies (shared inputs, residual connections, tied weights). The checker is public API: see [Verification](#verification).

---

## Installation

From the project root:

```bash
pip install -e .
```

**Optional extras** are declared in `pyproject.toml`:

```bash
pip install -r requirements-dev.txt        # tests, lint, types, examples, notebook, benchmarks
pip install -r requirements-external.txt   # + torch and tensorflow, for the comparison tests
```

Those files point at the extras declared in `pyproject.toml`, which is where versions are
pinned. Install a single group directly with `pip install -e ".[notebook]"` and so on;
[`USAGE.md`](USAGE.md) lists what each one covers.

Only NumPy is required at runtime — the library and `python -m pynn.verify` need nothing
else. The `torch` and `tensorflow` extras are used solely by the optional comparison
tests, which skip automatically when those packages are absent.

---

## Quick Start

```python
import numpy as np
from pynn.core import Tensor
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.optim import SGD

# Build a small MLP
model = Sequential([
    Linear(10, 16, activation="relu"),
    Linear(16, 16, activation="relu"),
    Linear(16, 1),
])

loss_fn = MeanSquaredError()
optimizer = SGD(model, learning_rate=0.01)

# Training step (one batch)
X = Tensor(np.random.randn(32, 10).astype(np.float64))
y = Tensor(np.random.randn(32, 1).astype(np.float64))

pred = model(X)
loss = loss_fn(y, pred)
model.zero_grad()
loss.backward()
optimizer.update()
```

---

## API Overview

| Area | Contents |
| ------ | ---------- |
| **`pynn.core`** | `Tensor` (autograd), `Module` (the layer/container tree), `Loss`, `Optimizer`, `Activation`, `Initializer`, `no_grad` / `enable_grad` / `set_grad_enabled`, `concat` / `stack` / `split`, `where` / `masked_fill`, types, constants. |
| **`pynn.core.math`** | `abs`, `sum`, `mean`, `exp`, `log` (import as a module — these shadow builtins). |
| **`pynn.core.utils`** | `unbroadcast`, `matrix_multiply_gradients` (backward-pass shape plumbing). |
| **`pynn.core.numeric`** | `stable_sigmoid` (overflow-free kernel shared by the activations and losses). |
| **`pynn.nn`** | Layers: `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Dropout`, `LayerNorm`, `BatchNorm1d`, `BatchNorm2d`, `Embedding`, `RNNCell`, `LSTMCell`, `Flatten`, `Unflatten`, `Identity`, `Activation`. Containers: `Sequential`, `ModuleList`, `ModuleDict`. Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `ELU`, `SELU`, `SoftPlus`, `Affine`. Losses: `BinaryCrossentropy`, `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`, `MeanSquaredError`, `MeanAbsoluteError`, `HuberLoss`, `KLDivLoss`, `HingeLoss` (aliases: `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `MSELoss`, `L1Loss`, `SmoothL1Loss`). |
| **`pynn.nn.factories`** | `activation_factory`, `initializer_factory`. |
| **`pynn.functional`** | Activation functions (`relu`, `sigmoid`, `softmax`, ...). Losses and module functions live in `pynn.functional.losses` and `pynn.functional.modules`. |
| **`pynn.optim`** | `SGD`, `Adam`, `AdamW`, `NAdam`, `RMSprop`, `Adagrad`, `Adadelta`; `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`; `clip_grad_norm`, `clip_grad_value`. |
| **`pynn.utils`** | `one_hot`, `get_batches` (shuffled by default), `make_pair`, `pad_for_conv`, `im2col` / `col2im`, `get_data_and_grad`. |
| **`pynn.verify`** | Self-verification suite: `check_gradients` and `numerical_gradient` for your own operations, plus `check_all_gradients`, `check_stability`, `check_invariants`, and `run_all`. |
| **`pynn.viz`** | `to_dot` — Graphviz DOT for the tape behind a Tensor, also reachable as `Tensor.to_dot()`. |

Activations and initializers can be specified by string in layers (e.g. `activation="relu"`, `weight_initializer="he_normal"`) or constructed via the factories.

---

## Examples

Two notebooks are committed **with their outputs**, so they render inline on GitHub
without being run — one per domain, because "it classifies images" and "it models
sequences" are different claims:

- [**MNIST**](examples/mnist.ipynb) — an MLP to ~98% test accuracy with training curves,
  a confusion matrix, and the learned first-layer filters, then the same data through a
  `Conv2d` / `BatchNorm2d` / `MaxPool2d` network.
- [**A character-level language model**](examples/char_rnn.ipynb) — `Embedding` →
  `LSTMCell` unrolled over a 64-character window → `Linear`, trained on Tiny Shakespeare
  with `AdamW` and `clip_grad_norm`. Text sampled at six checkpoints, so the progression
  from noise to something word-shaped is visible rather than asserted, and what clipping
  actually did over the run.

Scripts, from the project root:

```bash
# Regression
python -m examples.regression

# Binary classification (circles)
python -m examples.binary_classification

# MNIST classification — reaches ~98% test accuracy.
# Uses examples/data/mnist/train.csv if present (see scripts/download_mnist.py),
# otherwise falls back to synthetic data.
python -m examples.mnist

# The corpora the notebooks read. examples/data/ is gitignored.
python scripts/download_mnist.py         # ~227 MB
python scripts/download_shakespeare.py   # ~1.1 MB, stdlib only
```

---

## Testing

All commands run from the project root.

```bash
# Full suite. Needs only NumPy and pytest; the optional torch/tensorflow
# comparison tests skip themselves when those packages are absent.
pytest

# Gradient checks only
pytest tests/test_gradcheck.py

# The shipped self-verification suites
pytest tests/test_verify.py

# Skip the third-party comparison tests entirely
pytest -m "not external"

# Coverage
pytest --cov=pynn --cov-report=term-missing
```

A dependency-free smoke test is also available:

```bash
python scripts/smoke_test.py
```

### Lint and type checking

Install the tools with `pip install -e ".[dev]"`. Both are configured in
`pyproject.toml` — the rule set is pinned explicitly rather than inherited from
whichever ruff version happens to be installed, so results are reproducible.

```bash
ruff check .          # lint
ruff format .         # format
mypy                  # type check (files are configured in pyproject.toml)
```

`ruff check` and `mypy` are both clean across `pynn`, `tests`, `examples`,
`scripts`, and `benchmarks` — including the code cells of both notebooks.
Both are pinned to exact versions, since a newer `ruff format` can reformat code that
is clean today and turn a repository nobody touched red.

The same checks run as git hooks, pinned to the same versions:

```bash
pip install -e ".[hooks]" && pre-commit install
```

### Continuous integration

Every push and pull request to `main` runs the same checks across Python
3.10–3.14 on GitHub Actions: `ruff check`, `ruff format --check`, `mypy`,
`pytest -m "not external" --cov=pynn`, and `python -m pynn.verify`. Coverage is held
to a **95% floor** (`fail_under` in `[tool.coverage.report]`), currently at 98.5%, so
it cannot regress silently. The workflow lives at
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Reproducibility

Weight initialization draws from a module-level `numpy.random.Generator`, not the legacy
global `numpy.random` functions. To make a whole model's initialization reproducible:

```python
from pynn.functional.initializers import set_seed

set_seed(0)
model = Sequential([Linear(784, 256, activation="relu"), Linear(256, 10)])
```

The same seed also controls `Dropout`'s masks, so a run with stochastic layers is
reproducible end to end.

Individual initializers also take an explicit generator, for when you want one layer
drawn from a separate stream:

```python
import numpy as np
from pynn.functional.initializers import he_normal

weights = he_normal((784, 256), rng=np.random.default_rng(0))
```

---

## Verification

An autodiff library can be wrong while looking healthy: training loss still falls when a
gradient is quietly scaled by the batch size, and an activation that overflows to `inf`
only poisons a run once the inputs grow large enough. `pynn.verify` makes those failures
observable, and ships with the package so it can be run against an installed copy:

```bash
python -m pynn.verify              # failures only
python -m pynn.verify --verbose    # every check
python -m pynn.verify stability    # one suite
```

```
gradients: 226/226 passed (OK)
invariants: 161/161 passed (OK)
stability: 28/28 passed (OK)

pynn.verify: 415/415 passed (OK)
```

Three suites:

- **`check_all_gradients`** compares analytic gradients against central differences,
  `(f(x + eps) - f(x - eps)) / 2 eps`, for every operator, math function, activation,
  loss, and module function — across broadcasting shape combinations, and in graph
  topologies where a tensor feeds more than one consumer (shared inputs, rejoining
  branches, diamonds, residual connections, tied weights, auxiliary losses). Every unary
  and binary op is checked twice, once in isolation and once with its input reused, since
  a reverse pass that overwrites `grad` instead of accumulating into it is exactly
  correct in the single-consumer case.
- **`check_stability`** asserts activations and losses stay finite at `|x|` up to 1000,
  well past the `~709` where `exp` overflows in float64, with floating-point warnings
  promoted to errors so a silent overflow fails the check.
- **`check_invariants`** asserts behavioral properties of the tape (accumulation across
  passes, deep graphs without recursion limits, `backward`'s scalar-output contract, a
  freed tape that is reclaimed without the cyclic collector and refuses a second pass) and
  of the optimizers, each compared against a closed-form transcription of its published
  update rule rather than a "loss went down" assertion — a broken momentum buffer still
  descends, just more slowly.

`check_gradients` is also useful on its own for verifying a custom operation:

```python
import numpy as np
from pynn.core import Tensor
import pynn.core.math as pmath
import pynn.functional as F
from pynn.verify import check_gradients

x = Tensor(np.random.randn(4, 5))
result = check_gradients(lambda ts: pmath.sum(F.tanh(ts[0])), [x])

print(result)          # per-input relative errors, and the worst element if it fails
assert result.passed
```

The `tests/` suite covers the same ground for CI, and `tests/test_verify.py` drives these
suites so they cannot rot.

---

## Benchmarks

PyNN against PyTorch on CPU, same architecture and batch size on both, median over 30
steps of forward + backward + optimizer step:

| Model | Batch | Parameters | pynn ms/step | torch ms/step | Ratio | pynn examples/s |
| --- | --- | --- | --- | --- | --- | --- |
| MLP 784-256-256-10 | 128 | 269,322 | 4.3 | 2.3 | 1.8x | 29,636 |
| CNN 2 conv + 2 pool | 64 | 20,522 | 82.7 | 9.7 | 8.6x | 774 |
| LSTM 20 steps + embedding | 32 | 292,328 | 30.2 | 14.5 | 2.1x | 1,059 |

*Python 3.14, NumPy 2.4, PyTorch 2.13, Apple M-series CPU. Both libraries at their own
threading defaults, PyNN without the optional `numba` extra. Run-to-run variation is
roughly ±15%; reproduce with* `python -m benchmarks.benchmark`.

Installing `pip install -e ".[numba]"` compiles the scatter in the convolution backward
pass — about 40% of a CNN step, and the only Python-level loop left in the library. That
takes the CNN row to **59.3 ms/step (6.5x)**. The MLP is unchanged, since it never
touches that code.

Being slower than PyTorch is the expected outcome — the interesting part is the spread
between the rows, which is entirely about how much of each step is a C kernel rather
than Python.

The **MLP** is dominated by `matmul`, where both libraries hand the work to the same
BLAS, so PyNN's overhead is per-operation Python dispatch and it lands within about 2x.
Roughly a third of that overhead used to sit somewhere less defensible: `SGD.update` was
32% of a profiled step, almost all of it allocating a fresh full-size array per line. The
update rules now run in place and that share is ~15%. The table is unchanged, and
deliberately: the end-to-end effect is ~0.1–0.2 ms of a ~3.2 ms step against a ±0.4 ms
run-to-run spread, which is to say it could not be measured here at all. `docs/DESIGN.md`
§15 has what could be — the speedup scales with parameter size, from 1.2x on a
256-element bias to 3.1x on a 200k-element weight matrix — and why two attempts at an
end-to-end harness were thrown away.

The **CNN** is where PyTorch's fused, multithreaded convolution kernels pull away.
PyNN's `conv2d` is im2col plus a single `gemm` — a 10–100x improvement over the Python
loop it replaced — but it still materializes the column matrix and runs the `col2im`
scatter in the reverse pass.

The **LSTM** is the closest of the three, which is not a PyNN win so much as a level
playing field: both libraries run the *cell* in a Python loop of 20 steps, so both pay
per-step interpreter cost, and what remains is the same BLAS underneath. PyTorch's fused
`nn.LSTM` would be far faster than its own `nn.LSTMCell` here; the comparison is
deliberately cell-to-cell, since that is the same computation.

---

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) is a single walkthrough — adding a layer, end to
end — because that is the question this codebase's structure actually answers: the
functional op and its reverse closure, the `Module` around it, the gradcheck entry that
puts it in the shipped sweep, and when a layer needs an invariant a gradient check
cannot see. Released versions are recorded in [`CHANGELOG.md`](CHANGELOG.md).

---

## Design

[`docs/DESIGN.md`](docs/DESIGN.md) covers the parts worth explaining rather than reading:
why define-by-run over a static graph, how the tape is built from closures, the iterative
topological sort, how broadcasting is reversed, why softmax and cross-entropy are fused,
the `Module` tree and why `Sequential` had to become one, how `no_grad` turns recording
off in two places, and what was deliberately left out.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

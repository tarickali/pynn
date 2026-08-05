# PyNN

**PyNN** is a small, NumPy-based neural network library with automatic differentiation. It provides a PyTorch-like API for building and training feedforward and convolutional models from scratch, with no dependency on PyTorch or TensorFlow.

---

## Features

- **Automatic differentiation** — Define-by-run style: build a computation graph as you run the forward pass; gradients are computed via reverse-mode differentiation.
- **Layers** — `Linear`, `Conv2d`, `Flatten`, and a generic `Activation` wrapper; compose them with `Sequential`.
- **Activations** — Identity, ReLU (and LeakyReLU), Sigmoid, Tanh, Softmax (with configurable axis), ELU, SELU, SoftPlus, Affine.
- **Losses** — Binary and categorical cross-entropy (logits or probabilities), mean squared error, mean absolute error; MSE supports `reduction='mean'` or `'sum'`.
- **Optimizers** — SGD (momentum, weight decay, Nesterov), Adam, RMSprop, Adagrad, Adadelta, with standard hyperparameters.
- **Initializers** — Zeros, ones, constant, random uniform/normal, Xavier (Glorot), He, and LeCun variants (uniform and normal).
- **Utilities** — `one_hot`, batched data iteration (`get_batches`), and helpers for training loops.
- **Verified gradients** — every differentiable operation is checked against central-difference numerical gradients, including broadcasting and non-linear graph topologies (shared inputs, residual connections, tied weights). The checker is public API: see [Gradient checking](#gradient-checking).

---

## Installation

From the project root:

```bash
pip install -e .
```

**Optional extras** are declared in `pyproject.toml`:

```bash
pip install -e ".[examples,mnist]"   # scikit-learn, pandas for examples
pip install -e ".[dev]"              # pytest
pip install -e ".[test]"             # pytest + torch + tensorflow
```

Only NumPy is required at runtime. The `torch` and `tensorflow` extras are used solely by
the optional comparison tests, which skip automatically when those packages are absent.

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
optimizer = SGD(model.parameters, learning_rate=0.01)

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
|------|----------|
| **`pynn.core`** | `Tensor` (autograd), `Module`, `Model`, `Loss`, `Optimizer`, `Activation`, `Initializer`, types, constants. |
| **`pynn.core.math`** | `abs`, `sum`, `mean`, `exp`, `log` (import as a module — these shadow builtins). |
| **`pynn.core.utils`** | `unbroadcast`, `matrix_multiply_gradients` (backward-pass shape plumbing). |
| **`pynn.core.numeric`** | `stable_sigmoid` (overflow-free kernel shared by the activations and losses). |
| **`pynn.nn`** | Layers: `Linear`, `Conv2d`, `Flatten`, `Activation`, `Sequential`. Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `ELU`, `SELU`, `SoftPlus`, `Identity`, `Affine`. Losses: `BinaryCrossentropy`, `CategoricalCrossentropy`, `MeanSquaredError`, `MeanAbsoluteError` (aliases: `BCELoss`, `CrossEntropyLoss`, `MSELoss`, `L1Loss`). |
| **`pynn.nn.factories`** | `activation_factory`, `initializer_factory`. |
| **`pynn.functional`** | Activation functions (`relu`, `sigmoid`, `softmax`, ...). Losses and module functions live in `pynn.functional.losses` and `pynn.functional.modules`. |
| **`pynn.optim`** | `SGD`, `Adam`, `RMSprop`, `Adagrad`, `Adadelta`. |
| **`pynn.utils`** | `one_hot`, `get_batches`, `make_pair`, `pad_for_conv`, `get_data_and_grad`. |
| **`pynn.verify`** | Self-verification suite: `check_gradients` and `numerical_gradient` for your own operations, plus `check_all_gradients`, `check_stability`, `check_invariants`, and `run_all`. |

Activations and initializers can be specified by string in layers (e.g. `activation="relu"`, `weight_initializer="he_normal"`) or constructed via the factories.

---

## Examples

From the project root:

```bash
# Regression
python -m examples.regression

# Binary classification (circles)
python -m examples.binary_classification

# MNIST classification — reaches ~98% test accuracy.
# Uses examples/data/mnist/train.csv if present (see scripts/download_mnist.py),
# otherwise falls back to synthetic data.
python -m examples.mnist
```

Or run the Quick Start example:

```bash
python main.py
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

`ruff check` and `mypy` are both clean across `pynn`, `tests`, `examples`, and
`scripts`.

---

## Reproducibility

Weight initialization draws from a module-level `numpy.random.Generator`, not the legacy
global `numpy.random` functions. To make a whole model's initialization reproducible:

```python
from pynn.functional.initializers import set_seed

set_seed(0)
model = Sequential([Linear(784, 256, activation="relu"), Linear(256, 10)])
```

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
gradients: 111/111 passed (OK)
invariants: 56/56 passed (OK)
stability: 20/20 passed (OK)

pynn.verify: 187/187 passed (OK)
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
  passes, deep graphs without recursion limits, `backward`'s scalar-output contract) and
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

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

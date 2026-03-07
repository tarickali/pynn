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

---

## Installation

From the project root (`pynn` directory):

```bash
pip install numpy scipy
```

**Optional:** [Numba](https://numba.pydata.org/) can speed up some tensor operations; the library works without it.

```bash
pip install numba
```

For running the full test suite (which compares against PyTorch/TensorFlow):

```bash
pip install torch tensorflow
```

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
| **`pynn.core`** | `Tensor` (autograd), `Module`, `Model`, `Loss`, `Optimizer`, types, constants, `expand_array` / `shrink_array`, and math (`abs`, `sum`, `mean`, `exp`, `log`). |
| **`pynn.nn`** | Layers: `Linear`, `Conv2d`, `Flatten`, `Activation`, `Sequential`. Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `ELU`, `SELU`, `SoftPlus`, `Identity`, `Affine`. Losses: `BinaryCrossentropy`, `CategoricalCrossentropy`, `MeanSquaredError`, `MeanAbsoluteError` (aliases: `BCELoss`, `CrossEntropyLoss`, `MSELoss`, `L1Loss`). Factories: `activation_factory`, `initializer_factory`. |
| **`pynn.optim`** | `SGD`, `Adam`, `RMSprop`, `Adagrad`, `Adadelta`. |
| **`pynn.utils`** | `one_hot`, `get_batches`, `make_pair`, `pad_for_conv`, `get_data_and_grad`. |

Activations and initializers can be specified by string in layers (e.g. `activation="relu"`, `weight_initializer="he_normal"`) or constructed via the factories.

---

## Examples

From the `pynn` directory:

```bash
# Regression
python -m examples.regression

# Binary classification (circles)
python -m examples.binary_classification

# MNIST-style classification (uses CSV or synthetic data)
python -m examples.mnist
```

Or run the default example via the top-level entry point:

```bash
python main.py
```

---

## Testing

**Smoke test** (no optional dependencies):

```bash
cd pynn
python scripts/smoke_test.py
```

**Core tests** (NumPy; Numba recommended):

```bash
cd pynn
pytest tests/core/ -v
```

**Full test suite** (includes comparisons with PyTorch and TensorFlow; requires `torch` and `tensorflow`):

```bash
cd pynn
pytest tests/ -v
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

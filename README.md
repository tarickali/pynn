# pynn

A library for building and training neural networks.

## Testing

### 1. Install dependencies

From the `pynn` directory:

```bash
pip install numpy scipy
```

**Optional:** `numba` speeds up some tensor ops; pynn works without it. If installs time out, you can skip it or retry with a longer timeout:

```bash
pip install --timeout 120 numba
```

Optional (for the full test suite, which compares against PyTorch/TensorFlow):

```bash
pip install torch tensorflow
```

### 2. Run the smoke test (no PyTorch/TensorFlow)

Quick check that forward pass, backward pass, and optimizer step work:

```bash
cd pynn
python scripts/smoke_test.py
```

### 3. Run the test suite with pytest

From the `pynn` directory (so the project root is on `PYTHONPATH`):

```bash
cd pynn
pytest tests/ -v
```

- **Core tests** (`tests/core/`) need only `numpy` and `numba`. They test `Tensor`, `math`, and `utils`.
- **Functional tests** (`tests/functional/`) compare pynn to PyTorch and TensorFlow; install `torch` and `tensorflow` to run them.

To run only core tests (no torch/tf):

```bash
pytest tests/core/ -v
```

### 4. Run an example

```bash
cd pynn
python -m examples.regression
# or
python -m examples.binary_classification
```

MNIST example (needs the data file or will use random data):

```bash
python -m examples.mnist
```

# requirements

Each file here is a one-line pointer at an extra declared in `pyproject.toml`, which is
where versions are actually pinned. Two mechanisms, one source of truth — editing a
version here would do nothing.

```bash
pip install -r requirements/all.txt        # everything, the usual choice
pip install -r requirements/dev.txt        # just tests, lint, and types
```

| File | Installs | Needed for |
|---|---|---|
| `base.txt` | the library | `import pynn`, `python -m pynn.verify` |
| `dev.txt` | + pytest, pytest-cov, ruff, mypy | `pytest`, `ruff`, `mypy` — what CI installs |
| `examples.txt` | + scikit-learn, pandas | `examples/*.py`, `scripts/download_mnist.py` |
| `notebook.txt` | + matplotlib, jupyterlab, ipykernel, nbclient | `examples/mnist.ipynb` |
| `benchmark.txt` | + torch | `benchmarks/benchmark.py` |
| `external.txt` | + torch, tensorflow | the `external`-marked tests |
| `all.txt` | everything except `external.txt` and Numba | working on any part of the repository |
| `numba.txt` | + numba | reproducing the "Numba is slower" measurement |

**`external.txt` is separate from `all.txt` on purpose.** TensorFlow lags new Python
releases and has no wheel for 3.14, so including it would make the one-command install
fail on the interpreter this project is developed on. The tests that need it skip
themselves when it is absent, so a run without it is still green.

**Numba is separate too**, and is not recommended: it measured 1.9x *slower* than plain
NumPy and takes the test suite from 1.7s to 12.1s. It stays installable only so the
claim can be reproduced.

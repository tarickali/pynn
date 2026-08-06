# USAGE

Every command this project supports, what it does, and what a healthy result looks like.
Run everything from the repository root unless noted.

---

## 0. Setup

### Which Python

| | |
|---|---|
| Declared support | **3.10 – 3.13** (`requires-python = ">=3.10"`, CI matrix) |
| Local `.venv` | **3.14.1** (Homebrew `python@3.14`) |

The local virtualenv runs a version CI does not test. Everything passes on it, but a
3.14-only regression would not be caught by CI — see the note in
[§7 Known gaps](#7-known-gaps).

### Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

`.venv/` is gitignored, so it is yours to delete and rebuild at any time. Nothing in the
repository depends on its contents.

### Install

```bash
pip install -e .                    # the library itself, editable
pip install -e ".[dev]"             # + pytest, pytest-cov, ruff, mypy
```

> **`pynn` is currently *not* installed into `.venv`.** It resolves by accident from the
> working directory (pytest adds `.` via `pythonpath`). That works from the repo root and
> nowhere else. Run `pip install -e .` to fix it — the packaging was verified: the wheel
> ships all six subpackages (`core`, `nn`, `functional`, `optim`, `utils`, `verify`).

### Optional extras

| Extra | Pulls in | Needed for |
|---|---|---|
| `dev` | pytest, pytest-cov, ruff, mypy | tests, lint, types |
| `examples` | scikit-learn | `examples/regression.py`, `examples/binary_classification.py` |
| `mnist` | scikit-learn, pandas | `examples/mnist.py`, `scripts/download_mnist.py` |
| `notebook` | matplotlib, pandas, jupyter | `examples/mnist.ipynb` |
| `benchmark` | torch | `benchmarks/benchmark.py` |
| `test` | pytest, torch, tensorflow | the `external` comparison tests |
| `all` | everything above | |

```bash
pip install -e ".[dev,notebook,benchmark]"
```

---

## 1. Tests

```bash
pytest                              # everything; torch/tf tests skip if absent
pytest -m "not external"            # NumPy-only, what CI runs
pytest --cov=pynn                   # + coverage, enforces the 95% floor
pytest --cov=pynn --cov-report=term-missing   # + which lines are uncovered
```

**Expected:** `541 passed, 1 skipped, 6 deselected` · `Total coverage: 98.66%`

Narrower runs:

```bash
pytest tests/test_gradcheck.py      # the 151 gradient checks, one test per operation
pytest tests/core/module_test.py    # the module tree
pytest tests/nn/layers_test.py      # Dropout, LayerNorm, BatchNorm, pooling
pytest -k "batch_norm"              # anything matching a name
pytest -x -q                        # stop at the first failure
```

The coverage floor lives in `[tool.coverage.report] fail_under = 95` rather than in
pytest's `addopts`, so running one test file does not fail for covering one module. Only
runs that pass `--cov` are held to it.

---

## 2. The shipped verification suite

`pynn.verify` is part of the package, not a test helper — it runs against an installed
copy with no pytest and no test files present.

```bash
python -m pynn.verify                       # failures only
python -m pynn.verify --verbose             # every check, one line each
python -m pynn.verify gradients             # one suite
python -m pynn.verify stability invariants  # several
```

**Expected:**

```
gradients: 151/151 passed (OK)
invariants: 81/81 passed (OK)
stability: 20/20 passed (OK)

pynn.verify: 252/252 passed (OK)
```

Exit code is 0 on success, 1 on any failure, so it works as a CI gate.

Use it on your own layers:

```python
from pynn.core import Tensor
from pynn.verify import check_gradients

result = check_gradients(lambda ts: my_loss(ts[0]), [Tensor(x)])
assert result.passed, result      # str(result) names the worst-disagreeing element
```

---

## 3. Lint and types

```bash
ruff check pynn tests examples scripts benchmarks          # lint
ruff check --fix pynn tests examples scripts benchmarks    # + autofix
ruff format pynn tests examples scripts benchmarks         # format
ruff format --check pynn tests examples scripts benchmarks # check only, no writes
mypy                                                       # files configured in pyproject
```

**Expected:** `All checks passed!` · `77 files already formatted` · `Success: no issues
found in 44 source files`

Ruff covers the **code cells of `examples/mnist.ipynb`** too — it parses `.ipynb`
natively. `ruff format` on a notebook rewrites cell sources and leaves outputs alone.
Mypy is scoped to `pynn/` only (`files = ["pynn"]`).

---

## 4. Examples

```bash
python scripts/smoke_test.py            # dependency-free: forward, backward, one step
python -m examples.regression           # needs [examples]
python -m examples.binary_classification
python -m examples.mnist                # needs [mnist]; real data if present, else synthetic
```

**Expected:** smoke test prints `Smoke test passed: ...`; binary classification reaches
`Accuracy: 1.0000`; MNIST reaches ~98% test accuracy on the real dataset.

### MNIST data

`examples/data/` is gitignored, so the CSV is not in the repository.

```bash
python scripts/download_mnist.py        # writes examples/data/mnist/train.csv (~227 MB)
```

`examples/mnist.py` falls back to synthetic data when the file is absent;
`examples/mnist.ipynb` raises with instructions instead, because a notebook of synthetic
results would be misleading.

---

## 5. The notebook

`examples/mnist.ipynb` is committed **with its outputs**, so it renders on GitHub without
being run. Only re-run it if you change it.

```bash
pip install -e ".[notebook]"
jupyter lab examples/mnist.ipynb
```

To re-execute headlessly and write the outputs back in place:

```bash
pip install nbclient nbformat ipykernel      # already in .venv
python - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("examples/mnist.ipynb", as_version=4)
saved = nb.metadata                          # keep the generic python3 kernelspec
NotebookClient(nb, timeout=1800, kernel_name="pynn-venv",
               resources={"metadata": {"path": "."}}).execute()
nb.metadata = saved
nbformat.write(nb, "examples/mnist.ipynb")
PY
```

Takes about 5 minutes: ~40 s for the MLP, ~3 min for the CNN, the rest is plotting.
See [§7](#7-known-gaps) for why `kernel_name="pynn-venv"` and not `"python3"`.

After editing cells, re-lint and re-format before committing:

```bash
ruff format examples/mnist.ipynb && ruff check examples/mnist.ipynb
```

---

## 6. Benchmarks

```bash
pip install -e ".[benchmark]"
python -m benchmarks.benchmark              # plain table
python -m benchmarks.benchmark --markdown   # markdown, for pasting into the README
python -m benchmarks.benchmark --steps 50   # more samples, less noise
```

Reports the median of `--steps` iterations of forward + backward + optimizer step after 3
warmup iterations. PyTorch is optional — without it the PyNN timings still print, with the
comparison columns blank.

**Expected** (Apple M-series, both libraries at their own threading defaults):

| Model | pynn ms/step | torch ms/step | Ratio |
|---|---|---|---|
| MLP 784-256-256-10 | ~3.7 | ~2.2 | ~1.7x |
| CNN 2 conv + 2 pool | ~80 | ~8 | ~10x |

Run-to-run variation is roughly ±15%. Re-measure before quoting new numbers in the README.

---

## 7. Known gaps

**The local `.venv` runs Python 3.14; CI tests 3.10–3.13.** Everything passes on 3.14, so
adding `"3.14"` to the matrix in `.github/workflows/ci.yml` would close the gap for free.

**`pynn` is not installed into `.venv`.** `python -m pynn.verify` and `import pynn` work
only from the repository root today. `pip install -e .` fixes it.

**The `python3` Jupyter kernel on this machine is broken.** `~/Library/Jupyter/kernels/python3`
points at `clearpath-match-model/.venv/bin/python3`, which has no `ipykernel` installed —
any notebook opened with the default "Python 3" kernel dies with *"Kernel died before
replying to kernel_info"*. Two ways out:

```bash
# A: repoint the global kernel at something that has ipykernel
python3 -m ipykernel install --user --name python3 --display-name "Python 3"

# B: use this project's kernel (already registered, project-local)
.venv/bin/python -m ipykernel install --sys-prefix --name pynn-venv --display-name "Python 3"
```

`B` is already done — that is what `pynn-venv` is. It lives in
`.venv/share/jupyter/kernels/pynn-venv/`, so it disappears when the venv does, and it is
not committed. See [§9](#9-what-pynn-venv-actually-is).

---

## 8. What CI runs

`.github/workflows/ci.yml`, on every push to `main` and every pull request, across Python
3.10 / 3.11 / 3.12 / 3.13:

```bash
pip install -e ".[dev]"
ruff check pynn tests examples scripts benchmarks
ruff format --check pynn tests examples scripts benchmarks
mypy
pytest -m "not external" --cov=pynn --cov-report=xml --cov-report=term-missing
python -m pynn.verify
# then, on 3.12 only: upload coverage.xml to Codecov
```

To reproduce a CI run locally, in one line:

```bash
ruff check pynn tests examples scripts benchmarks \
  && ruff format --check pynn tests examples scripts benchmarks \
  && mypy \
  && pytest -m "not external" --cov=pynn \
  && python -m pynn.verify
```

---

## 9. What `pynn-venv` actually is

A **Jupyter kernelspec** — a small JSON file telling Jupyter which Python interpreter to
launch for a notebook:

```json
{
  "argv": [".venv/bin/python", "-Xfrozen_modules=off", "-m", "ipykernel_launcher",
           "-f", "{connection_file}"],
  "display_name": "Python 3",
  "language": "python"
}
```

It exists because the machine's default `python3` kernel points at an unrelated project's
virtualenv (see §7). Registering one under `--sys-prefix` puts it inside `.venv/share/`
rather than in your home directory, so it is scoped to this project, is not committed,
and vanishes with the venv.

The committed notebook records the **generic `python3`** kernelspec in its metadata, not
`pynn-venv`, so it opens for anyone who clones the repository. `pynn-venv` is only used
when *executing* it here.

**Extras are unrelated to this.** "Extras" are the optional dependency groups in
`pyproject.toml` — `[notebook]`, `[benchmark]`, `[dev]`, and so on. They are ordinary
packaging metadata: `pip install -e ".[notebook]"` installs matplotlib, pandas, and
jupyter alongside the library. Declaring them means a fresh clone can install exactly what
a given task needs without guessing.

---

## 10. Codecov

The coverage badge in the README will read *unknown* until the repository is enabled at
[codecov.io](https://codecov.io). CI already produces and uploads `coverage.xml`; nothing
in the repository needs to change except adding the token. See the setup walkthrough in
the chat, or `.github/workflows/ci.yml` for the upload step.

---

## Quick reference

| Task | Command |
|---|---|
| Tests | `pytest -m "not external"` |
| Tests + coverage floor | `pytest -m "not external" --cov=pynn` |
| Self-verification | `python -m pynn.verify` |
| Lint | `ruff check pynn tests examples scripts benchmarks` |
| Format | `ruff format pynn tests examples scripts benchmarks` |
| Types | `mypy` |
| Smoke test | `python scripts/smoke_test.py` |
| Benchmarks | `python -m benchmarks.benchmark --markdown` |
| MNIST data | `python scripts/download_mnist.py` |
| Notebook | `jupyter lab examples/mnist.ipynb` |

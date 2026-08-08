# USAGE

Every command this project supports, what it does, and what a healthy result looks like.
Run everything from the repository root unless noted.

---

## 0. Setup

### Which Python

| | |
| --- | --- |
| Declared support | **3.10 – 3.14** (`requires-python = ">=3.10"`, CI matrix) |
| Local `.venv` | **3.14.1** (Homebrew `python@3.14`) |

CI tests every version in that range, including the 3.14 this project is developed on, so
a local pass and a CI pass mean the same thing.

### Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`.venv/` is gitignored — yours to delete and rebuild at any time. Nothing in the
repository depends on its contents; those three commands reproduce it exactly.

### Dependency groups

Three files cover the three situations anyone is actually in:

| File | Installs | For |
| --- | --- | --- |
| `requirements.txt` | the library, NumPy only | running `pynn`, `python -m pynn.verify` |
| `requirements-dev.txt` | + tests, lint, types, examples, notebook, benchmarks | working on the repository |
| `requirements-external.txt` | + torch, tensorflow | the `external`-marked comparison tests |

Finer control lives in `pyproject.toml`, which declares the extras those files point at
and is where versions are pinned:

```bash
pip install -e ".[dev]"        # pytest, pytest-cov, ruff, mypy — what CI installs
pip install -e ".[examples]"   # scikit-learn
pip install -e ".[mnist]"      # + pandas
pip install -e ".[notebook]"   # matplotlib, jupyterlab, ipykernel, nbclient
pip install -e ".[benchmark]"  # torch
pip install -e ".[numba]"      # compiles the convolution backward pass
```

**Two groups sit outside `requirements-dev.txt` on purpose:**

- **`external`** — TensorFlow lags new Python releases and has **no wheel for 3.14**, so
  folding it in would make the one-command install fail on the very interpreter this
  project uses. The tests that need it are marked `external` and skip themselves.
- **`numba`** — compiles `col2im`, the scatter in the reverse pass of `conv2d` and the
  pooling layers, which profiling puts at about 40% of a CNN training step. Worth roughly
  1.27x end-to-end on a small CNN and nothing at all on an MLP, which never touches it.
  Optional because `llvmlite` is 130 MB; the library is fully functional without it and
  the results are identical either way.

### Check it worked

```bash
python -c "import pynn; print(pynn.__file__)"   # works from any directory
python -m pynn.verify                           # 332/332 passed (OK)
```

`requirements-dev.txt` installs the library in editable mode, so `pynn` imports from
anywhere, not only from the repository root.

---

## 1. Tests

```bash
pytest                              # everything; torch/tf tests skip if absent
pytest -m "not external"            # NumPy-only, what CI runs
pytest --cov=pynn                   # + coverage, enforces the 95% floor
pytest --cov=pynn --cov-report=term-missing   # + which lines are uncovered
```

**Expected:** `767 passed, 1 skipped, 6 deselected` · `Total coverage: ~98.3%`

Narrower runs:

```bash
pytest tests/test_gradcheck.py      # the 201 gradient checks, one test per operation
pytest tests/core/module_test.py    # the module tree
pytest tests/nn/layers_test.py      # Dropout, LayerNorm, BatchNorm, pooling
pytest tests/nn/containers_test.py  # ModuleList and ModuleDict
pytest tests/nn/recurrent_test.py    # Embedding, RNNCell, LSTMCell
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
gradients: 201/201 passed (OK)
invariants: 107/107 passed (OK)
stability: 24/24 passed (OK)

pynn.verify: 332/332 passed (OK)
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
python - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("examples/mnist.ipynb", as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "examples/mnist.ipynb")
PY
```

Takes about 5 minutes: ~40 s for the MLP, ~3 min for the CNN, the rest is plotting.
`kernel_name="python3"` is the kernel the notebook itself records, and inside this
virtualenv it resolves to this project's interpreter — see [section 9](#9-jupyter-kernels).

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
| --- | --- | --- | --- |
| MLP 784-256-256-10 | ~3.7 | ~2.2 | ~1.7x |
| CNN 2 conv + 2 pool | ~80 | ~8 | ~10x |

Run-to-run variation is roughly ±15%. Re-measure before quoting new numbers in the README.

---

## 7. Known gaps

Nothing environmental is outstanding. The three that were open are closed:

- ~~The local venv runs a Python CI does not test~~ — 3.14 is in the CI matrix.
- ~~`pynn` is not installed into `.venv`~~ — installed editable via `requirements-dev.txt`.
- ~~The machine's default Jupyter kernel is broken~~ — see [section 9](#9-jupyter-kernels).

One dormant item remains, outside this repository: the user-level `cs224n` kernelspec at
`~/Library/Jupyter/kernels/cs224n` points at a miniforge environment that no longer
exists. It is harmless — a kernel is only launched if you select it — but it will fail if
you do. Remove it with:

```bash
jupyter kernelspec uninstall cs224n
```

---

## 8. What CI runs

`.github/workflows/ci.yml` runs two jobs on every push to `main` and every pull request.

**`check`**, across Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14:

```bash
pip install -e ".[dev]"
ruff check pynn tests examples scripts benchmarks
ruff format --check pynn tests examples scripts benchmarks
mypy
pytest -m "not external" --cov=pynn --cov-report=xml --cov-report=term-missing
python -m pynn.verify
# then, on 3.12 only: upload coverage.xml to Codecov
```

**`numba`**, on 3.12 only, installing `".[dev,numba]"`. The `check` matrix installs only
`[dev]`, so it never runs the compiled `col2im` — a Numba-specific compilation failure
would reach a user before it reached CI. The job asserts `NUMBA_AVAILABLE` is true before
running anything, so it cannot pass on the NumPy fallback if the extra failed to install.


To reproduce a CI run locally on the current interpreter:

```bash
ruff check pynn tests examples scripts benchmarks \
  && ruff format --check pynn tests examples scripts benchmarks \
  && mypy \
  && pytest -m "not external" --cov=pynn \
  && python -m pynn.verify
```

To reproduce it on **every** version in the matrix — which is the useful one, because
the failures worth catching are the ones a single interpreter cannot see:

```bash
scripts/ci_matrix.sh
```

It builds a throwaway virtualenv per version under `$TMPDIR/pynn-ci-matrix` (reused on
later runs, override with `CI_MATRIX_ENVS`) and runs each CI step in order, reporting
per-step pass or fail. Versions not on `PATH` as `python3.10` … `python3.14` are
reported and skipped; `brew install python@3.11` and so on to fill the gaps.

Three separate CI failures have already come from version-specific behaviour that the
newest interpreter cannot reproduce: mypy aborting on older NumPy stubs, `argparse`
rejecting an empty `nargs="*"` before 3.12, and annotation faults only the newest NumPy
stubs catch. Run this before pushing anything that touches typing, the CLI, or NumPy
usage.

---

## 9. Jupyter kernels

A **kernelspec** is a small JSON file telling Jupyter which interpreter to launch for a
notebook. They are discovered from several locations at once, and a user-level one
*shadows* the environment-level one of the same name — which is how a single stale file
breaks every notebook on a machine.

That is what had happened here. `~/Library/Jupyter/kernels/python3` pointed at an
unrelated project's virtualenv that had no `ipykernel` installed, so any notebook opened
with the default "Python 3" kernel died with *"Kernel died before replying to
kernel_info"* — in this repository and everywhere else.

**Fixed by removing it**, with a backup at `~/Library/Jupyter/kernels-backup-2026-08-06/`
in case it is ever wanted. With the stale file gone, the working kernel that `ipykernel`
installs into each virtualenv is visible again:

```
$ jupyter kernelspec list
  cs224n     ~/Library/Jupyter/kernels/cs224n        # dormant, see section 7
  python3    .venv/share/jupyter/kernels/python3     # this project's interpreter
```

That `python3` comes from `ipykernel` inside `.venv`, so it resolves to this project's
Python whenever Jupyter runs from the activated environment. It is also the kernel
`examples/mnist.ipynb` records in its metadata, so the notebook opens and runs with no
special-casing — for you, and for anyone who clones the repository.

If a virtualenv ever lacks a kernel, this recreates one scoped to it:

```bash
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"
```

Use `--sys-prefix` (inside the venv), never `--user` (in your home directory) — `--user`
is exactly what created the problem above.

---

## 10. Codecov

The badge in the README reads *unknown* until the repository is enabled at
[codecov.io](https://codecov.io). Everything on this side is in place: CI produces
`coverage.xml` and uploads it with `codecov/codecov-action@v5`, passing
`token: ${{ secrets.CODECOV_TOKEN }}`.

What is left is outside the repository:

1. Sign in to Codecov with GitHub and select `tarickali/pynn`.
2. Copy the repository upload token.
3. GitHub, repo, **Settings → Secrets and variables → Actions → New repository secret**,
   named exactly **`CODECOV_TOKEN`**.
4. Push to `main`. The badge resolves after the first successful upload.

Do not paste Codecov's suggested workflow snippet — the upload step already exists, and a
second one would upload the same report twice.

---

## Quick reference

| Task | Command |
| --- | --- |
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
| Full install | `pip install -r requirements-dev.txt` |
| CI on every Python | `scripts/ci_matrix.sh` |
| Refresh READ_FILES.md | `python scripts/generate_read_files.py` |

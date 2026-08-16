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
| `requirements-dev.txt` | + tests, lint, types, examples, notebook, benchmarks, hooks, release tooling | working on the repository |
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
pip install -e ".[release]"    # build, twine — see section 11
pip install -e ".[hooks]"      # pre-commit — see section 3
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
python -c "import pynn; print(pynn.__file__)"       # works from any directory
python -c "import pynn; print(pynn.__version__)"    # 0.1.0
python -m pynn.verify                               # 346/346 passed (OK)
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

**Expected:** `822 passed, 1 skipped, 6 deselected` · `Total coverage: ~98.3%`

On Python 3.10 it is `820 passed, 3 skipped`: two of the packaging checks parse
`pyproject.toml`, and `tomllib` is standard library only from 3.11.

Narrower runs:

```bash
pytest tests/test_gradcheck.py      # the 209 gradient checks, one test per operation
pytest tests/core/module_test.py    # the module tree
pytest tests/nn/layers_test.py      # Dropout, LayerNorm, BatchNorm, pooling
pytest tests/nn/containers_test.py  # ModuleList and ModuleDict
pytest tests/nn/recurrent_test.py    # Embedding, RNNCell, LSTMCell
pytest tests/viz_test.py            # the DOT dump of the tape
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
gradients: 209/209 passed (OK)
invariants: 113/113 passed (OK)
stability: 24/24 passed (OK)

pynn.verify: 346/346 passed (OK)
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

**Expected:** `All checks passed!` · `94 files already formatted` · `Success: no issues
found in 51 source files`

Ruff covers the **code cells of `examples/mnist.ipynb` and `examples/char_rnn.ipynb`**
too — it parses `.ipynb` natively. `ruff format` on a notebook rewrites cell sources and
leaves outputs alone.
Mypy is scoped to `pynn/` only (`files = ["pynn"]`).

`ruff` and `mypy` are pinned to exact versions in `pyproject.toml`, not floors. They are
the only two dependencies whose *output* is the check, so a release that lands overnight
turns a repository nobody touched red — a newer `ruff format` reformats code that is
clean today, a newer mypy reports errors the current one does not. The cost is that
upgrading is now a deliberate edit; run `scripts/ci_matrix.sh` before moving a pin.

### Pre-commit hooks

The same checks, run before the commit rather than after the push.

```bash
pip install -e ".[hooks]"
pre-commit install          # once per clone; writes .git/hooks/pre-commit
pre-commit run --all-files  # the whole tree, without committing
```

**Expected:** five `Passed` lines and nothing modified.

```
ruff check...............................................................Passed
ruff format..............................................................Passed
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
mypy.....................................................................Passed
```

Three of the five hooks *fix* rather than report — `ruff format`,
`trailing-whitespace`, and `end-of-file-fixer`. pre-commit fails any run in which a hook
modified a file, so a rewritten file is not quietly committed: re-stage it and commit
again.

Hook revisions in `.pre-commit-config.yaml` are pinned to the same ruff and mypy the
`dev` extra installs. Bump both in one commit or the hook and CI start disagreeing about
what clean means.

**Activate the virtualenv first.** The mypy hook is a `local` / `language: system` hook,
so it runs the `mypy` on `PATH` rather than one pre-commit installed on its own. That is
deliberate: mypy's findings depend on the NumPy stubs it can see, and an isolated hook
environment has no NumPy unless a version is pinned into it — a numpy pin this project
avoids everywhere else, precisely so the CI matrix can test several. Without the
virtualenv the hook fails with `Executable mypy not found`, which is a loud failure
rather than a check that quietly passes on nothing.

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

### Data

`examples/data/` is gitignored, so neither corpus is in the repository.

```bash
python scripts/download_mnist.py        # writes examples/data/mnist/train.csv (~227 MB)
python scripts/download_shakespeare.py  # writes examples/data/shakespeare/input.txt
```

`examples/mnist.py` falls back to synthetic data when the file is absent;
`examples/mnist.ipynb` raises with instructions instead, because a notebook of synthetic
results would be misleading. `examples/char_rnn.ipynb` raises for the same reason.

The Shakespeare script needs nothing beyond the standard library — the corpus is one
plain text file over HTTP, and it is ~1.1 MB, so it is quick enough that the notebook
does not cache anything beyond the file itself. It refuses a response under 1 MB rather
than writing it, since a truncated transfer or an error page would otherwise be trained
on without complaint.

### The tape figure

`docs/tape.svg`, the computation graph the README opens with, is generated rather than
drawn:

```bash
python scripts/generate_tape_figure.py   # writes docs/tape.dot and docs/tape.svg
```

**Expected:** `wrote docs/tape.dot — 15 nodes`, then `wrote docs/tape.svg`.

Only the SVG step needs Graphviz on `PATH` (`brew install graphviz`, `apt install
graphviz`). Without it the DOT is still written, the script says so, and it exits 1 —
`pynn.viz` emits text and never shells out, so the library itself needs nothing.

---

## 5. The notebooks

Both are committed **with their outputs**, so they render on GitHub without being run.
Only re-run one if you change it.

```bash
pip install -e ".[notebook]"
jupyter lab examples/mnist.ipynb
jupyter lab examples/char_rnn.ipynb
```

To re-execute headlessly and write the outputs back in place:

```bash
python - <<'PY'
import nbformat
from nbclient import NotebookClient
path = "examples/mnist.ipynb"          # or examples/char_rnn.ipynb
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, path)
PY
```

`mnist.ipynb` takes about 5 minutes: ~40 s for the MLP, ~3 min for the CNN, the rest is
plotting. `char_rnn.ipynb` is about the same, nearly all of it the LSTM's training loop.
`kernel_name="python3"` is the kernel the notebooks themselves record, and inside this
virtualenv it resolves to this project's interpreter — see [section 9](#9-jupyter-kernels).

`char_rnn.ipynb` calls `loss.free_graph()` inside its training loop, and that is
deliberate rather than superstition. Each Tensor holds its reverse pass as a closure over
itself, so a finished graph is a reference cycle that nothing reclaims on its own; one
step of a 64-step unrolled LSTM is ~150 MB, and CPython's heuristic for running a full
collection counts objects rather than bytes. `free_graph` clears each node's `children`
and `reverse` over the order `backward` walks, which breaks every cycle, so reference
counting takes the step's graph back before the next one is built. Measured over 400
steps, a fresh process per row, with `python -m benchmarks.memory`:

| | ms/step | peak RSS |
| --- | --- | --- |
| left to CPython | 242.0 | 3,813 MB |
| `gc.collect()` every 4 steps | 79.1 | 1,806 MB |
| `loss.free_graph()` | **81.0** | **860 MB** |
| both | 82.1 | 918 MB |

Leaving it alone is **3.0x slower**, because allocating against a heap that is mostly
garbage costs more than sweeping it. `free_graph` matches the collector cadence on wall
clock — the difference between those two rows is inside run-to-run variation — and halves
the peak, since it hands the memory back at a known point rather than whenever the
collector next runs. **Collecting on top of it buys nothing**, which is the expected
result: there are no cycles left to find. The notebook used to call `gc.collect()` every
fourth step and no longer does.

Two traps if you re-measure. **Use at least 400 steps**: over 60 the same benchmark says
the uncollected run is the fast one, because the cost of the garbage is the cost of
allocating around it and that takes a while to show up. And **read peak RSS from `ps`**,
not from `resource.getrusage`, which reported near-identical peaks here for
configurations whose real peaks differed by more than 2 GB.

Anything that unrolls a long recurrence will want the same line. A feedforward model
will not notice either way — its graph is a few dozen nodes.

After editing cells, re-lint and re-format before committing:

```bash
ruff format examples/char_rnn.ipynb && ruff check examples/char_rnn.ipynb
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

### Memory

A second harness, for the other resource:

```bash
python -m benchmarks.memory                    # every configuration, 400 steps each
python -m benchmarks.memory --only none free   # a subset
python -m benchmarks.memory --steps 800        # longer
```

It runs `examples/char_rnn.ipynb`'s model — batch 32, a 64-step unrolled `LSTMCell` — in
a **fresh process per configuration**, sampling each child's RSS from `ps` while it runs,
and reports mean ms/step and peak RSS. Needs only NumPy. The table it produces is the one
in [section 5](#5-the-notebooks); the module docstring carries the two traps.

Expect the uncollected configuration to want ~4 GB, and to be killed by the OS rather
than merely run slowly on a machine that does not have it free. That is the failure this
measures.

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
Python whenever Jupyter runs from the activated environment. It is also the kernel both
notebooks record in their metadata, so they open and run with no special-casing — for
you, and for anyone who clones the repository.

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

## 11. Building and publishing

```bash
pip install -e ".[release]"
rm -rf dist/                  # twine uploads everything it is handed, stale files included
python -m build               # writes dist/pynn-<version>-py3-none-any.whl and .tar.gz
twine check dist/*
```

**Expected:** `Successfully built pynn-0.1.0.tar.gz and pynn-0.1.0-py3-none-any.whl`,
then `PASSED` for both artifacts. `dist/` and `build/` are gitignored.

`twine check` renders the README the way PyPI will and fails on markup PyPI would
reject — the failure mode it exists for is a project page that shows raw markup, which
is only visible after the upload it is too late to take back.

### The version

Written in exactly one place, `pynn/__init__.py`:

```python
__version__ = "0.1.0"
```

`pyproject.toml` carries `dynamic = ["version"]` and reads that attribute, so a release
is a single edit and there is no second literal to disagree with it. Keep it a plain
string: setuptools parses the file rather than importing it, and importing it would
need NumPy, which an isolated build environment does not have.

`tests/test_packaging.py` asserts the attribute and the installed distribution's
metadata agree, so a bump that was never reinstalled fails a test rather than shipping.

### The type marker

`pynn/py.typed` is [PEP 561](https://peps.python.org/pep-0561/)'s marker. Without it a
type checker in a consuming project ignores this library's annotations entirely and
reports it as untyped — the library is fully annotated and none of it is visible. It is
listed under `[tool.setuptools.package-data]` because setuptools ships `.py` files and
nothing else on its own, so a marker that exists in the repository and is not declared
there is absent from every wheel built from it.

Confirm it survived a build rather than assuming it did:

```bash
python -m zipfile -l dist/pynn-0.1.0-py3-none-any.whl | grep py.typed
```

### TestPyPI

**Nothing has been uploaded.** Publishing is a deliberate act — a version number on an
index can never be reused, even after a delete — so the command lives here rather than
in a script or a CI job.

```bash
python -m twine upload --repository testpypi dist/*
```

It needs an API token from [test.pypi.org](https://test.pypi.org/manage/account/token/),
either in `~/.pypirc` under a `[testpypi]` section or as `TWINE_USERNAME=__token__` and
`TWINE_PASSWORD=pypi-...` in the environment.

Then install it back into a throwaway environment, which is the part that actually
proves the packaging:

```bash
python -m venv /tmp/pynn-check && /tmp/pynn-check/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ pynn
/tmp/pynn-check/bin/python -m pynn.verify
```

The `--extra-index-url` is not optional: TestPyPI is a separate index and NumPy is not
reliably on it, so without a fallback the dependency fails to resolve.

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
| All of the above, pre-commit | `pre-commit run --all-files` |
| Smoke test | `python scripts/smoke_test.py` |
| Benchmarks | `python -m benchmarks.benchmark --markdown` |
| Memory benchmark | `python -m benchmarks.memory` |
| Build the artifacts | `python -m build && twine check dist/*` |
| MNIST data | `python scripts/download_mnist.py` |
| Shakespeare corpus | `python scripts/download_shakespeare.py` |
| Notebooks | `jupyter lab examples/mnist.ipynb examples/char_rnn.ipynb` |
| Full install | `pip install -r requirements-dev.txt` |
| CI on every Python | `scripts/ci_matrix.sh` |
| Refresh READ_FILES.md | `python scripts/generate_read_files.py` |
| Refresh the tape figure | `python scripts/generate_tape_figure.py` |

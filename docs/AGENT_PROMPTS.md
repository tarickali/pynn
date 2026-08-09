# Agent prompts

Four independent work packages from [`TASKS.md`](../TASKS.md). Each is self-contained and
touches a mostly disjoint set of files, so they can run in any order — or in parallel, if
each agent works on its own branch and rebases before merging.

Copy one prompt verbatim into a fresh agent session.

| Prompt | TASKS.md items | Rough size |
| --- | --- | --- |
| A — Packaging and process | 1, 2, 3, 4 | small, mostly config |
| B — Testing, layers, performance | 6, 7, 8 | large |
| C — A second example domain | 9 | medium, one notebook |
| D — Graph visualizer | 5 | small |

---

## Shared preamble

Every prompt below already includes this. It is repeated here so it can be edited once.

> You are working on **PyNN**, a NumPy autodiff and neural-network library at
> `/Users/tarickali/workspace/github/pynn`. It is a portfolio project for software/ML
> engineering roles, so the bar is production-quality: correct, tested, documented, and
> defensible in an interview.
>
> **Read first, in this order:** `README.md`, `USAGE.md`, `TASKS.md`, `docs/DESIGN.md`.
> `PROJECT_REVIEW.md` is a historical record of two review passes — read it for context,
> but `TASKS.md` is the live queue.
>
> **State of the repo.** Green on Python 3.10–3.14: 767 tests, 332 checks from
> `python -m pynn.verify` (201 of them numerical gradient checks), 98.3% line coverage
> with a 95% floor enforced in CI. `ruff` and `mypy` are clean across `pynn tests
> examples scripts benchmarks`, including the code cells of `examples/mnist.ipynb`.
>
> **Environment.** Use `.venv/bin/python` (Python 3.14) — a bare `pytest` on `PATH`
> resolves to a Homebrew install without `pytest-cov`. `pynn` is installed editable.
>
> **Before every commit, all of these must pass:**
> ```bash
> .venv/bin/ruff check pynn tests examples scripts benchmarks
> .venv/bin/ruff format --check pynn tests examples scripts benchmarks
> .venv/bin/mypy
> .venv/bin/python -m pytest -m "not external" --cov=pynn
> .venv/bin/python -m pynn.verify
> ```
> If you touch typing, the CLI, NumPy usage, or anything version-sensitive, also run
> `scripts/ci_matrix.sh`, which runs every CI step against all five supported Pythons in
> throwaway virtualenvs. Three CI failures have already come from behaviour the newest
> interpreter cannot reproduce.
>
> **House style.**
> - Match the surrounding code: NumPy-style docstrings, comments that explain *why* and
>   what breaks otherwise, never *what* the line does.
> - **Any new differentiable operation must be added to
>   `pynn.verify.gradients.gradient_cases`**, not only covered by hand-written tests.
>   Include a reused-input variant — that is the shape that catches a reverse pass which
>   assigns to `grad` instead of accumulating into it, and it is invisible in the
>   single-consumer form.
> - Behavioural properties that a test cannot see go in `pynn/verify/invariants.py`.
> - Prefer failing loudly over failing silently. The library refuses a plain list of
>   Modules, a non-scalar `backward()`, and an optimizer handed a bare parameter dict,
>   all for this reason.
> - Commits: short imperative subject, body explaining *why*. No `Co-authored-by`
>   trailer. One concern per commit. Only commit when green. **Do not push.**
> - After committing, run `.venv/bin/python scripts/generate_read_files.py` and commit
>   the refreshed `READ_FILES.md` on its own.
> - Update `TASKS.md` to mark what you finished, and `README.md` / `USAGE.md` /
>   `docs/DESIGN.md` if you changed anything they claim.
>
> Report back with: what shipped, commit SHAs, the test/verify/coverage numbers, and
> anything you deliberately left out and why.

---

## Prompt A — Packaging and process

**TASKS.md items 1, 2, 3, 4.** Small and mostly configuration, but it is what makes the
project look finished to someone landing on the repo.

```text
[paste the shared preamble here]

Your job is TASKS.md items 1, 2, 3, and 4 — packaging and process. Read those entries
first; the detail below is what matters beyond them.

1. CONTRIBUTING.md, CHANGELOG.md, and a v0.1.0 tag.
   The contributing guide should be a "how to add a layer" walkthrough, because that is
   the question this codebase's structure actually answers. Walk through a real example
   end to end: subclass Module, implement forward / build / hyperparameters, write the
   functional op with its reverse closure, add a gradcheck entry so the op joins the
   sweep, add an invariant if the layer has behaviour a gradient check cannot see (a
   train/eval difference, a buffer). Point at pynn/nn/modules.py::Dropout as the smallest
   complete example and pynn/functional/modules.py::layer_norm as one with a hand-written
   fused reverse. Explain *why* the gradcheck entry is required, not just that it is.

   CHANGELOG.md should follow Keep a Changelog. There is no released history, so v0.1.0
   is one entry describing the library as it stands. Do not invent dates for past work —
   use the git history if you want them (`git log --date=short`).

   Create the tag as an annotated tag locally. Do not push it.

2. Packaging polish.
   - `pynn/py.typed` plus the setuptools package-data wiring, so the type hints are
     visible to consumers. Verify it actually ships: build a wheel and confirm the marker
     is inside it.
   - `__version__` in `pynn/__init__.py`. The version currently lives only in
     pyproject.toml, and two sources of truth will drift — read it from installed
     metadata via importlib.metadata with a fallback, or make pyproject read it from the
     package. Pick one, and say in a comment why.
   - Add a test asserting the two agree, so the drift is caught rather than discovered.
   - TestPyPI: prepare everything needed (build, twine check) and document the upload
     command in USAGE.md, but DO NOT upload. Publishing is the user's call.

3. Pin the tool versions.
   `ruff>=0.6` and `mypy>=1.11` let CI install anything newer than what is installed
   locally, and a newer `ruff format` can reformat code that is clean here. Pin them to
   the versions currently installed in .venv (check with `.venv/bin/python -m pip list`).
   Explain the trade-off in a comment: pinning trades "silent CI breakage on an upgrade"
   for "you have to bump it deliberately". Do not pin numpy — it is a runtime dependency
   and the matrix deliberately tests several versions.

4. .pre-commit-config.yaml with ruff, ruff-format, mypy, trailing-whitespace, and
   end-of-file-fixer. Pin the hook revisions to match item 3. Verify it runs clean on the
   current tree (`pre-commit run --all-files`) and document it in USAGE.md. If any hook
   would reformat committed files, fix the files rather than loosening the hook.

Constraints:
- Do not change library behaviour. This package is configuration, docs, and metadata.
- If pre-commit or the packaging changes force a code change, that is a signal — flag it
  rather than quietly reformatting half the repo.
```

---

## Prompt B — Testing, layers, and performance

**TASKS.md items 6, 7, 8.** The largest package. Item 8 is measurement-driven and has
numbers already recorded in `TASKS.md` — respect them.

```text
[paste the shared preamble here]

Your job is TASKS.md items 6, 7, and 8. Read those entries first — item 8 in particular
already contains measurements you should not re-derive from scratch, only extend.

6. Property-based tests over unbroadcast.
   Add Hypothesis to the dev extra and write property tests for
   pynn/core/utils.py::unbroadcast and matrix_multiply_gradients. These are the two
   fiddliest functions in the library — unbroadcast composes two different reduction
   rules, and matrix_multiply_gradients handles matmul's vector promotion and batch
   broadcasting. Both are currently covered by a hand-written list of shapes, which is
   exactly the kind of coverage that misses the case nobody thought of.

   The property worth testing: for any pair of broadcast-compatible shapes,
   unbroadcast(ones(broadcast_shape), operand_shape) equals the number of times the
   operand was replicated. And the adjoint identity for matmul:
   <A @ B, G> == <A, dA> for the dA that matrix_multiply_gradients returns.

   Generate shapes with Hypothesis strategies rather than hand-listing them. Keep the
   example budget modest so the suite stays under ~10s.

7. More layers, losses, and optimizers.
   Pick from the table in TASKS.md item 7. Do NOT do all of it — choose what is coherent
   and finish it properly rather than half-landing six things. Suggested slice, in order
   of value:
     - Unflatten / Reshape as the inverse of Flatten, and Identity as a layer. Small, and
       they complete an obvious gap.
     - A packed-sequence RNN / LSTM layer wrapping the existing cells, so a caller who
       does not need a custom loop does not have to write one. The cells are deliberately
       loop-free; this is the convenience layer over them.
     - KLDivLoss and HingeLoss, using the shared _reduce helper in
       pynn/functional/losses.py so the three reduction modes stay consistent.
     - NAdam, ReduceLROnPlateau, OneCycleLR.
   Every new op needs a gradcheck entry with a reused-input variant. Every new optimizer
   needs a closed-form reference transcription in tests/optim/optimizers_test.py — "the
   loss went down" does not distinguish a correct update rule from a nearly-correct one.

8. Further acceleration — measurement first, and the first win needs no dependency.
   TASKS.md item 8 records that SGD.update is 32% of an MLP training step, almost all of
   it allocation: every line builds a fresh full-size array. An in-place NumPy rewrite
   measured 1.8-2.5x with no dependency and no second implementation; a fused numba
   kernel measured 4.6-7.2x but costs a dual implementation per optimizer, five of them
   with four flag variants each.

   Do the in-place rewrite for all five optimizers. Keep the maths identical — the
   closed-form reference tests must pass unchanged, and that is the point of them.
   Watch for aliasing: velocity buffers are stored in the optimizer's cache and mutating
   them in place is fine, but param.data must not be mutated in a way that surprises a
   caller holding a reference.

   Then re-profile with:
     .venv/bin/python -m benchmarks.benchmark --steps 25
   and a cProfile run of an MLP step, and report the new share. Only if SGD.update is
   still a large fraction should you consider a numba kernel — and if you do, put it
   behind the same NUMBA_AVAILABLE pattern as pynn/utils/array.py::scatter_windows, with
   tests asserting the two implementations agree.

   Update the benchmark table in README.md if the numbers move.

Constraints:
- Item 8 must not change any optimizer's arithmetic. If a reference test needs updating,
  you have changed behaviour — stop and flag it.
- Do not add a dependency without a measurement justifying it.
```

---

## Prompt C — A second example domain

**TASKS.md item 9.** One notebook, but it is the most visible artifact in the repo after
the README.

```text
[paste the shared preamble here]

Your job is TASKS.md item 9: a second example domain, showing the library generalizes
past image classification.

Build examples/char_rnn.ipynb — a character-level language model on a small public-domain
text. Everything it needs already exists: Embedding, LSTMCell, differentiable slicing,
SparseCategoricalCrossentropy, AdamW, clip_grad_norm, and the LR schedules.

Study examples/mnist.ipynb first and match it: it is committed WITH its outputs so it
renders on GitHub, its code cells are held to the same ruff rules as the library, and it
explains what is happening rather than only showing it.

What the notebook should cover:
- A small corpus, downloaded by a script in scripts/ the way scripts/download_mnist.py
  works, or embedded if it is small enough to commit. Public domain only — Tiny
  Shakespeare is the conventional choice. Do not commit anything large; examples/data/ is
  gitignored.
- Character vocabulary and windowed batching. get_batches shuffles by default, which is
  what you want here.
- Embedding -> LSTMCell unrolled over the window -> Linear head, trained with AdamW.
  Use clip_grad_norm — recurrent models are exactly the case it exists for, and the
  notebook should say why and show the clipped norm over training.
- Training curves, and *sampled text at several checkpoints* so a reader can see it go
  from noise to something word-shaped. That progression is the whole point of the demo.
- A temperature-controlled sampling function, with a short explanation of what
  temperature does to the softmax.
- Honest framing: this is a small model on a small corpus and the output will be
  imperfect. Say so. Do not oversell it.

Then re-execute the notebook so the committed outputs match the committed source, lint it
(`.venv/bin/ruff format examples/char_rnn.ipynb && .venv/bin/ruff check
examples/char_rnn.ipynb`), and link it from README.md next to the MNIST notebook.

If you use matplotlib, read the dataviz guidance the repo already follows in
examples/mnist.ipynb: a validated two-colour categorical palette for train/test series,
a single-hue sequential ramp for magnitude, a diverging ramp with a neutral midpoint for
signed values, never a dual y-axis.

Budget: keep total training under ~5 minutes on a laptop CPU. It is a demonstration, not
a result. Report the wall-clock time in the notebook.

Constraints:
- Do not add library code unless the notebook genuinely cannot be written without it. If
  it can't, that is a finding worth reporting — say what was missing.
- The notebook must not require the numba extra.
```

---

## Prompt D — Graph visualizer

**TASKS.md item 5.** Small and self-contained, and it produces the single most useful
figure the README is missing.

```text
[paste the shared preamble here]

Your job is TASKS.md item 5: a computation-graph visualizer.

Add Tensor.to_dot() (or a pynn.viz module — your call, argue for it) that walks the tape
from a Tensor and emits Graphviz DOT. The tape already carries what you need: every
Tensor has `children` and a `forward` string naming the operation that produced it.

Requirements:
- Pure Python, no new runtime dependency. Emit DOT text; do not shell out to Graphviz or
  require the graphviz package to *produce* the output. Rendering it is the caller's
  problem, and the docstring should show both `dot -Tpng` and the IPython display route.
- Label each node with its operation and shape. Leaves — tensors with no children — should
  be visually distinct from computed nodes, and parameters distinct again if you can
  detect them (`trainable` is a reasonable proxy; say so in a comment if you use it).
- Handle the shapes that make this non-trivial: a tensor consumed twice must appear once
  with two edges, not twice. Use id() for node identity, matching how backward's visited
  set works.
- A `max_nodes` cap with a clear truncation marker. An unrolled 300-step LSTM is tens of
  thousands of nodes and would produce an unusable file; the docstring should say so.
- Tests: node and edge counts for a known small graph, the diamond case (a tensor used
  twice appears once), a graph built under no_grad (which has no children, so the output
  is a single node), and the truncation path.

Then use it: generate a figure for a small MLP forward+loss, commit it as an SVG or PNG
under docs/, and put it in README.md near the top. docs/DESIGN.md §2 describes the tape
as "closures, not an operator registry" — a picture of an actual tape is the single most
effective way to show a reader that is real, and the README currently has no figure at
all.

Keep the committed image small (a 2-layer MLP on a batch of 2 is enough — a dozen nodes,
not a hundred).

Constraints:
- Do not add graphviz, pydot, or networkx as a dependency. Emitting text is the whole job.
- Do not change Tensor's existing behaviour. This is read-only over the tape.
```

---

## Running these in parallel

The four packages touch mostly disjoint files, but three overlaps are worth knowing:

- **A, B, C, and D all touch `README.md`, `USAGE.md`, and `TASKS.md`.** Expect conflicts
  there; they are prose, so they resolve by hand easily.
- **B and D both add tests**, but in different files.
- **B's item 8 touches every optimizer**; nothing else does.
- `READ_FILES.md` is generated, so never merge it — regenerate after merging with
  `.venv/bin/python scripts/generate_read_files.py`.

Suggested order if running sequentially: **A** (cheap, makes the repo look finished),
then **D** (produces the README figure), then **C** (the big visible artifact), then **B**
(the largest, and the one most likely to want its own review).

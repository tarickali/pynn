# Agent prompts

Independent work packages covering items 1-3 of [`TASKS.md`](../TASKS.md). Each is
self-contained and touches a mostly disjoint set of files, so they can run in any order —
or in parallel, if each agent works on its own branch and rebases before merging.

Items 4 and 5, and items 6-8, the sequence-modelling chain, are deliberately not
covered here: 4 is a decision rather than a package, 5 postdates the prompts, and
the chain is a dependency chain rather than independent packages — future work
rather than queued work. They would want their own prompts, written when they are
actually scheduled.

Copy one prompt verbatim into a fresh agent session.

| Prompt | TASKS.md items | Rough size |
| --- | --- | --- |
| ~~A — Packaging and process~~ | — | **done** |
| B — Testing, layers, performance | 1, 2, 3 | large |
| ~~C — A second example domain~~ | — | **done** |
| ~~D — Graph visualizer~~ | — | **done** |
| ~~E — Reclaim the tape~~ | — | **done** |

Prompts A, C, D, and E are finished and their sections removed. A produced
`CONTRIBUTING.md`, `CHANGELOG.md`, the annotated `v0.1.0` tag, `pynn/py.typed`,
`pynn.__version__`, the exact ruff and mypy pins, and `.pre-commit-config.yaml`;
C produced
`examples/char_rnn.ipynb` and `scripts/download_shakespeare.py`, executed and committed
with its outputs, and needed no library code to do it; D produced `pynn/viz.py`,
`Tensor.to_dot`, and the tape figure the README now opens with; E produced
`Tensor.free_graph()`, `benchmarks/memory.py`, and a char-RNN notebook that no longer
calls `gc.collect()`. The remaining letters are left as they were rather than shifted
up, so a prompt already in flight still means what it said.

E is the only prompt here that came out of another prompt's work rather than out of a
review pass: C turned up the reference cycle on its way through writing the notebook,
and that finding took the `TASKS.md` slot the example domain had vacated. **B is the
only package left.**

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
> **State of the repo.** Green on Python 3.10–3.14: 807 tests, 341 checks from
> `python -m pynn.verify` (209 of them numerical gradient checks), 98.4% line coverage
> with a 95% floor enforced in CI. `ruff` and `mypy` are clean across `pynn tests
> examples scripts benchmarks`, including the code cells of both notebooks.
> `pre-commit run --all-files` is clean too, and `ruff` and `mypy` are pinned exactly —
> if you bump one, bump `.pre-commit-config.yaml` in the same commit.
>
> **Environment.** Use `.venv/bin/python` (Python 3.14) — a bare `pytest` on `PATH`
> resolves to a Homebrew install without `pytest-cov`. `pynn` is installed editable.
>
> **Before every commit, all of these must pass:**
>
> ```bash
> .venv/bin/ruff check pynn tests examples scripts benchmarks
> .venv/bin/ruff format --check pynn tests examples scripts benchmarks
> .venv/bin/mypy
> .venv/bin/python -m pytest -m "not external" --cov=pynn
> .venv/bin/python -m pynn.verify
> ```
>
> If you touch typing, the CLI, NumPy usage, or anything version-sensitive, also run
> `scripts/ci_matrix.sh`, which runs every CI step against all five supported Pythons in
> throwaway virtualenvs. Three CI failures have already come from behaviour the newest
> interpreter cannot reproduce.
>
> **House style.**
>
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
> - Update `TASKS.md` to mark what you finished, add a `## [Unreleased]` entry to
>   `CHANGELOG.md`, and update `README.md` / `USAGE.md` / `docs/DESIGN.md` if you
>   changed anything they claim. `CONTRIBUTING.md` §7 lists what to check.
>
> Report back with: what shipped, commit SHAs, the test/verify/coverage numbers, and
> anything you deliberately left out and why.

---

## Prompt B — Testing, layers, and performance

**TASKS.md items 1, 2, 3.** The largest package. Item 3 is measurement-driven and has
numbers already recorded in `TASKS.md` — respect them.

```text
[paste the shared preamble here]

Your job is TASKS.md items 1, 2, and 3. Read those entries first — item 3 in particular
already contains measurements you should not re-derive from scratch, only extend.

1. Property-based tests over unbroadcast.
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

2. More layers, losses, and optimizers.
   Pick from the table in TASKS.md item 2. Do NOT do all of it — choose what is coherent
   and finish it properly rather than half-landing six things. Suggested slice, in order
   of value:
     - Unflatten / Reshape as the inverse of Flatten, and Identity as a layer. Small, and
       they complete an obvious gap.
     - KLDivLoss and HingeLoss, using the shared _reduce helper in
       pynn/functional/losses.py so the three reduction modes stay consistent.
     - NAdam, ReduceLROnPlateau, OneCycleLR.
   Every new op needs a gradcheck entry with a reused-input variant. Every new optimizer
   needs a closed-form reference transcription in tests/optim/optimizers_test.py — "the
   loss went down" does not distinguish a correct update rule from a nearly-correct one.

3. Further acceleration — measurement first, and the first win needs no dependency.
   TASKS.md item 3 records that SGD.update is 32% of an MLP training step, almost all of
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
- Item 3 must not change any optimizer's arithmetic. If a reference test needs updating,
  you have changed behaviour — stop and flag it.
- Do not add a dependency without a measurement justifying it.
```

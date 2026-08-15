# Agent prompts

Independent work packages covering items 1-4 of [`TASKS.md`](../TASKS.md). Each is
self-contained and touches a mostly disjoint set of files, so they can run in any order —
or in parallel, if each agent works on its own branch and rebases before merging.

Items 5-7, the sequence-modelling chain, are deliberately not covered here: they are a
dependency chain rather than independent packages, and they are future work rather than
queued work. They would want their own prompts, written when they are actually scheduled.

Copy one prompt verbatim into a fresh agent session.

| Prompt | TASKS.md items | Rough size |
| --- | --- | --- |
| ~~A — Packaging and process~~ | — | **done** |
| B — Testing, layers, performance | 1, 2, 3 | large |
| ~~C — A second example domain~~ | — | **done** |
| ~~D — Graph visualizer~~ | — | **done** |
| E — Reclaim the tape | 4 | small, but it is an API decision |

Prompts A, C, and D are finished and their sections removed. A produced `CONTRIBUTING.md`,
`CHANGELOG.md`, the annotated `v0.1.0` tag, `pynn/py.typed`, `pynn.__version__`, the
exact ruff and mypy pins, and `.pre-commit-config.yaml`; C produced
`examples/char_rnn.ipynb` and `scripts/download_shakespeare.py`, executed and committed
with its outputs, and needed no library code to do it; D produced `pynn/viz.py`,
`Tensor.to_dot`, and the tape figure the README now opens with. The remaining letters are
left as they were rather than shifted up, so a prompt already in flight still means what
it said.

Item 4 of `TASKS.md` is no longer the item prompt C was written against. C turned up a
measured resource problem on its way through — a finished graph is a reference cycle —
and that finding took the slot the example domain vacated. **Prompt E is the new item 4**,
and it is the only prompt here that came out of another prompt's work rather than out of
a review pass.

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

---

## Prompt E — Reclaim the tape

**TASKS.md item 4.** The smallest package here and the only one that is already costing
something: the workaround is in the repository, in `examples/char_rnn.ipynb`'s training
loop. The patch is a few lines; choosing which patch is the work.

```text
[paste the shared preamble here]

Your job is TASKS.md item 4: reclaim the tape without waiting for the cyclic collector.
Read that entry first. It carries measurements taken over 400 steps with a fresh process
per row — do not re-derive them, and do not replace them with a shorter run.

The problem. Every Tensor holds its reverse pass as a closure that references the Tensor
it belongs to, so a finished graph is a reference cycle. Dropping the last name pointing
at the loss frees nothing; only CPython's cyclic collector can, and it decides when to run
a full collection from how much the object count has grown, which bears no relation to the
hundreds of megabytes of NumPy arrays hanging off those objects. A feedforward model never
notices — its graph is a few dozen nodes. Anything that unrolls a recurrence does:
examples/char_rnn.ipynb calls gc.collect() on a cadence inside its training loop for this
reason, and an earlier draft of it, at batch 64, was killed by the OS at 7 GB.

The decision is the work. TASKS.md item 4 lists three options in increasing order of
commitment: document it (where it stands today), a Tensor.free_graph() the caller invokes
after backward, or backward(retain_graph=False) as the default, which is how PyTorch
spells the same trade. Pick one and defend it in the commit body. free_graph() is the
recommendation: it changes no default behaviour, so nothing that works today stops
working, and it is the option that can still become a default later — the reverse is not
true. Whichever you pick, breaking the cycle means clearing a node's `reverse` and
`children` as the traversal passes it, over the topological order backward already walks.

What has to come with it:
- An invariant in pynn/verify/invariants.py: after whichever call frees the graph, the
  loss's children are empty and a second backward raises rather than silently computing
  something wrong. That is the house rule about failing loudly, and it matters more here
  than usual: a freed graph that quietly returns zeros looks like a converged model.
- A test that the freed path produces the same gradients as the unfreed one on a graph
  with a reused input. Freeing mid-traversal must not free a node that another node still
  needs, and the single-consumer case cannot see that.
- Before-and-after peak RSS for the notebook's configuration, measured the way TASKS.md
  item 4 was measured.
- examples/char_rnn.ipynb updated to use it. If the fix removes the need for the cadence,
  the gc.collect() and the paragraph explaining it both go, and the notebook is
  re-executed so its committed outputs match its committed source — budget ~4 minutes
  for that, and expect the wall-clock figure in its Summary table to move. If the cadence
  still helps on top of the fix, say by how much rather than leaving both in silently.
- USAGE.md section 5 documents the workaround today, with numbers. It has to say whatever
  is true afterwards.

Traps, every one of them met while measuring this:
- Measure over at least 400 steps. The same benchmark over 60 steps reports the opposite
  conclusion — 82.3 ms/step uncollected against 88.0 with gc.collect() every 4, i.e.
  that collecting costs 7% and should be deleted. The effect only appears once enough
  garbage has accumulated to make allocation expensive.
- Fresh process per configuration, and read RSS from `ps -o rss= -p <pid>`.
  resource.getrusage(...).ru_maxrss reported near-identical peaks for configurations whose
  true peaks differed by more than 2 GB.
- Nothing in tests/ or pynn/verify depends on re-running a graph. That was checked rather
  than assumed, and item 4 names the one test that calls backward() twice and why it does
  not count. Re-check it rather than trusting a note, but do not expect it to block you.

Constraints:
- Tensor.backward is an explicit-stack traversal so that a deeply unrolled graph does not
  hit the recursion limit. Do not turn it back into recursion while restructuring it.
- No gradient changes. python -m pynn.verify must pass unchanged, all 209 gradient cases
  included — this is a memory fix, and a memory fix that moves a number is a bug.
```

---

## Running these in parallel

The two remaining packages barely overlap, but the places they do are worth knowing:

- **B and E both touch `TASKS.md`** and `CHANGELOG.md`. Prose, so they resolve by hand.
- **B's item 3 touches every optimizer**; E touches `pynn/core/tensor.py`,
  `pynn/verify/invariants.py`, `examples/char_rnn.ipynb`, and `USAGE.md`. Nothing is
  shared.
- `READ_FILES.md` is generated, so never merge it — regenerate after merging with
  `.venv/bin/python scripts/generate_read_files.py`.

Suggested order if running sequentially: **E** first. It is much the smaller package, it
is the one item in `TASKS.md` already costing something, and B's item 3 re-profiles a
training step — better done after the tape's memory behaviour has settled than before.

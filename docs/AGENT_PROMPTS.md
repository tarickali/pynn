# Agent prompts

A record of the work packages handed to agents, what each produced, and what each turned
up that nobody had asked for. **All five are done**; the sections below are kept for the
shared preamble, which is still the right starting point for a new one, and for the
"what X left behind" notes, which are the most useful part of the file.

Items 4 and 5, and items 6-8, the sequence-modelling chain, were deliberately never
covered: 4 is a decision rather than a package, 5 postdates the prompts, and the chain is
a dependency chain rather than independent packages — future work rather than queued work.
Item 3a postdates them too, and is a measurement problem rather than a feature.

To hand out new work, copy the shared preamble verbatim and write the brief underneath it.

| Prompt | TASKS.md items | Rough size |
| --- | --- | --- |
| ~~A — Packaging and process~~ | — | **done** |
| ~~B — Testing, layers, performance~~ | 1, 2, 3 | **done** |
| ~~C — A second example domain~~ | — | **done** |
| ~~D — Graph visualizer~~ | — | **done** |
| ~~E — Reclaim the tape~~ | — | **done** |

**All five are finished and their sections removed.** A produced `CONTRIBUTING.md`,
`CHANGELOG.md`, the annotated `v0.1.0` tag, `pynn/py.typed`, `pynn.__version__`, the
exact ruff and mypy pins, and `.pre-commit-config.yaml`; B produced the property tests,
`Unflatten` / `Identity` / `KLDivLoss` / `HingeLoss` / `NAdam` / `OneCycleLR` /
`ReduceLROnPlateau`, and the in-place optimizer rewrite — see below for what else;
C produced `examples/char_rnn.ipynb` and `scripts/download_shakespeare.py`, executed and
committed with its outputs, and needed no library code to do it; D produced `pynn/viz.py`,
`Tensor.to_dot`, and the tape figure the README now opens with; E produced
`Tensor.free_graph()`, `benchmarks/memory.py`, and a char-RNN notebook that no longer
calls `gc.collect()`. The letters are left as they were rather than compacted, so a
prompt already in flight still means what it said.

E and B both came out of other work rather than out of a review pass: C turned up the
reference cycle on its way through writing the notebook, and B turned up items 1a and 3a
on its way through items 1 and 3. **There is no package left to hand out.** What remains
in `TASKS.md` — items 1a, 3a, the rest of 2, 4, 5, and the sequence-modelling chain 6-8 —
has no prompt written for it, and the note at the top of this file about why still holds:
4 is a decision rather than a package, 5 is mechanical, 3a is a measurement problem
rather than a feature, and 6-8 are a dependency chain. Anything new wants its own prompt,
written when it is actually scheduled.

### If you are writing the next one

Three things about prompt B are worth carrying forward, and none of them are about its
subject matter.

- **It gave a number to respect and the number was wrong.** "An in-place NumPy rewrite
  measured 1.8-2.5x" was an isolated microbenchmark, and the prompt said to extend those
  measurements rather than re-derive them — which is right for saving effort and wrong
  when the recorded number is the thing that needs checking. A prompt that quotes a
  measurement should say *how it was taken*, so the agent can tell whether the method
  survives the new question being asked of it.
- **"Report what you left out and why" did most of the work.** Nearly everything in
  `TASKS.md`'s new **Decisions** section came from having to write that paragraph. It is
  cheaper than a review pass and catches a different class of thing.
- **The scope instruction held.** "Do NOT do all of it — choose what is coherent and
  finish it properly rather than half-landing six things" is the line that kept item 2
  from becoming eight shallow additions, and it is worth reusing verbatim.

### What E left behind

Worth recording, because it is a pattern rather than an accident: **E's brief was one
paragraph of `TASKS.md` and it closed three items' worth of ground.** The prompt asked
for a memory fix. Reviewing what the fix now refuses, and then auditing where the
library diverges from what a PyTorch user would expect, turned up more than the fix
itself did.

Shipped beyond the brief:

- **`Tensor.__setitem__` refuses a Tensor an operation produced.** In-place assignment
  was documented as non-differentiable, but the guard was a docstring, and a reverse
  closure that reads its inputs' `data` when it runs takes the gradient at values the
  forward pass never saw. Loud now, with the residual hole stated in the docstring,
  `docs/DESIGN.md` §14, and `CHANGELOG.md` rather than left to be discovered.
- **`docs/DESIGN.md` §11's check counts**, which had said 151 and "nine graph
  topologies" since before the recurrent cells landed. 209 and eight.

Queued rather than built, each with its measurements in `TASKS.md`:

- **Item 4** — a second `backward` over one graph compounds instead of doubling, and
  the overshoot grows with depth. Found by asking what the call `free_graph` refuses
  would have done. Pinned by a test that asserts the wrong numbers so they cannot drift
  while the decision is open.
- **Item 5** — a float32 path. The tape already preserves float32; the initializers and
  scalar promotion do not.
- **Item 2's `index_update`** and its companion guard, from following the `__setitem__`
  question to the case the guard does not catch.

The lesson for whoever writes the next prompt: **the brief is a floor.** Three of these
came from asking what a change makes *newly* possible to get wrong, which is a question
worth asking on purpose rather than stumbling into.

### What B left behind

The same pattern again, and the interesting half was the measurement rather than the code.

Shipped beyond the brief:

- **Two bugs in existing code**, both found sideways. The pooling gradient checks passed
  at `DEFAULT_SEED` and failed at three of six other seeds — inserting new cases earlier
  in `gradient_cases` shifted every later case's random draw and exposed it. And
  `ReduceLROnPlateau`'s relative threshold, transcribed from PyTorch as
  `best * (1 - threshold)`, means a *negative* metric improves by getting worse; that one
  came from reviewing the new code rather than from a test.
- **A discrepancy in PyTorch worth knowing about.** `torch.optim.NAdam` keeps
  `mu_product` and `step` as float32 tensors regardless of the parameter's dtype, so on a
  float64 parameter it disagrees with the published rule by ~1e-10. The reference
  transcription matches exactly and the external comparison's tolerance is sized for
  torch's error, with a docstring saying so — a tolerance nobody can explain is how a
  real disagreement gets absorbed later.
- **`docs/DESIGN.md` §15 gained a second half**, on the thing the profile named after
  `col2im`, and then had to be rewritten once the measurement fell apart.

Retracted rather than shipped, which is the part worth reading:

- **"3.0x isolated becomes 1.6x in a real training loop" was wrong**, and it was in the
  README, `TASKS.md`, `docs/DESIGN.md` and `CHANGELOG.md` before it was caught. It came
  from a paired in-loop timer that has a **1.5x ordering bias** — whichever side of the
  pair runs first wins. The bias was found by filling in the rows the first pass had
  skipped: three optimizers came out *slower* after removing thirteen allocations, which
  an independent size sweep says is impossible. The end-to-end whole-step number went the
  same way: ~0.1–0.2 ms of a ~3.2 ms step against a ±0.4 ms spread.
- What replaced it is better: **the speedup is a function of parameter size**, 1.2x on a
  256-element bias to 3.1x on a 200k-element weight matrix, which is the actual
  explanation for why the MLP step barely moves — four of its six parameters are tiny.

Queued rather than built, each with its reasoning in `TASKS.md`:

- **Item 1a** — the rest of the property tests. The strategies exist now, so the
  expensive part is paid for.
- **Item 3a** — a harness that can measure an end-to-end speedup, which this repository
  does not have. It **blocks** the two remaining acceleration candidates, since their
  3.3–3.7x and 7.1–7.5x were taken by the method that just failed.
- **A `Decisions` section in `TASKS.md`** for the five things deliberately not built, so
  they are not re-proposed as oversights.

The lesson this time: **check how a recorded measurement was taken before extending it.**
The prompt said to respect the numbers in `TASKS.md` and not re-derive them, which was
the right instruction for the wrong number.

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
> **State of the repo.** Green on Python 3.10–3.14: 982 tests, 415 checks from
> `python -m pynn.verify` (226 of them numerical gradient checks), 98.5% line coverage
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

## No open prompts

Every letter above is finished. `TASKS.md` is the live queue; nothing in it currently has
a prompt written for it, and two of its entries would want a different *shape* of prompt
than the five above:

- **Item 3a** is a measurement problem, not a feature. A brief for it should say what a
  usable harness has to prove — that its own ordering bias is smaller than the effect it
  reports — rather than what to build, and it should treat "no measurable difference" as
  an acceptable finding rather than a failed task. Prompt B's experience is the reason:
  its brief assumed the end-to-end number was there to be found.
- **Item 4** is a decision with three options already measured and written up. It wants a
  reviewer, not an implementer.

The rest — items 1a, 2's remainder, 5, and the 6-8 chain — are ordinary packages and
would follow the shape of A through E: the shared preamble, a numbered brief, and a
constraints list that says what must *not* change.

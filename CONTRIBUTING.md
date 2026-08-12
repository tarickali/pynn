# Contributing

Almost every change to this library is the same shape: *a new differentiable
operation*. A layer, an activation, a loss, a normalization — they all reduce to a
function that computes something and a closure that pushes a gradient back through it.
So this file is one walkthrough of that, end to end, rather than a list of rules.

The parts that are not obvious are the last two steps. A new operation is not finished
when it produces the right numbers; it is finished when the gradient sweep knows about
it, and when any behaviour a gradient check cannot see has an invariant of its own.
[Section 5](#5-the-gradcheck-entry-and-why-it-is-not-optional) is the one to read even
if you skip the rest.

- [Before you start](#before-you-start)
- [The rule that decides most of the bugs](#the-rule-that-decides-most-of-the-bugs)
- [Adding a layer, end to end](#adding-a-layer-end-to-end)
  - [Where the pieces go](#where-the-pieces-go)
  - [1. The functional op](#1-the-functional-op)
  - [2. The Module](#2-the-module)
  - [3. Export it](#3-export-it)
  - [4. Hand-written tests](#4-hand-written-tests)
  - [5. The gradcheck entry, and why it is not optional](#5-the-gradcheck-entry-and-why-it-is-not-optional)
  - [6. An invariant, when a gradient check cannot see it](#6-an-invariant-when-a-gradient-check-cannot-see-it)
  - [7. The docs that claim things](#7-the-docs-that-claim-things)
- [Two files worth reading first](#two-files-worth-reading-first)
- [Checklist](#checklist)
- [Commits](#commits)
- [House style](#house-style)

---

## Before you start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

Everything below assumes the virtualenv is active. [`USAGE.md`](USAGE.md) documents
every command this project supports and what a healthy result looks like.

These five must pass before a commit. They are what CI runs, in the order CI runs them:

```bash
ruff check pynn tests examples scripts benchmarks
ruff format --check pynn tests examples scripts benchmarks
mypy
pytest -m "not external" --cov=pynn
python -m pynn.verify
```

`pre-commit install` wires the first three into a git hook so they run on the way in.
The last two are yours to run.

If you touched typing, the CLI, or NumPy usage, run `scripts/ci_matrix.sh` as well. It
repeats every step above against all five supported interpreters in throwaway
virtualenvs. Three CI failures so far have come from behaviour the newest interpreter
cannot reproduce, and none of them were visible locally.

---

## The rule that decides most of the bugs

A reverse pass **accumulates into** `grad`. It never assigns to it.

```python
x.grad += something * output.grad     # yes
x.grad = something * output.grad      # no
```

A tensor consumed by more than one downstream operation gets one gradient contribution
per consumer, and they must be summed. Assigning keeps whichever ran last.

This survives review because it is invisible in a plain feedforward network: an
unbranched chain gives every tensor exactly one consumer, so overwriting and
accumulating agree *exactly*. It surfaces with a residual connection, a tied weight, an
auxiliary loss, a regularization term, or a recurrent cell unrolled over time — which
is to say, the moment the model gets interesting. Everything in
[section 5](#5-the-gradcheck-entry-and-why-it-is-not-optional) is downstream of this
one fact.

---

## Adding a layer, end to end

The worked example is **`RMSNorm`** — root-mean-square layer normalization, which
divides each example by the root mean square of its trailing axes and scales the result
by a learnable `gamma`. It is not in the library. It is used here because it is small,
because it is a real layer people ask for, and because its reverse pass has to be
written out by hand for the same reason `layer_norm`'s does.

Follow along by actually adding it, or read it and go do your own.

### Where the pieces go

Every operation exists twice, and the split is deliberate: the function is the thing
that is differentiable, and the Module is the thing that owns parameters and state.

| File | Holds |
| --- | --- |
| `pynn/functional/modules.py` | The op. Takes and returns `Tensor`s, holds no state. |
| `pynn/nn/modules.py` | The `Module`. Owns parameters, `training`, buffers. |
| `pynn/nn/__init__.py` | The export. |
| `pynn/verify/gradients.py` | The gradcheck entries. **Not optional** — section 5. |
| `pynn/verify/invariants.py` | Behaviour a gradient check cannot see — section 6. |
| `tests/nn/layers_test.py` | Shapes, errors, and anything with a closed form. |

An activation goes in `pynn/functional/activations.py` and `pynn/nn/activations.py`
instead; a loss in `pynn/functional/losses.py` and `pynn/nn/losses.py`. The shape of the
work is identical. Layers and their functional ops are tested together in
`tests/nn/layers_test.py`; activations, losses, and initializers have their own files
under `tests/functional/`.

### 1. The functional op

An operation is a plain function. There is no `Op` class to subclass and nothing to
register — the tape is closures, so the only contract is: build the output `Tensor`,
declare what it was computed from, and attach a closure that pushes the gradient back.

```python
def rms_norm(
    x: Tensor,
    gamma: Tensor | None = None,
    normalized_shape: Shape | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if normalized_shape is None:
        normalized_shape = (x.shape[-1],)
    axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))

    inverse_rms = 1.0 / np.sqrt((x.data**2).mean(axis=axes, keepdims=True) + eps)
    normalized = x.data * inverse_rms
    scale = None if gamma is None else gamma.data

    output = Tensor(normalized if scale is None else normalized * scale)
    output.add_children((x,) if gamma is None else (x, gamma))

    def reverse() -> None:
        upstream = output.grad
        if gamma is not None:
            # unbroadcast, because gamma is (features,) against an (..., features)
            # output: the copies NumPy made along the leading axes each contributed.
            gamma.grad += unbroadcast(upstream * normalized, gamma.shape)

        d_normalized = upstream if scale is None else upstream * scale
        # The subtracted term is the path through the statistic. Every element of x
        # moves the root mean square, so the normalized output of *each* element
        # depends on *all* of them; dropping it leaves a gradient that still descends.
        x.grad += (
            d_normalized
            - normalized * (d_normalized * normalized).mean(axis=axes, keepdims=True)
        ) * inverse_rms

    output.forward = "rms_norm"
    output.reverse = reverse
    return output
```

Five things in there are load-bearing.

**`x.data`, not `x`.** Inside the op you are working with NumPy arrays. Arithmetic on
`Tensor`s would record a second graph for the same computation.

**`add_children` declares the edges.** It is also the gate that `no_grad` closes: with
recording off it drops the edges and marks the output as not requiring gradients, so
the op itself never has to check the mode. The `reverse` setter is the second gate — it
discards the closure instead of storing it, which is what actually frees the memory a
closure would otherwise hold.

**The closure is the save list.** `inverse_rms` and `normalized` are captured because
the backward pass needs them. Nothing has to be declared as "saved for backward";
Python's closure capture already is that.

**`unbroadcast` for any operand that broadcast.** A `(features,)` parameter against an
`(N, features)` output was implicitly copied `N` times, and the derivative of a copy is
a sum. Skip it and the shapes will not even match; get it subtly wrong and the
parameter's gradient is off by a factor of the batch size, which trains, just worse.

**The reverse is written out rather than composed.** `rms_norm` could be built from
`pmath.mean`, `**`, `sqrt`, and `*`, and autodiff would handle it. It is not, because
the statistic depends on every element being normalized, so the composed graph
re-derives that dependency once per element instead of once. Same reasoning as
`layer_norm` and `batch_norm`. Compose when the naive graph is fine; write it out when
it is not, and say which in a comment.

### 2. The Module

Three things: `forward`, `build`, and `hyperparameters`. `Module` is an ABC, so
`forward` and `hyperparameters` are required and `build` defaults to doing nothing.

```python
class RMSNorm(Module):
    """Normalize each example by the root mean square of its trailing axes.

    Layer normalization without the mean subtraction: it rescales but does not
    recenter, which is one statistic per example instead of two, and is what most
    recent transformer implementations use.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | None = None,
        eps: float = 1e-5,
        name: str = "RMSNorm",
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: tuple[int, ...] | None = normalized_shape
        self.eps = eps
        self.name = name

    def build(self, input_shape: Shape) -> None:
        if self.normalized_shape is None:
            self.normalized_shape = (input_shape[-1],)
        self.register_parameter("gamma", np.ones(self.normalized_shape))
        self.initialized = True

    def forward(self, X: Tensor) -> Tensor:
        if not self.initialized:
            self.build(X.shape)
        return rms_norm(X, self.parameters["gamma"], self.normalized_shape, self.eps)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"normalized_shape": self.normalized_shape, "eps": self.eps}
```

**`super().__init__()` first, always.** It creates `_modules`, which `__setattr__`
consults on every subsequent assignment. Assigning a child module before it exists
raises rather than silently failing to register.

**`build` is called on the first forward pass**, not in `__init__`, so a layer can
infer its shape from real data — `RMSNorm()` with no arguments takes the last axis of
whatever it first sees. Guard it with `self.initialized` and set that flag at the end.
This is also why optimizers hold *live* references to `parameters` dictionaries rather
than copies: an optimizer built before the first forward pass is looking at
dictionaries that are still empty, and the parameters simply appear.

**`register_parameter`, not `self.parameters[...] = ...`.** It propagates the module's
frozen state to the new parameter. A module frozen before its first forward pass has no
parameters to mark yet, and would otherwise come back trainable once built. Use
`register_buffer` for state the module owns but no optimizer should ever step —
`BatchNorm`'s running statistics are the only ones in the library today.

**`hyperparameters` is what `summary()` reports.** Everything needed to reconstruct the
layer, nothing that is a parameter.

**Assigning a plain list of Modules to an attribute raises.** Use `ModuleList` or
`ModuleDict`. The alternative is layers that run in the forward pass, receive gradients,
and are never handed to an optimizer.

### 3. Export it

Add the function to `__all__` in `pynn/functional/modules.py`, and the class to the
imports and `__all__` in `pynn/nn/modules.py` and `pynn/nn/__init__.py`. `__all__` is
sorted, and ruff will tell you if it is not.

### 4. Hand-written tests

`tests/nn/layers_test.py` holds the layer and its functional op together — shapes, the
errors it raises, the flags it honours, and any property with a closed form. `rng` is a
seeded fixture from `tests/conftest.py`; use it rather than an unseeded generator, since
a test that fails once and passes on re-run tells you nothing.

```python
def test_rms_norm_rescales_each_row_to_unit_rms(rng) -> None:
    x = Tensor(rng.standard_normal((6, 5)) * 10.0)
    rms = np.sqrt((rms_norm(x).data ** 2).mean(axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-4)
```

These are the tests that say *what the layer is*. The next section is the one that says
its gradient is right, and the two are not interchangeable.

### 5. The gradcheck entry, and why it is not optional

Add the new op to `gradient_cases` in `pynn/verify/gradients.py`. At minimum: the op
itself, and the op with its input reused.

```python
def _rms_norm_case(normalized_shape: tuple[int, ...]) -> ScalarFn:
    """``sum(rms_norm(x, gamma) * c)`` over one set of trailing axes."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(rms_norm(ts[0], ts[1], normalized_shape) * ts[2])

    return scalar
```

```python
    for label, shape, normalized in [
        ("rms_norm (4, 5)", (4, 5), (5,)),
        ("rms_norm over two axes", (3, 4, 5), (4, 5)),
    ]:
        cases.append(
            (
                label,
                _rms_norm_case(normalized),
                [
                    normal(*shape),
                    Tensor(rng.uniform(0.5, 1.5, normalized)),
                    normal(*shape),
                ],
            )
        )
    cases += [
        (
            "rms_norm without gamma",
            lambda ts: pmath.sum(rms_norm(ts[0]) * ts[1]),
            [normal(4, 5), normal(4, 5)],
        ),
        (
            "rms_norm with a reused input",
            lambda ts: pmath.sum(rms_norm(ts[0]) * ts[1]) + pmath.sum(ts[0] * ts[2]),
            [normal(4, 5), normal(4, 5), normal(4, 5)],
        ),
    ]
```

Each case is a name, a function from a list of `Tensor`s to a **scalar** `Tensor`, and
the inputs to differentiate. The checker perturbs each input element by `±eps`,
compares `(f(x + eps) - f(x - eps)) / 2 eps` against what the reverse pass produced,
and reports the worst-disagreeing element by index.

The factory function is not decoration. A `lambda` written inside the loop would close
over `normalized` by reference, so every case built in that loop would end up using the
loop's *final* value — and each would still pass, because a wrong-but-consistent shape
is differentiated consistently. Any case parameterized by a loop variable gets a
factory; the ones above that are not use a `lambda` directly.

#### Why this and not just a test

A hand-written gradient test asserts the derivative *you believed in when you wrote the
op*. If the algebra was wrong, the test is usually wrong the same way — you derive an
expression, implement it, and then write an assertion from the same derivation. Central
differences do not consult your derivation. They ask the forward pass what it actually
does.

One entry then buys three things a local test does not:

1. **A named pytest case.** `tests/test_gradcheck.py` parametrizes over
   `gradient_cases()`, so a failure reads `test_gradient_case[rms_norm-(4,5)]` rather
   than one failure for a 200-case sweep.
2. **A check in `python -m pynn.verify`.** That subpackage ships with the library and
   runs against an *installed* copy with no pytest and no test files present. A
   gradient checked only in `tests/` is checked in this repository; a gradient checked
   in the sweep is checked wherever the library is installed.
3. **Protection from a silent shrink.** `test_sweep_covers_every_operation` asserts a
   lower bound on the number of cases and on the number of reuse cases, so a refactor
   that drops a block of them fails rather than quietly reducing coverage.

#### Why the reused-input variant specifically

Because the single-consumer form proves nothing about the one bug that matters.

Take the `rms_norm` above and change its one accumulating line to an assignment —
`x.grad = (...)` instead of `x.grad += (...)`. The measured result:

```
rms_norm (4, 5)                    PASS  max rel err 3.22e-09
rms_norm over two axes             PASS  max rel err 1.57e-08
rms_norm without gamma             PASS  max rel err 8.81e-09
rms_norm with a reused input       FAIL  max rel err 1.00e+00
```

Three checks of the same broken function, all green to nine digits. With one consumer
there is nothing to accumulate *onto*, so overwriting and accumulating produce
identical arrays. The reuse case — the same tensor feeding both the op and a second
term — is the smallest graph where they differ, and it fails by a relative error of
1.0.

So: every unary and binary op in the sweep appears twice, once alone and once with its
input reused. Yours should too.

#### Choosing inputs

The checker differences a real function, so the inputs have to sit where that function
is actually differentiable and numerically well behaved.

| Situation | What the sweep does | Why |
| --- | --- | --- |
| Kinks (`relu`, `abs`, `elu`) | `away_from_zero()` | A probe that steps across the kink measures a derivative that does not exist there. |
| `max_pool2d` | `separated()` — a shuffled range | Values far enough apart that `±eps` cannot change which element wins a window. |
| Randomness (`dropout`) | a fixed `rng=0` | Every probe must see the same mask, or the function differs between the two evaluations. |
| Large magnitudes | a larger `eps` | Differencing two nearby large values loses significant digits. |
| Logs and divisions | `positive()` | Keeps the input off the singularity. |

If a case fails at `1e-8` but passes at `1e-4`, suspect the step size before suspecting
the gradient. If it fails at `1.0`, it is the gradient.

### 6. An invariant, when a gradient check cannot see it

`pynn/verify/invariants.py` holds the properties that are true of the library's
*behaviour* rather than of its derivatives — the ones that break without anything going
red, because the library keeps running and training keeps roughly working.

A gradient check is blind to, among other things:

- **A train/eval difference.** `Dropout` in eval mode must be exactly the identity. A
  model left in training mode returns a different answer every call and nothing raises.
- **A buffer.** `BatchNorm`'s running statistics must update while training, must be
  used instead of the batch's at evaluation, and must survive a `state_dict` round
  trip. A model that loses them evaluates differently after a checkpoint.
- **Initializer scale.** `he_normal` once computed fan-in as `shape[0]`, which is the
  *output* channel count for a conv kernel. Every gradient check passed. What it did
  was make a small CNN collapse into predicting a constant.
- **An optimizer's update rule.** A momentum buffer with a dropped term still descends,
  just more slowly. Each optimizer is therefore compared against a closed-form
  transcription of its published rule, never against "the loss went down".

`RMSNorm` has none of these — no mode, no buffer, no state, and its scale is fixed by
construction — so it needs no invariant. That is the answer often enough that the
question is worth asking explicitly rather than defaulting either way: **what does this
layer do that a number cannot disagree with?** If there is an answer, it goes here.

An invariant is one line:

```python
report.add(
    "dropout is deterministic at evaluation",
    bool(np.array_equal(stochastic(X).data, stochastic(X).data)),
)
```

### 7. The docs that claim things

Grep for whatever your change makes untrue.

- `README.md` — the feature list, the API table, and the check counts it quotes.
- `USAGE.md` — expected output, including the test and verification counts.
- `docs/DESIGN.md` — if the change involved a decision worth explaining rather than
  reading. A fused reverse pass is one; a fourth activation is not.
- `CHANGELOG.md` — under `## [Unreleased]`, in the Keep a Changelog categories.
- `TASKS.md` — if you finished something it lists, remove it and record it under
  *Done, for reference*.

Then, after committing, refresh the generated history:

```bash
python scripts/generate_read_files.py    # rewrites READ_FILES.md
```

Commit that on its own. It is generated — never edit it by hand.

---

## Two files worth reading first

**`pynn/nn/modules.py::Dropout`** is the smallest complete example in the library: a
Module with no `build` and no parameters, a functional op whose reverse is a single
line, two gradcheck entries, and two invariants. It is the whole of this document at a
size you can hold in your head. It is also the canonical reason modes exist at all, so
read `pynn/functional/modules.py::dropout` beside it and note that `training=False`
returns the input *unchanged* rather than scaling it — inverted dropout divides the
survivors during training, so evaluation needs no compensating factor.

**`pynn/functional/modules.py::layer_norm`** is the other end: a hand-written fused
reverse, sharing `_normalize` with `batch_norm`. Read it for the shape of the argument
about when to write a reverse out instead of composing one, and for how
`differentiate_statistics` splits two genuinely different functions — batch norm at
evaluation normalizes by fixed running estimates, which makes it an affine map of its
input and its gradient a different expression entirely.

[`docs/DESIGN.md`](docs/DESIGN.md) covers why the library is built this way;
[`READ_FILES.md`](READ_FILES.md) has a suggested reading order for the source.

---

## Checklist

- [ ] Functional op in `pynn/functional/`, with a `reverse` that uses `+=`
- [ ] `Module` in `pynn/nn/`, with `forward`, `build` if it has parameters, and
      `hyperparameters`
- [ ] Exported from `__all__` and from `pynn/nn/__init__.py`
- [ ] Hand-written tests for shapes, errors, and closed forms
- [ ] **Gradcheck entries in `gradient_cases`, including a reused-input variant**
- [ ] An invariant, if the layer does anything a gradient check cannot see
- [ ] NumPy-style docstrings; comments explain *why*, not *what*
- [ ] `README.md` / `USAGE.md` / `docs/DESIGN.md` still true; `CHANGELOG.md` updated
- [ ] All five commands green
- [ ] `scripts/ci_matrix.sh` too, if you touched typing, the CLI, or NumPy usage

---

## Commits

Short imperative subject, no trailing period. The body explains **why**, since the diff
already covers what. One concern per commit; a refactor and a fix are two commits.

```
add RMSNorm

Layer normalization without the mean subtraction: one statistic per
example rather than two. The reverse pass is written out rather than
composed for the same reason layer_norm's is — the statistic depends on
every element, so the composed graph re-derives that dependency once per
element.
```

Only commit when all five checks are green. Do not commit generated files alongside
source changes: `READ_FILES.md` gets its own commit, after.

---

## House style

- **NumPy-style docstrings**, on anything public.
- **Comments explain why, and what breaks otherwise.** Never what the line does. If a
  comment restates the code, delete it; if it names the failure the line prevents, keep
  it.
- **Fail loudly.** The library refuses a plain list of Modules, a non-scalar
  `backward()`, and an optimizer handed a bare parameter dict, all because the silent
  version of each is a model that trains and is quietly wrong. When in doubt, raise
  with a message naming the fix.
- **Match the file you are in.** Naming, structure, and density of explanation are
  already set by the surrounding code.
- Line length 88, enforced by ruff. Configuration lives in `pyproject.toml`; the rule
  set is pinned explicitly rather than inherited from whichever ruff is installed, and
  the ruff and mypy versions are pinned exactly for the same reason.

# PyNN — Full Codebase Review

Reviewed against the goal of using this as a portfolio project for software / ML engineering roles.

Every claim below was verified by running code (Python 3.14, NumPy 2.5.1, Numba 0.66) rather than
by reading alone. Where a bug is reported, the reproduction is included.

**Headline:** the architecture and API design are genuinely good, but **backpropagation was
incorrect in several places**, and `pip install .` produced a broken package. Both would be found
by an interviewer in minutes.

> **Status:** Tier 1 (items 1.1–1.8) and Tier 2 are **fixed and verified**. Of Part 2, sections
> A (gradient checking), B (layers, minus the recurrent/embedding items), C (`no_grad`, `detach`,
> iterative sort), D (`train`/`eval`, `state_dict`, recursive `parameters`, `DataLoader`-style
> shuffling), E (tests at 98% coverage), F (CI, badges, coverage floor), and G (`DESIGN.md`,
> benchmarks, the MNIST notebook) are done. See [Part 4](#part-4--work-completed) and
> [Part 5](#part-5--second-pass). What remains is listed in
> [Part 6](#part-6--what-is-still-open).

> **Warning:** the per-item "open" / "fixed" markers below are the state at the time each pass was
> written and some are stale. Part 5 and Part 6 are authoritative.

---

## Part 1 — Things that are incorrect

### Tier 1 — Correctness bugs in the core value proposition — **FIXED**

Everything in this tier has been fixed. The descriptions below document the original bugs; each
one now has a regression test. Read Part 4 for the resulting changes.

#### 1.1 Gradients are overwritten instead of accumulated

Most backward closures assign `x.grad = ...` where they must do `x.grad += ...`. Any tensor
consumed by more than one downstream op therefore ends up with only the *last* contribution,
silently producing wrong gradients.

Affected (all of these use `=`):

| File | Functions |
|------|-----------|
| `pynn/core/math.py` | `abs`, `sum`, `mean`, `exp`, `log` |
| `pynn/functional/activations.py` | `affine`, `relu`, `sigmoid`, `tanh`, `elu`, `selu`, `softplus`, `softmax` |
| `pynn/functional/losses.py` | `binary_crossentropy`, `categorical_crossentropy` |
| `pynn/functional/modules.py` | `conv2d` (for `X`, `K`, and `B`) |

Correct already (uses `+=`): all `Tensor` dunder ops, `identity`, `flatten`.

Reproduction — `f(x) = sum(relu(x)) + sum(x)`, so `df/dx` should be `1 + [x>0]`:

```python
x = Tensor(np.array([[1.0, -2.0]]))
(pmath.sum(F.relu(x)) + pmath.sum(x)).backward()
# x.grad == [[1., 1.]]   (relu's contribution was erased)
# correct  == [[2., 1.]]
```

This is why the library appears to work: plain MLPs are a single unbranched chain, so nothing
gets overwritten. It breaks the moment you add a residual connection, a multi-head output, an
auxiliary loss, or a regularization term — i.e. exactly the things an interviewer will ask about.

#### 1.2 `categorical_crossentropy` gradient is wrong (both `logits=True` and `logits=False`)

`pynn/functional/losses.py` computes `grad = (pred - true) / batch` and assigns it to `pred.grad`.
But `pred` at that point is the *softmax output*, not the logits. `(s - y)/B` is `dL/dlogits`;
what belongs on `pred.grad` is `dL/dsoftmax`. Softmax's own `reverse` then applies the softmax
Jacobian a second time to a quantity that already went through it.

```
pynn  dL/dlogits[0]: [-0.03887  0.01116  0.02771]
true  dL/dlogits[0]: [-0.10394  0.03734  0.06660]
```

Magnitudes are off by roughly 3x and **some entries have the wrong sign**. The `logits=False`
branch is wrong too: `dL/dp` for `-sum(y log p)/B` is `-y/(p·B)`, not `(p-y)/B`.

The fix is the standard one: fuse softmax into the loss, compute `(s - y)/B` directly against the
*logits* tensor, and make `logits=False` use the true `-y/(p·B)`.

Two consequences worth knowing:

- The repo's own `tests/functional/losses_test.py::test_multi_loss` compares against
  `torch.nn.CrossEntropyLoss` and **would fail today**. It has never run because the module can't
  be imported (see 1.5).
- `examples/mnist.py` still reaches ~94% test accuracy, because the wrong gradient is positively
  correlated with the right one. A correct implementation should get 97–98%; the gap is the bug.

#### 1.3 `Tensor.transpose` / `Tensor.T` silently detaches from the graph

```python
def transpose(self) -> Tensor:
    return Tensor(transpose(self.data))   # no children, no reverse
```

Anything reached through `.T` gets a zero gradient with no error. Verified: `sum(x.T @ y)` yields
`x.grad == 0`. Needs `add_children` plus a `reverse` that transposes the incoming gradient.

#### 1.4 `expand_array` / `shrink_array` corrupt pre-existing gradients

The broadcast-handling path in the binary ops does
`grad = expand_array(grad, out_shape)` → `+= out.grad` → `shrink_array(grad, orig_shape)`.
`expand_array` broadcasts the gradient that is *already accumulated* into the larger shape, and
`shrink_array` then sums it back down — multiplying the old value by the broadcast factor.

```python
b = Tensor(np.random.randn(2,))          # bias, batch of 4
pmath.sum(X @ W + b).backward()          # b.grad == [4., 4.]
pmath.sum(X @ W + b).backward()          # b.grad == [20., 20.]  — expected [8., 8.]
```

Gradient accumulation (two backward passes before a step, gradient accumulation over
micro-batches) is therefore broken. The right design is to leave `.grad` alone and reduce only
the *incoming* gradient: sum `out.grad` over the broadcast axes, then add.

#### 1.5 `pytest tests/` fails at collection on a clean checkout

Three of the five test modules `import torch` / `import tensorflow` at module scope, so collection
aborts before a single test runs:

```
ERROR tests/core/math_test.py            — No module named 'torch'
ERROR tests/functional/activations_test.py — No module named 'tensorflow'
ERROR tests/functional/losses_test.py    — No module named 'torch'
!!!!! Interrupted: 3 errors during collection !!!!!
```

Not "3 skipped" — the whole run exits non-zero. A reviewer who clones the repo and runs the tests
sees a red failure. Guard them with `pytest.importorskip("torch")` (and mark them
`@pytest.mark.slow` / `external`), so the default run is green with NumPy alone.

#### 1.6 `pip install .` ships a broken package

```toml
[tool.setuptools.packages.find]
include = ["pynn"]   # matches only the top-level package
```

`include` patterns match full dotted names, so `pynn.core`, `pynn.nn`, `pynn.optim`,
`pynn.functional`, and `pynn.utils` are all excluded. Verified:

```
$ ls site-packages/pynn
__init__.py

$ python -c "import pynn"
ModuleNotFoundError: No module named 'pynn.core'
```

Fix: `include = ["pynn*"]`. This also removes the need for the `sys.path` hacks in `main.py` and
`scripts/smoke_test.py`.

#### 1.7 SGD momentum implements the wrong formula

```python
cache["velocity"][key] = self.momentum * v + self.dampening * g   # dampening defaults to 0.0
```

PyTorch uses `buf = momentum*buf + (1 - dampening)*grad`. With the default `dampening=0.0` the
`pynn` velocity has **no gradient term at all** after the first step, so it decays to zero and
momentum does nothing (in fact it *slows* training).

Constant gradient of 1.0, `lr=0.1`, `momentum=0.9`:

```
pynn  : [0.9, 0.81,  0.729, 0.6561, 0.59049]   ← geometric decay, velocity dying
torch : [0.9, 0.71,  0.439, 0.0951, -0.31441]  ← accelerating, as intended
```

Note that `examples/mnist.py` and `examples/binary_classification.py` pass `dampening=0.1`, which
looks like it was tuned to work around this bug. Fix the formula, then drop `dampening` from the
examples.

#### 1.8 Numba, the advertised optional speedup, breaks the API and is slower

`@njit` decorates `Tensor.__getitem__` and `Tensor.__setitem__`. Numba cannot compile a method
taking `self`, so with Numba installed **all indexing raises**:

```python
x = Tensor(np.random.randn(3, 4))
x[0]        # numba.core.errors.TypingError: non-precise type pyobject
x[0] = ...  # same
```

Separately, the `@njit` primitives in `pynn/core/primitives.py` are elementwise wrappers over
operations NumPy already dispatches to BLAS/SIMD. Measured on 2000×2000 `add`, 20 iterations:

```
njit  add: 0.1745s
numpy add: 0.0919s   ← 1.9x faster without Numba
```

My recommendation: **delete the Numba layer entirely.** It adds a dependency, an import-time
try/except in two files, six `# TODO: FIXME` comments, a broken indexing path, and negative
performance. Removing it is a stronger engineering signal than keeping it. If you want a real
performance story, do `conv2d` via im2col + a single `matmul` — that is a 10–100x win and is
genuinely interesting to talk about.

### Tier 2 — Design and robustness problems

Status: 2.4, 2.5, 2.8, 2.9, and 2.10 were fixed alongside Tier 1. Items 2.1, 2.2, 2.3, 2.6, 2.7,
and 2.11 remain open.

#### 2.1 `freeze()` does nothing — **fixed**

`Module.freeze()` sets `trainable = False` on the module and its tensors, but no optimizer ever
reads `.trainable`. Verified: a frozen `Linear`'s weights are still updated by `SGD.update()`.
Either honor the flag in `Optimizer.update` or remove the method.

#### 2.2 `Sequential` cannot be nested — **fixed**

`Sequential` subclasses `Model`, not `Module`, and its `parameters` is a `list[dict]` while
`Module.parameters` is a `dict`. Nesting type-checks and even runs forward, then explodes at the
optimizer:

```python
m = Sequential([Sequential([Linear(3, 4)]), Linear(4, 2)])
opt.update()   # AttributeError: 'list' object has no attribute 'items'
```

This is the single most valuable structural fix. Make `Sequential` a `Module`, and give `Module` a
recursive `parameters()` that walks child modules. Everything in Part 2 (Dropout, BatchNorm,
blocks, `state_dict`) gets easier once containers compose.

#### 2.3 `Conv2d` bias is per-position instead of per-channel — **fixed**

`self.parameters["B"]` is built with shape `(out_channels, out_h, out_w)`. Standard `Conv2d` bias
is `(out_channels,)`. Consequences: parameter count scales with spatial resolution (for
`Conv2d(1, 4, 3)` on 8×8 input, 144 bias parameters vs. PyTorch's 4), the layer is no longer
translation-invariant, and the layer hard-fails on any other input resolution:

```python
c = Conv2d(1, 4, 3); c.build((2, 1, 8, 8))
c(Tensor(np.zeros((2, 1, 16, 16))))   # AssertionError — spatial size locked in at build
```

#### 2.4 `backward()` is recursive and unguarded — **fixed**

- Recursive topological sort → `RecursionError` at ~3000 graph nodes. Use an explicit stack.
- `self.grad = np.ones_like(self.data)` assumes a scalar root but never checks. Calling
  `backward()` on a non-scalar tensor silently seeds all-ones, which is a different function than
  the user thinks they differentiated. Raise unless the output is size-1 (or accept an explicit
  seed gradient).
- `backward()` doesn't zero gradients first, and given 1.4 that makes repeated calls actively
  wrong rather than merely accumulating.

#### 2.5 Numerical stability — **fixed**, except the bare `assert`s in `pynn/nn/modules.py`

- `softplus` computes `np.exp(x)` then `log(1 + e)`. `softplus(800)` returns `inf` and raises
  `RuntimeWarning: overflow encountered in exp`. Use `np.logaddexp(0, x)`.
- `sigmoid` uses `1/(1 + exp(-x))`, which overflows for large negative `x`. Use the
  branched/`tanh` form.
- `pynn/core/math.py::log` adds `EPSILON` *inside* the log. `EPSILON` is
  `np.finfo(float).eps ≈ 2.2e-16`, far too small to prevent `log(0) = -inf` and it biases the
  function. Clip the input instead.
- `Loss` and shape agreement are enforced with bare `assert`, which vanishes under `python -O`.
  Raise `ValueError` in library code.

#### 2.6 dtype handling — **fixed**

`Tensor.__init__` defaults `dtype=np.float64` unconditionally, so a float32 array is **silently
upcast**. And `self.grad` is hardcoded `np.zeros_like(self.data, dtype=np.float64)`, so an
explicitly-float32 tensor carries a float64 gradient. Default to preserving the input dtype, and
match the gradient dtype to the data.

#### 2.7 `get_batches` never shuffles — **fixed**

There is no `shuffle` parameter, so every epoch iterates identical batches in identical order —
that is not stochastic gradient descent. All three examples are affected. Add
`shuffle: bool = True` with a permutation, and make it a generator rather than materializing the
whole list.

#### 2.8 `activation_factory("elu")` raises — **fixed**

`ELU.__init__` requires `alpha` with no default, so `Linear(4, 4, activation="elu")` fails with
`TypeError: ELU.__init__() missing 1 required positional argument: 'alpha'`. Give it
`alpha: float = 1.0`, matching PyTorch and Keras. Also `functional.elu` should have the same
default.

#### 2.9 README documents an API that doesn't exist — **fixed**

Verified false claims in the API Overview table:

| README says | Reality |
|---|---|
| `pynn.core` exports math: `abs`, `sum`, `mean`, `exp`, `log` | `pynn/core/__init__.py` does not import `math`; none are accessible |
| `pynn.core` exports `expand_array` / `shrink_array` | Not exported from `pynn.core` |
| `pynn.nn` exports `activation_factory`, `initializer_factory` | `pynn/nn/__init__.py` doesn't import `factories` |
| "Full test suite: `pytest tests/ -v`" | Aborts at collection without torch + tensorflow |
| "`pip install numpy`" as the install step | Doesn't install the package; `pip install -e .` is needed (and is broken per 1.6) |

`pynn/functional/__init__.py` is a single `from .activations import *`, so `pynn.functional`
exposes no losses, initializers, or modules despite the package layout implying it does.

Also: the paths are wrong. README says `cd pynn` then `pytest tests/`, but `tests/` is at the repo
root, not inside `pynn/`.

#### 2.10 Test quality issues — **fixed**

- `tests/functional/losses_test.py` and `math_test.py` call `np.random.seed()` with **no
  argument**, which reseeds from OS entropy. Failures are unreproducible. Use a fixed seed or a
  `numpy.random.Generator` fixture.
- `activations_test.py` loops 50 times over fresh random data instead of parametrizing — slow,
  and on failure you can't tell which case broke.
- `keras_loss` in `test_binary_loss` is computed and never asserted on (dead code, flagged by
  ruff as `F841`).
- `tests/core/utils_test.py` has a stray `print(x_shrink)`.
- `tests/core/tensor_test.py::test_cast` uses `x.cast(int)`; elsewhere dtypes are NumPy dtypes.
- Every test asserts only shape/value equality against torch or TF. There is no test that any
  *gradient* is correct without an external framework installed.

#### 2.11 Lint / type-check baseline — **fixed**

Nothing is configured today. Current state if you turn the tools on:

| Tool | Result at review time | After Tier 1 fixes |
|------|-----------------------|--------------------|
| `ruff check` (default rules) | **64 errors** — 37 `I001` unsorted imports, 17 `RUF022` unsorted `__all__`, 3 `F841` unused vars, 2 `RUF013` implicit `Optional`, 2 `TRY004`, plus `RET501`, `RUF059`, `UP031` | still outstanding (config is step 9) |
| `ruff check --select F,E9 --ignore F403,F405` | 3 `F841` | **clean** |
| `ruff check --select F403,F405` | **73 errors** — 16 star-imports, 57 names used from them | unchanged |
| `ruff format --check` | 4 files need reformatting | library and tests formatted |
| `mypy pynn/ --ignore-missing-imports` | **69 errors in 8 files** | still outstanding |
| `pytest --cov=pynn` | **25%** line coverage | **84%** |

Two root causes drive most of the mypy noise:

- `Shape = tuple[None | int, ...]` in `pynn/core/types.py`. A shape never contains `None`; that
  `None` propagates into every arithmetic use and produces ~30 of the 69 errors
  (`Unsupported operand types for / ("float" and "None")` etc.). Use `tuple[int, ...]`.
- `Tensor.forward: str = None` should be `str | None`. Same for `math.sum(axis=None)` /
  `mean(axis=None)`, which are typed `int | tuple[int]` (also note `tuple[int]` means a 1-tuple;
  you want `tuple[int, ...]`).

Also `A001 builtin-variable-shadowing`: `pynn/core/math.py` shadows the builtins `abs` and `sum`.
That's a defensible choice for a numeric library (NumPy does it) but it forces the
`import pynn.core.math as pmath` dance and is why `math` isn't in `pynn/core/__init__.py`.
Consider exposing them only as `Tensor` methods and via `pynn.functional`.

### Tier 3 — Small things

Fixed alongside Tier 1: the unused `input_shape` in `conv2d`; `true_division` now actually used;
`__truediv__` no longer routes through `__pow__`; `__pow__` and `convert_tensor_input` raise
`TypeError`. Note that `pynn/core/primitives.py::transpose` became unused when `Tensor.transpose`
switched to `np.transpose` for axis support — step 8 removes that module anyway. The rest below
remain open.

- `pynn/core/tensor.py` — `TensorLike = ArrayLike` makes `isinstance(value, Tensor | TensorLike)`
  in `convert_tensor_input` misleading.
- `Module.summary()` returns live `Tensor` objects under `"parameters"`, so printing a summary
  dumps entire weight matrices. Return shapes and counts.
- `Optimizer` has `update()` where PyTorch has `step()`, and no `zero_grad()` — the README bills
  the API as PyTorch-like. Add `step()` and `Optimizer.zero_grad()` (aliases are fine).
- Both `lr` and `learning_rate` are accepted on every optimizer with no warning when both are
  passed. Pick one.
- `requirements.txt` lists `numba` as if required, and duplicates it in `requirements-dev.txt`
  even though `-r requirements.txt` already pulls it in. With `pyproject.toml` carrying optional
  dependency groups, the `requirements*.txt` files are redundant — delete them or make them
  one-line pointers.
- `pyproject.toml` has a `dev` group (pytest only) and a `test` group (pytest + torch +
  tensorflow); the naming is backwards from convention and neither includes ruff or mypy.
- No `py.typed` marker, so the type hints aren't visible to consumers.
- `pynn/__init__.py` has no `__version__` (version lives only in `pyproject.toml`).
- Docstrings say "computes the computetion" in ~8 places in `pynn/nn/activations.py` (typo, and
  the phrasing is circular). `Conv2d`'s default `name` is `"Conv2D"` while the class is `Conv2d`.
- `mean_squared_error`'s docstring references "Mininet's SquaredError", and `get_batches` mentions
  "mininet/pynet compatibility" — leftovers from a predecessor project that will confuse a reader.
- `.gitignore` ignores `data/` under the "Personal" heading, which silently excludes
  `examples/data/mnist/` (227 MB locally, correctly untracked, but a reader following the README
  won't know where the CSV comes from — `scripts/download_mnist.py` should be referenced).

---

## Part 2 — What to add

Ordered by return on effort for a resume project.

### A. Numerical gradient checking — **DONE**

This was the highest-value item on the list, and it is what caught 1.1–1.4 above. Items 1–6 below
are implemented in `pynn/testing.py` and `tests/test_gradcheck.py`; the remaining idea is the README
summary table. Kept here as a record of what the suite covers.

Build `pynn/testing.py` (shippable, not just a test helper) with:

```python
def numerical_gradient(fn, tensors, eps=1e-6): ...   # central differences
def check_gradients(fn, tensors, rtol=1e-5) -> GradCheckResult: ...
```

Central differences, `(f(x+h) - f(x-h)) / 2h`, on float64, compared with relative error
`|a - n| / max(|a| + |n|, tiny)`. Then a `tests/test_gradcheck.py` that parametrizes over:

1. **Every op individually** — each `Tensor` dunder, each `math` function, each activation, each
   loss, `linear`, `flatten`, `conv2d`.
2. **Broadcasting shapes** — `(3,4)+(4,)`, `(3,4)+(1,4)`, `(2,3,4)+(3,4)`, `(3,1)+(1,4)`. This is
   where 1.4 lives.
3. **Non-linear graph topologies** — this is the part that catches 1.1 and is usually missing from
   from-scratch autodiff projects:
   - a tensor used twice: `sum(x) + sum(x * 2)`
   - a branch that rejoins: `sum(relu(x)) + sum(x)`
   - a diamond: `sum(exp(x) * log(x + 3))`
   - tied weights: `tanh(tanh(x @ W) @ W)`
   - two heads sharing a trunk
4. **Kinks and edge cases** — `relu`/`abs` near 0 (offset your probe points, or assert the
   subgradient), `softmax` on large-magnitude logits, `log` near 0.
5. **End-to-end** — every parameter of a small `Sequential` against numerical gradients of the
   full loss.
6. **Optimizers against closed-form references** — a hand-written 5-line NumPy reference for each
   of SGD/Adam/RMSprop/Adagrad/Adadelta stepped 5 times on a fixed gradient. This catches 1.7,
   which no gradient check will.

Put a `--gradcheck` summary table in the README. "Every operation is verified against
central-difference numerical gradients, including branching graph topologies" is a strong,
concrete, verifiable claim.

### B. Layers and modules

You currently have `Linear`, `Conv2d`, `Flatten`, `Activation`, `Sequential`. For "sufficient
coverage of basic neural network layers", in priority order:

**Essential — a reviewer will notice these are missing:**

| Layer | Why |
|-------|-----|
| `Dropout` | The canonical reason to need train/eval modes (see D) |
| `BatchNorm1d` / `BatchNorm2d` | Running statistics + separate train/eval behavior; the single best demonstration that your autodiff handles non-trivial layers |
| `LayerNorm` | Simpler than BatchNorm, and the modern default |
| `MaxPool2d` / `AvgPool2d` | `Conv2d` without pooling is an incomplete CNN story; max-pooling's argmax-routed backward is a good gradient-check target |
| `Embedding` | Sparse gradient scatter-add; needed for any text example |

**Strong additions:**

- `Reshape` / `Unflatten` (you have `Flatten` but no inverse)
- `Sequential` as a `Module` so it nests, plus a `ModuleList`
- `Identity` as a layer
- `ConvTranspose2d` if you want an autoencoder example
- One recurrent cell (`RNNCell`, ideally `LSTMCell`) — backprop-through-time exercises the graph
  in a way feedforward nets don't, and it is the most interview-relevant thing you could add

**Activations to round out:** `GELU` (exact and tanh approximation), `SiLU`/`Swish`,
`Mish`, `Hardswish`, `LogSoftmax`, `PReLU` (a *learnable* activation — a good test that
`Activation` and `Module` compose).

**Losses:** `HuberLoss`/`SmoothL1Loss`, `BCEWithLogitsLoss` as a distinct fused class,
sparse/integer-label categorical cross-entropy (currently you must one-hot manually),
`KLDivLoss`, `HingeLoss`. Add `reduction` (`'mean'|'sum'|'none'`) uniformly — right now only MSE
has it.

**Optimizers:** `AdamW` (decoupled weight decay, and a nice contrast with Adam's L2), `NAdam`,
and a `pynn.optim.lr_scheduler` with `StepLR`, `ExponentialLR`, `CosineAnnealingLR`. Also
gradient clipping (`clip_grad_norm_`).

### C. Autodiff engine features

- `no_grad()` context manager — inference currently builds a full graph you throw away, which is
  both slow and a memory leak in a training loop that evaluates on a validation set.
- `Tensor.detach()` and a real `requires_grad` flag (`trainable` exists on `Tensor` but nothing
  reads it).
- `Tensor` methods for `reshape`, `sum`, `mean`, `max`, `min`, `transpose(axes)`, `squeeze`,
  and differentiable `__getitem__` — indexing currently returns a raw NumPy array and drops the
  graph, so slicing a tensor breaks backprop.
- `concat` / `stack` / `split`.
- Iterative (stack-based) topological sort — see 2.4.
- A `Tensor.graph()` / `to_dot()` visualizer. Cheap to write, and a computation-graph diagram in
  the README is the single most effective way to show a reader you understand autodiff.

### D. Model infrastructure

- `train()` / `eval()` mode propagating through the module tree (prerequisite for Dropout and
  BatchNorm).
- `state_dict()` / `load_state_dict()` plus `save`/`load` via `np.savez`. Checkpointing is table
  stakes and takes 30 lines.
- Recursive `named_parameters()` / `parameters()` on `Module`, and `num_parameters()`.
- A real `summary()` — layer name, output shape, parameter count, total (see 2.13).
- `Dataset` / `DataLoader` with shuffling, batching, and optional transforms, replacing
  `get_batches` (2.7).
- A `pynn.metrics` module: accuracy, precision/recall/F1, confusion matrix, MSE/MAE/R².
- `pynn.set_seed(n)` for reproducibility.

### E. Testing

Target ≥90% line coverage (**84%** after the Tier 1 work, up from 25%). Structure:

```
tests/
  conftest.py               # seeded rng fixture, tolerance constants, shared factories  <- TODO
  core/                     # tensor ops, math, utils, graph construction
  functional/               # activations, losses, module functions
  nn/                       # each layer: shapes, param init, build, backward     <- STILL MISSING
  optim/                    # each optimizer vs. closed-form reference            <- DONE
  test_gradcheck.py         # section A                                           <- DONE
  test_integration.py       # trains to a target accuracy on a tiny fixed dataset <- STILL MISSING
  external/                 # torch/tf comparisons (currently importorskip-guarded in place)
```

The largest remaining coverage gaps are `pynn/nn/modules.py` (45%) and `pynn/nn/factories.py` (55%),
both of which a `tests/nn/` directory would cover.

Specific things to add beyond section A:

- Per-layer tests: output shape, parameter shapes, lazy `build`, `include_bias=False`,
  `padding='same'`/`'valid'`, stride > 1, that `zero_grad` actually zeros, that `freeze` prevents
  updates (currently would fail — 2.1).
- Convergence tests: XOR to 100% in <500 steps; a 3-class linearly-separable blob to >95%. These
  are the tests that catch "gradients are subtly wrong but training still sort of works", which is
  precisely the failure mode of 1.2.
- Error-path tests: mismatched shapes raise, unknown activation/initializer names raise, bad
  `reduction` raises, `backward()` on non-scalar raises (after you add the check).
- Determinism: same seed produces identical weights after N steps.
- `pytest.ini` markers (`slow`, `external`) and `--cov-fail-under=90`.
- Property-based tests via Hypothesis over shapes/dtypes for the broadcasting logic in
  `expand_array`/`shrink_array` — that function is fiddly enough to deserve it.

### F. Tooling and CI

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF", "NPY"]

[tool.mypy]
python_version = "3.10"
strict = true          # start with warn_return_any + disallow_untyped_defs and ratchet up
plugins = []

[tool.pytest.ini_options]
addopts = "--strict-markers --cov=pynn --cov-report=term-missing"
markers = ["slow", "external"]
```

Then:

- **`.github/workflows/ci.yml`** — matrix over Python 3.10–3.13 × (with/without Numba, if you keep
  it), running `ruff check`, `ruff format --check`, `mypy pynn/`, and `pytest`. A separate job for
  the torch/TF comparison tests so the fast job stays fast.
- **`.pre-commit-config.yaml`** — ruff, ruff-format, mypy, trailing-whitespace, end-of-file-fixer.
- **Coverage badge** (Codecov or the GitHub Actions badge) plus a CI badge in the README. Badges
  are the fastest way for a reviewer skimming your GitHub to see the project is maintained.
- **`py.typed`** marker file in `pynn/` (2.20).
- Add ruff/mypy/pytest-cov to a properly-named `[dev]` extra.

### G. Documentation and presentation

- **README restructure.** Lead with a one-line pitch and the badges, then a computation-graph
  figure, then Quick Start. Fix everything in 2.9 first — a README that documents a
  non-existent API is worse than a short one.
- **`docs/DESIGN.md`** — the document that makes this project interview-ready. Cover: why
  define-by-run over a static graph; how the tape is built (`children` + `reverse` closures); the
  topological sort; how broadcasting is reversed; why softmax+cross-entropy is fused; the
  `Module`/`Model` hierarchy; what you deliberately left out and why. Interviewers ask "what were
  the hard parts" — this is that answer, written down.
- **A benchmark page.** `pynn` vs. PyTorch (CPU) on a fixed MLP and CNN: time per step and final
  accuracy. Being 20x slower than PyTorch is fine and expected; *knowing and reporting the number*
  is what reads as senior. This is also where an im2col `conv2d` rewrite pays off visibly.
- **Notebook example** — `examples/mnist.ipynb` with loss curves, a confusion matrix, and learned
  first-layer filters. Renders inline on GitHub, so it's the first thing anyone actually looks at.
- **`CONTRIBUTING.md`** with a "how to add a new layer" walkthrough (subclass `Module`, implement
  `forward`/`build`/`hyperparameters`, add the functional op with its `reverse`, add a gradcheck
  entry). Demonstrates you think about extensibility.
- **`CHANGELOG.md`** and a tagged `v0.1.0` GitHub release.
- Fix the docstring typos (2.20) and remove the mininet/pynet references.
- Consider publishing to TestPyPI — being able to say `pip install pynn-<yourname>` works is a
  differentiator, and it forces you to fix 1.6.

### H. Optional, high-signal stretch items

- **im2col `conv2d`.** Current implementation is a Python double loop over output positions. im2col
  + one `matmul` is 10–100x faster and is a classic interview topic.
- **A second example domain** — a char-level RNN on a small text file, or an MLP autoencoder on
  MNIST with a reconstruction grid. Shows the library generalizes past classification.
- **Gradient-checking as a public API.** Exporting `pynn.verify.check_gradients` so *users* can
  verify custom layers is the kind of API-design instinct that stands out.

---

## Part 3 — Suggested order of work

Steps 1–7 are **done**; see Part 4.

1. ~~**`include = ["pynn*"]`** in `pyproject.toml` (one character, unblocks installation).~~
2. ~~**Guard the torch/TF tests** with `importorskip` so `pytest` is green.~~
3. ~~**Build the gradient checker** (section A, items 1–2).~~
4. ~~**Fix `+=` accumulation** across `math.py`, `activations.py`, `losses.py`, `modules.py`; fix
   `transpose`; fix `expand_array`/`shrink_array`.~~
5. ~~**Add graph-topology gradient checks** (section A item 3) and confirm they pass.~~
6. ~~**Fix cross-entropy** by fusing softmax into the loss; verify MNIST accuracy moves to 97–98%.~~
7. ~~**Fix SGD momentum**; add the closed-form optimizer reference tests.~~
8. **Remove Numba**; delete the `# TODO: FIXME` comments it necessitated.
9. **Configure ruff + mypy**; fix `Shape`, then work down the 69 mypy errors.
10. **Make `Sequential` a `Module`** with recursive `parameters()`.
11. **Add `train()`/`eval()`, then `Dropout`, `LayerNorm`, `BatchNorm`, `MaxPool2d`.**
12. **Add `no_grad`, `detach`, `state_dict`.**
13. **Fill out tests to ~90%**; add convergence and integration tests.
14. **Add CI + pre-commit + badges.**
15. **Write `docs/DESIGN.md`**, fix the README, add the benchmark table.

Steps 1–7 are the ones that matter. Everything after that is polish, and polish on a correct
library is worth far more than polish on a broken one.

---

## Part 4 — Work completed

Steps 1–7 of the plan above. Every Tier 1 bug is fixed, with a regression test for each.

### Measured effect

| | Before | After |
|---|---|---|
| `pytest` on a clean checkout | aborts at collection (3 import errors) | 175 passed, 1 skipped |
| Line coverage | 25% | 84% |
| MNIST test accuracy (`examples/mnist.py`) | 94.38% | **98.04%** |
| Binary classification final loss | 0.032 | **0.00035** |
| `pip install .` then `import pynn` | `ModuleNotFoundError: No module named 'pynn.core'` | imports cleanly |
| Indexing a `Tensor` with Numba installed | `numba TypingError` | works |

### Changes

**Packaging** — `pyproject.toml`: `include = ["pynn*"]`, plus `strict-markers` and `external` /
`slow` markers.

**Autodiff correctness**

- Every backward closure now accumulates with `+=`, across `pynn/core/math.py`,
  `pynn/functional/activations.py`, `pynn/functional/losses.py`, and `pynn/functional/modules.py`.
- `pynn/core/utils.py`: `expand_array` / `shrink_array` replaced by `unbroadcast`, which reduces the
  *incoming* gradient over broadcast axes instead of mutating the operand's accumulated gradient.
  Also adds `matrix_multiply_gradients`, which handles `matmul`'s vector promotion and batch
  broadcasting (the old `.T` version was silently wrong for anything other than 2-D operands).
- `Tensor.transpose` now records a graph node and accepts an `axes` argument; `Tensor.T` inherits it.
- `Tensor.__truediv__` is a real division node rather than `self * other ** -1`, so it no longer
  fails on list operands and does not lose precision. Added `__rmatmul__`.
- `Tensor.backward` uses an explicit stack (5000-deep chains no longer raise `RecursionError`) and
  takes an optional `gradient` seed, raising `ValueError` instead of silently seeding ones when the
  output is not a single element.
- Removed the `@njit` decorators from `Tensor.__getitem__` / `__setitem__`, which made all indexing
  raise whenever Numba was installed.

**Losses**

- `categorical_crossentropy` fuses softmax so the gradient is taken with respect to the logits;
  the `logits=False` branch now uses the correct `-y / (p · batch)`. Verified against
  `torch.nn.CrossEntropyLoss` and against the closed form `(softmax(z) - y) / batch`.
- `binary_crossentropy` fuses sigmoid, using the overflow-free
  `max(z, 0) - z·y + log1p(exp(-|z|))` formulation. Verified against `torch.nn.BCEWithLogitsLoss`.
- Shape checks raise `ValueError` instead of using bare `assert`.

**Numerical stability** — new `pynn/core/numeric.py` with a branch-free stable sigmoid, used
by `sigmoid` and `softplus`; `softplus` uses `logaddexp` (`softplus(800)` returned `inf`, now
returns `800.0`); `elu` / `selu` use `expm1` on a clamped input; `math.log` clamps its input rather
than adding `EPSILON` to it.

**Optimizers**

- `SGD` momentum is now `buf = momentum·buf + (1 - dampening)·grad`. Verified to match
  `torch.optim.SGD` exactly for plain, Nesterov, dampened, and weight-decay configurations.
- `RMSprop`'s `maximize` flag was accepted and then ignored; it now works.
- `maximize` handling is consistent across all five optimizers (negate the gradient first, then
  apply weight decay, then always descend), matching PyTorch.
- State initialization keys off parameter presence rather than `time == 1`, so lazily-built layers
  are handled correctly.

**Activations** — `alpha` now defaults to `1.0` on both `functional.elu` and the `ELU` class, so
`activation="elu"` no longer raises `TypeError` from the factory. The class half of this was missed
on the first pass and caught later by `pynn.verify.check_api`, which walks every factory name.

### New tests

- `pynn/verify/gradients.py` — public `check_gradients` / `numerical_gradient`, with a
  `GradCheckResult` that reports the worst-disagreeing element so failures are diagnosable.
- `tests/test_gradcheck.py` — 117 checks: every operator across 7 broadcasting shape combinations
  and 8 matmul shape combinations, every activation at both normal and large magnitudes, every loss
  in both logits and probability forms, `linear` / `flatten` / `conv2d` across strides and paddings,
  9 graph-topology cases (shared input, rejoining branch, diamond, self-product, tied weights, two
  heads, residual, auxiliary loss), and full-MLP parameter gradients.
- `tests/optim/optimizers_test.py` — each optimizer against an independent closed-form transcription
  of its documented update rule, plus loss-reduction, reset-determinism, and lazy-parameter tests.
- `tests/core/utils_test.py` — rewritten for `unbroadcast` and `matrix_multiply_gradients`.
- Existing torch/TF tests now `importorskip` and are marked `external`; their unseeded
  `np.random.seed()` calls were replaced with seeded generators, and two tests were added for the
  loss paths that previously had no coverage.

### The `pynn/verify` package

The ad-hoc scripts used to find the bugs above were promoted into a shipped subpackage rather than
thrown away, so the properties they established stay established. It runs against an installed copy
of PyNN without pytest — `python -m pynn.verify` — and is also driven from `tests/test_verify.py`.

| Module | Suite | Checks |
|--------|-------|--------|
| `verify/gradients.py` | `check_all_gradients` | 111 — every op against central differences |
| `verify/stability.py` | `check_stability` | 20 — finiteness at `|x|` up to 1000 |
| `verify/invariants.py` | `check_invariants` | 56 — tape, optimizer, and factory behavior |

Two things make this more than a duplicate of `tests/`:

- **Every unary and binary op is checked twice**, once alone and once with its input reused by a
  second consumer. This is the shape that catches `grad =` where `grad +=` was meant, and it is
  invisible in the single-consumer form, where overwriting produces an exactly correct gradient.
  Bug 1.2 would have been caught immediately by the reused variant and not at all by the plain one.
- **Optimizers are compared against closed-form transcriptions** of their published update rules,
  not against "the loss went down". A momentum buffer with a dropped gradient term still descends.

The suite was mutation-tested to confirm it is not vacuously green. Reintroducing each of the four
original bug classes produces failures: reverting accumulation in `core/math.py` → 16 failures, in
`functional/activations.py` → 12; removing `unbroadcast` → 39; breaking the SGD momentum term → 3;
restoring the naive sigmoid → 2.

Deliberately *not* asserted, because they encode current bugs rather than regressions: `Module.freeze`
is ignored by the optimizers (2.1), `Conv2d` bias is per-position rather than per-channel, and
comparing two Tensors returns a Tensor rather than a bool. These are noted here instead.

### Note on Numba

With the `@njit` decorators removed from indexing, Numba no longer breaks the API. It still does not
help: the test suite takes 1.7s without Numba and 12.1s with it, and `add` on a 2000×2000 array
measured 1.9x slower than plain NumPy. Step 8 (removing it) still stands.

---

## Appendix — What's already good

Worth saying explicitly, because these are the parts you should talk about in interviews:

- The `functional` / `nn` split mirrors PyTorch and is applied consistently.
- Closure-based `reverse` functions are an elegant, readable tape implementation.
- Lazy `build()` with input-shape inference (`Linear(16)`) is a nice touch that Keras has and
  PyTorch doesn't.
- Broadcasting is handled *at all* in the backward pass — most from-scratch autodiff projects skip
  it and only support matched shapes.
- `softmax`'s backward uses the full Jacobian-vector product rather than assuming a
  cross-entropy downstream, and the reasoning is documented in the docstring.
- Optimizer hyperparameters genuinely match PyTorch's (`amsgrad`, `centered`, `nesterov`,
  `maximize`, `dampening`), which shows you read the papers and the reference implementations.
- `Conv2d`'s `'same'`/`'valid'` padding resolution is correct, including the asymmetric-padding
  arithmetic.
- All three examples run and converge; MNIST reaches 94% test accuracy.
- Clean commit history with sensible, incremental messages.


---

## Part 5 — Second pass

Steps 8–15 of the plan in Part 3, plus the structural work in Part 2 that they depended on.

### Measured effect

| | After Tier 1 | Now |
|---|---|---|
| Line coverage | 84% | **98.6%**, with a 95% floor enforced in CI |
| CI | none | green on Python 3.10–3.14, reproducible locally |
| A plain list of layers on a Module | silently never trained | `TypeError` naming `ModuleList` |
| `pytest -m "not external"` | 175 passed | **674 passed** |
| `python -m pynn.verify` | 187 checks | **302 checks** |
| Gradient-checked operations | 111 | **178** |
| `Sequential([Sequential([...]), ...])` | `AttributeError` in `update()` | trains |
| Evaluating a model | builds a full graph and discards it | `no_grad()` records nothing |
| A float32 array through `Tensor` | silently float64 | stays float32, gradient included |
| `get_batches` | identical order every epoch | shuffled, with a seedable generator |

### Coverage

Whole modules had no tests: the class-based losses, the class-based activations, both factories,
and `pynn/utils/data.py` at 21%. Closing them took the total from 92% to 98.6%. The threshold
lives in `[tool.coverage.report]` rather than pytest's `addopts`, so running a single test file
does not fail for covering a single module; CI opts in by passing `--cov=pynn`.

Writing the missing `Tensor` tests surfaced a live bug: NumPy won the dispatch for
`array + tensor` and coerced the Tensor to a 0-d object array, so `__radd__` was never called and
the result was an object-dtype array with no gradient. `__array_ufunc__ = None` (NEP 13) hands
those back to the reflected methods.

### The module tree

`Sequential` subclassed a separate `Model` type whose `parameters` was a `list[dict]` while
`Module.parameters` was a `dict`. `Module` now owns the tree:

- Child modules register automatically on attribute assignment, so a user-written composite layer
  does not silently train nothing.
- `named_parameters`, `parameter_groups`, `named_buffers`, `state_dict`, `zero_grad`,
  `train`/`eval`, `freeze`/`unfreeze` and `num_parameters` all walk it recursively.
- `parameter_groups()` hands the optimizer the modules' **live** dictionaries, which is what keeps
  lazily built layers working.
- Optimizers take the model itself (`SGD(model, ...)`), and raise a `TypeError` naming the fix
  rather than silently stepping nothing when handed the old `model.parameters`.
- `Model` had nothing left to do and was removed.

`summary()` now reports parameter shapes and counts instead of returning live Tensors, and
`Optimizer` gained `step()` and `zero_grad()` aliases.

### New layers

`Dropout`, `LayerNorm`, `BatchNorm1d`, `BatchNorm2d`, `MaxPool2d`, `AvgPool2d` — each a functional
op with a hand-written reverse plus an `nn` Module with a lazy build, and each on the gradcheck
sweep in both its plain and reused-input forms.

The normalization reverses are written out rather than composed from primitives, because the mean
and the variance depend on every element being normalized. Batch normalization at evaluation is a
genuinely different function (fixed statistics, so the gradient does not pass through them) and is
checked separately. Running statistics are **buffers**: `Module.register_buffer` keeps them out of
`parameter_groups` and inside `state_dict`.

### Autodiff controls

`no_grad()` / `enable_grad()` / `set_grad_enabled()`, `Tensor.detach()`, and a real
`requires_grad`, deliberately separate from `trainable` (a frozen parameter still receives
gradients; the optimizer is what skips it). Recording is gated in exactly two places inside
`Tensor` — `add_children` and the `reverse` setter — so no operation checks the mode itself. The
second gate is the one that matters for memory: the closure captures the forward pass's
intermediates, so a validation loop that collects predictions would otherwise retain every batch's
graph.

### Reproducibility and dtypes

`pynn.core.random` holds one generator shared by the initializers and `Dropout`, so a single
`set_seed(n)` covers a whole run. `Tensor` preserves a floating input's precision and matches the
gradient dtype to the data.

### An initialization bug found on the way

Building the notebook's CNN surfaced one that had been there all along: `he_normal` and
friends computed the fan-in as `shape[0]`, which is fan-in only for a 2-D `(in, out)`
weight matrix. For a `Conv2d` kernel, shaped `(out_channels, in_channels, kh, kw)`, that
reads the *output* channel count and drops the receptive field entirely — `Conv2d(16, 32, 3)`
came out with a standard deviation of 0.25 instead of 0.118, a factor of 2.1.

The weights still looked plausible, so nothing failed. What it did was compound through
depth: activations grew ~2x per layer, initial logits reached a standard deviation of 7.8,
the first loss was 15.2 instead of ln(10) ≈ 2.3, and the network collapsed into predicting
a constant at a learning rate a correctly initialized one handles. `fans()` now branches on
rank, since a `Linear` weight is `(in, out)` and a `Conv2d` kernel is `(out, in, kh, kw)`.

### Containers, and the last hole in the module tree

Auto-registration keys on `isinstance(value, Module)`, which left one hole: a plain list.
`self.blocks = [Linear(64), Linear(64)]` is not a Module, so nothing registered it — the
layers ran in the forward pass and received gradients, but were absent from
`named_parameters`, so no optimizer stepped them and no checkpoint saved them. The model
trained, the loss fell, and those layers stayed at their initial weights.

`ModuleList` and `ModuleDict` register their contents. Neither has a `forward`: they hold
modules whose wiring the enclosing module decides, and `Sequential` already covers "apply
these in order". Beyond that, `Module.__setattr__` now **refuses** a plain list, tuple, or
dict containing a Module and names the wrapper to use, so the mistake is a `TypeError`
where it is written rather than a model that never trains part of itself.

### CI, which had never actually run

The workflow was added in an earlier pass and no commit had been pushed since, so its
first run failed on all five versions at once — three separate faults, none of which
reproduce on the 3.14 the project is developed on:

- **mypy exited 2 on 3.11 and up.** `python_version = "3.10"` pinned the analysis target
  while the stubs on disk belonged to whatever NumPy the interpreter installed. Modern
  NumPy's `__init__.pyi` uses PEP 695 `type` statements, which mypy refuses to parse when
  told to assume 3.10; it aborted before checking anything.
- **`python -m pynn.verify` exited 2 on 3.10 and 3.11** — the exact command CI and the
  README use. `nargs="*"` combined with `choices=` makes argparse validate the empty
  default against the choice list, so it rejected `[]` as an invalid choice. Fixed in
  CPython 3.12; now validated by hand so the command behaves the same everywhere.
- **NumPy 2.5's stricter stubs surfaced two real annotation faults** that 2.4 does not:
  `np.prod` feeding `reshape` without an `int()`, and `np.number` reaching `Generator`
  methods that take a float.

Two smaller things came out of the same pass. The coverage upload was the last step, so
any earlier failure skipped it — a red run is exactly when the coverage delta is worth
seeing — and `relative_files` was off, so the report named an absolute path from the
machine that produced it.

The lesson is in `scripts/ci_matrix.sh`: every failure above was invisible on the newest
interpreter, and the loop for discovering that was a push and a wait. It runs every CI
step against every supported Python in a throwaway virtualenv per version.

### Activations, losses, and optimizers

**Activations.** `GELU` (exact and the tanh approximation), `SiLU`/Swish, `LogSoftmax`,
and `PReLU`. Three are stateless and subclass `Activation`; `PReLU` learns its negative
slope, which makes it the one with a design question. A parameter outside the module
tree is one no optimizer ever steps, so it is a `Module` — which means
`Linear(4, 3, activation="prelu")` registers it as a child and trains its slope at
`act_fn.alpha` alongside the weights. The factory's return type widens to cover both.

Storing its `num_parameters` argument under that name shadowed `Module.num_parameters()`
and turned every call into `"int object is not callable"`. It is `num_slopes`
internally, and a test now walks every shipped Module for attributes that hide a Module
method.

`log_softmax` exists because `log(softmax(z))` is `-inf` for any class the softmax
rounds to zero — exactly the class a cross-entropy loss cares about. GELU's exact path
needs `erf`, which NumPy does not have and SciPy supplies only through an optional
extra, so it goes through `math.erf` per element and the tanh approximation stays the
default.

**Losses.** `HuberLoss`/`SmoothL1Loss`, `SparseCategoricalCrossentropy` for integer
labels, `BCEWithLogitsLoss` as a distinct class, and a uniform
`reduction='mean' | 'sum' | 'none'` on every loss. The reduction goes through one shared
reducer: each loss supplies its per-item value and a `backward`, and reduction is the
only thing that differs between the three modes, which is what stops them drifting
apart.

`BCELoss` was an alias for `BinaryCrossentropy`, which defaults to `logits=True`. That
is backwards from PyTorch, where `BCELoss` takes probabilities and `BCEWithLogitsLoss`
takes logits — so anyone reaching for the familiar name got the other function, and got
it silently, since both accept the same shapes and return a plausible number. They are
now distinct classes with the PyTorch meanings.

**Optimizers.** `AdamW`, the `StepLR` / `ExponentialLR` / `CosineAnnealingLR` schedules,
and `clip_grad_norm` / `clip_grad_value`.

AdamW differs from `Adam(weight_decay=...)` by one line, and the line is the point: Adam
folds the decay into the gradient, so it passes through the same `1/sqrt(v)` rescaling
as everything else and a parameter with a large second moment receives *less* decay than
one with a small moment. The regularization strength ends up depending on gradient
history. AdamW applies it to the parameter directly. The closed-form transcription in
the tests is what pins the distinction down; "the loss went down" would not.

The schedules are pure functions of the epoch rather than of the previous learning rate,
so they resume from a state dict and an extra `step()` cannot compound a rounding error.
One check asserts the rate actually reaches the optimizer — a schedule computing rates
nobody reads would otherwise pass every other test.

`clip_grad_norm` takes the norm over every parameter at once and scales the whole set by
a single factor. Scaling each tensor separately would change their relative sizes, which
is to say it would change the direction of the step, and the direction is the part the
gradient got right. Clipping is meant to shorten the step, not turn it; the tests assert
the cosine with the original is 1.

### Presentation

- `docs/DESIGN.md` — the tape, the topological sort, `unbroadcast`, why the losses are fused, the
  module tree, `no_grad`, im2col, and what was left out.
- `benchmarks/benchmark.py` — PyNN against PyTorch on CPU. 1.7x on a matmul-dominated MLP, ~10x on
  a CNN. Reported, not hidden.
- `examples/mnist.ipynb` — training curves, confusion matrix, misclassified digits, first-layer
  weights as 28×28 images, a CNN with its learned kernels and feature maps, and a checkpoint round
  trip. Executed, so it renders on GitHub.
- README: CI, coverage, Python-version and licence badges; benchmark table; accurate API tables.

---

## Part 6 — What is still open

Nothing here is a correctness bug. In rough order of value:

1. **Differentiable indexing, `concat`, `stack`, `split`.** `Tensor.__getitem__` returns a raw
   NumPy array and drops the graph, which is the blocker for everything below it: `Embedding`
   needs a differentiable gather, attention needs `concat`, and a recurrent cell needs to slice a
   sequence. This is the structural item, the way `Sequential` was before it.
2. **`Embedding` and a recurrent cell.** Backprop-through-time is the most interview-relevant
   thing missing, and the iterative topological sort exists precisely so an unrolled recurrence
   does not blow the stack — it has just never been exercised. Depends on item 1.
3. **Numba removal.** Step 8 of Part 3, still never carried out. `pynn/core/primitives.py`
   decorates every elementwise primitive with `@njit` (a no-op fallback when Numba is absent),
   and the measurements in 1.8 stand: 1.9x slower than plain NumPy on a 2000x2000 add, and a test
   suite that goes from 1.7s to 12.1s with it installed. The reason it loses is worth writing
   down rather than just asserting — see the note below.
4. **`CONTRIBUTING.md`** with a "how to add a layer" walkthrough: subclass `Module`, implement
   `forward`/`build`/`hyperparameters`, add the functional op with its `reverse`, add a gradcheck
   entry. Plus `CHANGELOG.md` and a tagged `v0.1.0`.
5. **Packaging polish.** `py.typed` so the type hints are visible to consumers, `__version__` in
   `pynn/__init__.py`, and TestPyPI — publishing forces the packaging to stay honest.
6. **A graph visualizer** (`Tensor.to_dot()`). Cheap to write, and a computation-graph diagram in
   the README is the most effective way to show a reader the tape is real.
7. **`.pre-commit-config.yaml`**, and Hypothesis property tests over `unbroadcast` — that function
   is fiddly enough to deserve generated shapes rather than a hand-written list.
8. **Pinned tool versions.** `ruff>=0.6` and `mypy>=1.11` let CI install anything newer than the
   local versions, and a newer `ruff format` can reformat code that is clean here.
9. **Remaining nice-to-haves.** `Mish` and `Hardswish`; `KLDivLoss` and `HingeLoss`; `NAdam`;
   `ReduceLROnPlateau` and `OneCycleLR`; `ConvTranspose2d` for an autoencoder example;
   `Reshape`/`Unflatten` as the inverse of `Flatten`; a `pynn.metrics` module (accuracy,
   precision/recall/F1, confusion matrix); a second example domain such as a char-level RNN.

### Why Numba loses here, since the measurement is counter-intuitive

`@njit` is supposed to be faster, and on the code it is designed for it is. The primitives in
`pynn/core/primitives.py` are not that code. Each one is a thin elementwise wrapper — `add(a, b)`
is `a + b` — over an operation NumPy already dispatches to a vectorized SIMD or BLAS kernel
written in C. There is no Python-level loop left for Numba to remove, so there is nothing to win,
and three things to lose:

- **Dispatch cost per call.** Every call crosses the Python/JIT boundary and unboxes its
  arguments. For an operation whose body is one C call, that overhead is the whole cost.
- **Compilation on first use.** Each new dtype and rank combination triggers a fresh compile,
  which is most of the 1.7s → 12.1s the test suite loses: it exercises many small shapes once
  each and pays the compile every time without ever amortizing it.
- **No fusion across calls.** Numba can fuse loops *within* a compiled function. These are
  separate functions called one at a time by the tape, so each still writes a full intermediate
  array to memory. Elementwise work on large arrays is memory-bandwidth-bound, and the bandwidth
  is unchanged.

Numba would pay off on the parts that *do* have a Python loop, and the library has exactly one
left: `col2im`'s scatter over the output grid in the reverse pass of `conv2d` and the pooling
layers. That is the only place worth measuring before deciding, and it is a much better argument
for keeping an optional Numba path than the elementwise primitives are. The honest options are to
delete the layer, or to move it to `col2im` and re-measure; what is not defensible is keeping it
where it is because it sounds fast.

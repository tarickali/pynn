# TASKS

Work that is queued but not scheduled. Nothing here is a correctness bug — the library
is green on Python 3.10–3.14 with 681 tests, 302 verification checks, and 98% coverage.

Ordered by what unblocks the most. The first two are the current plan; everything below
is optional and independent, so it can be picked up in any order or dropped.

---

## In progress next

### 1. Differentiable indexing, `concat`, `stack`, `split`

`Tensor.__getitem__` returns a raw NumPy array and drops the graph. Nothing shipped
today needs it, which is exactly why it keeps getting deferred — and it blocks
everything in item 2.

- `__getitem__` records a node whose reverse scatters the gradient back into a
  zero array at the same indices. Basic slices, integer arrays, and boolean masks.
- `concat(tensors, axis)` — reverse splits the gradient back along `axis`.
- `stack(tensors, axis)` — `concat` of unsqueezed tensors.
- `split(tensor, sizes, axis)` — returns several tensors, each contributing to one
  slice of the input's gradient.
- Gradient cases in `pynn.verify.gradients.gradient_cases`, including the overlapping
  case where the same index is read twice and the contributions must sum.

### 2. `Embedding` and a recurrent cell

Backprop-through-time is the most interview-relevant thing missing, and the iterative
topological sort in `Tensor.backward` exists precisely so an unrolled recurrence does
not exhaust the Python stack — it has never actually been exercised. Depends on item 1.

- `Embedding(num_embeddings, dim)` — a differentiable gather, whose reverse is a
  scatter-add into the rows that were looked up. Repeated indices in one batch must
  accumulate.
- `RNNCell`, then ideally `LSTMCell`.
- A tied-weight test: the same cell applied at every timestep, which is the shape that
  catches a reverse pass overwriting rather than accumulating.
- A char-level text example would be the natural demonstration.

---

## Packaging and process

### 3. `CONTRIBUTING.md`, `CHANGELOG.md`, and a `v0.1.0` tag

The contributing guide should be a "how to add a layer" walkthrough, because that is the
question the codebase's structure actually answers: subclass `Module`, implement
`forward` / `build` / `hyperparameters`, add the functional op with its `reverse`, then
add a gradcheck entry so the new op joins the sweep.

### 4. Packaging polish

- `py.typed` marker, so the type hints are visible to consumers.
- `__version__` in `pynn/__init__.py` — the version currently lives only in
  `pyproject.toml`.
- TestPyPI. Publishing forces the packaging to stay honest.

### 5. Pin the tool versions

`ruff>=0.6` and `mypy>=1.11` let CI install anything newer than what is installed
locally, and a newer `ruff format` can reformat code that is clean here.
`scripts/ci_matrix.sh` makes it cheap to find out when that starts happening.

### 6. `.pre-commit-config.yaml`

ruff, ruff-format, mypy, trailing-whitespace, end-of-file-fixer.

---

## Nice to have

### 7. `Tensor.to_dot()`

A Graphviz dump of the tape. Cheap to write, and a computation-graph figure in the
README is the most effective way to show a reader the tape is real.

### 8. Property-based tests over `unbroadcast`

Hypothesis over shapes and dtypes. That function is fiddly enough — two reduction rules
that have to compose correctly — to deserve generated cases rather than a hand-written
list.

### 9. More layers, losses, and optimizers

Each is small and independent; this is the pile to draw from when time is short.

| Area | Candidates |
|---|---|
| Activations | `Mish`, `Hardswish` |
| Losses | `KLDivLoss`, `HingeLoss` |
| Optimizers | `NAdam` |
| Schedules | `ReduceLROnPlateau`, `OneCycleLR`, warmup |
| Layers | `ConvTranspose2d` (enables an autoencoder example), `Reshape` / `Unflatten` as the inverse of `Flatten`, `Identity` as a layer |
| Metrics | a `pynn.metrics` module: accuracy, precision / recall / F1, confusion matrix, MSE / MAE / R² |

### 10. A second example domain

A char-level RNN on a small text file, or an MLP autoencoder on MNIST with a
reconstruction grid. Shows the library generalizes past classification. Depends on item
2 for the recurrent version.

---

## Done, for reference

Recorded so this file does not re-propose them. Details in `PROJECT_REVIEW.md` Part 5.

- Autodiff correctness, `pynn.verify`, gradient checking as public API
- Nestable `Sequential` and the recursive `Module` tree; `ModuleList` / `ModuleDict`
- `Dropout`, `LayerNorm`, `BatchNorm1d` / `BatchNorm2d`, `MaxPool2d` / `AvgPool2d`
- `no_grad`, `detach`, `requires_grad`, dtype preservation
- `state_dict` / `load_state_dict`, `save` / `load`, buffers
- `GELU`, `SiLU`, `LogSoftmax`, `PReLU`
- `HuberLoss`, `SparseCategoricalCrossentropy`, `BCEWithLogitsLoss`, uniform `reduction`
- `AdamW`, `StepLR` / `ExponentialLR` / `CosineAnnealingLR`, `clip_grad_norm`
- im2col `conv2d`, and the JIT-compiled `col2im` scatter behind the `numba` extra
- CI across Python 3.10–3.14, coverage floor, `scripts/ci_matrix.sh`
- `docs/DESIGN.md`, benchmarks, the executed MNIST notebook

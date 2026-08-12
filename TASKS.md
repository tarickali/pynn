# TASKS

Work that is queued but not scheduled. Nothing here is a correctness bug — the library
is green on Python 3.10–3.14 with 790 tests, 340 verification checks, and 98% coverage.

Everything structural is done — the last item of that kind, differentiable indexing, is
what unblocked the recurrent cells — and so is everything in the packaging and process
section that used to lead this file. Items 1-5 are independent and can be picked up in
any order or dropped; the sequence-modelling section at the end is a dependency chain,
and is future work rather than queued work.

---

## Nice to have

### 1. `Tensor.to_dot()`

A Graphviz dump of the tape. Cheap to write, and a computation-graph figure in the
README is the most effective way to show a reader the tape is real.

### 2. Property-based tests over `unbroadcast`

Hypothesis over shapes and dtypes. That function is fiddly enough — two reduction rules
that have to compose correctly — to deserve generated cases rather than a hand-written
list.

### 3. More layers, losses, and optimizers

Each is small and independent; this is the pile to draw from when time is short.

| Area | Candidates |
| --- | --- |
| Activations | `Mish`, `Hardswish` |
| Losses | `KLDivLoss`, `HingeLoss` |
| Optimizers | `NAdam` |
| Schedules | `ReduceLROnPlateau`, `OneCycleLR`, warmup |
| Layers | `ConvTranspose2d` (enables an autoencoder example), `Unflatten` as the inverse of `Flatten`, `Identity` as a layer |
| Metrics | a `pynn.metrics` module: accuracy, precision / recall / F1, confusion matrix, MSE / MAE / R² |

### 4. Further acceleration, in measured order

`col2im` is compiled and the CNN profile is now flat. These are the remaining
candidates, each timed rather than guessed at. Every one of them costs a *second*
implementation — a NumPy fallback plus a scalar kernel that has to stay equivalent — so
the bar is "worth the duplication", not "faster in a microbenchmark".

| Candidate | Isolated speedup | Share of a step | Verdict |
| --- | --- | --- | --- |
| `col2im` scatter | 2–13x | 42% of a CNN step | **done** |
| Optimizer step | 4.6–7.2x | **32% of an MLP step** | in-place NumPy first |
| `stable_sigmoid` | 3.3–3.7x | model-dependent | maybe |
| `erf` | 7.1–7.5x | GELU exact path only | cheap, narrow |
| `im2col`'s copy | none | 4.6% of a CNN step | no |

**The optimizer step is the biggest remaining win, and the first move needs no
dependency.** `SGD.update` is 32% of an MLP training step, and most of that is
allocation: every line builds a fresh full-size array. Rewriting it with in-place NumPy
gives **1.8–2.5x** for free — no dependency, no second implementation:

```python
velocity *= momentum          # instead of velocity = momentum * velocity + g
velocity += grad
data -= lr * velocity
```

A fused `njit` kernel reaches 4.6–7.2x, so roughly half the available gain is free and
the other half costs a dual implementation *per optimizer* — five of them, each with
`maximize` / `nesterov` / `amsgrad` / `centered` variants. That is far more duplication
than `col2im`'s single scatter. Do the in-place rewrite, re-profile, and only then decide
whether the remainder is worth compiling.

**`stable_sigmoid`** builds a boolean mask and two fancy-indexed temporaries. A fused
kernel is 3.3–3.7x. It backs `sigmoid`, `softplus`, `silu`, and BCE-with-logits, so the
end-to-end share depends entirely on the model — worth profiling a sigmoid-heavy one
before committing.

**`erf`** is the best ratio and the smallest prize. It is `np.frompyfunc(math.erf)` today,
which is a Python call per element — a hidden interpreter loop of exactly the kind a JIT
removes, hence 7x. But it is only on GELU's exact path, and the tanh approximation is the
default. Cheap to add if the compiled module already exists.

**`im2col`'s `ascontiguousarray`** is 4.6% of a CNN step and is not a candidate: pure
memory movement with no arithmetic, where NumPy's copy is already a tuned memcpy. There
is no interpreter overhead to remove. Better addressed by avoiding the transpose than by
compiling the copy.

### 5. A second example domain

A char-level RNN on a small text file, or an MLP autoencoder on MNIST with a
reconstruction grid. Shows the library generalizes past classification. Everything the
recurrent version needs is now in place — `Embedding`, `LSTMCell`, differentiable
slicing, and `SparseCategoricalCrossentropy` — so this is mostly a notebook and a corpus.

---

## Sequence modelling

A dependency chain rather than a pile: each item is buildable once the one above it
exists, and the whole of it is buildable on the autodiff engine as it stands today.

**Nothing here needs a new autodiff primitive.** That was checked rather than assumed —
batched 4-D `matmul`, `softmax` over any axis, `transpose(axes)`, `reshape`, `concat` /
`stack` / `split`, differentiable indexing, and `where` / `masked_fill` are all in place,
and gradients flow through a full scaled-dot-product attention built from them. A causal
mask is `masked_fill(scores, future, -1e9)` ahead of the softmax, and the filled
positions come back with exactly zero gradient.

### 6. Fused recurrent layers

`RNNCell`, `LSTMCell`, and a `GRUCell` are the primitives; these are the layers that own
the loop, so a caller who does not need a custom one does not have to write it.

- **`GRUCell` first**, since it does not exist yet. Three gates rather than four:

  ```
  r = sigmoid(x @ W_ir + h @ W_hr + b)          reset
  z = sigmoid(x @ W_iz + h @ W_hz + b)          update
  n = tanh(x @ W_in + r * (h @ W_hn + b_hn))    candidate
  h' = (1 - z) * n + z * h
  ```

  Note where the reset gate goes. `LSTMCell` computes one `4 * hidden` block and slices
  it, because every gate consumes the same `x @ W_ih + h @ W_hh` sum. A GRU cannot do
  that: `r` multiplies the **hidden** projection only, before it is added to the input
  projection. So compute `x @ W_ih` and `h @ W_hh` as two `3 * hidden` blocks and split
  each — two gemms, not one, and not six. Getting this wrong produces a cell that trains
  and is simply not a GRU.

- **`RNN`, `LSTM`, `GRU` sequence layers** wrapping their cells. Return
  `(outputs, final_state)` where `outputs` is every timestep's hidden state stacked —
  `stack` already does this differentiably. Support `num_layers` (feed one layer's
  outputs to the next), `batch_first`, and dropout *between* layers but not after the
  last, matching PyTorch.

- **Bidirectional.** Run the sequence forward, run it again over the reversed sequence,
  and `concat` the two along the feature axis, so the output width is `2 * hidden_size`.
  Reversing is `x[:, ::-1]` — differentiable since indexing is. The reverse pass's
  outputs must be flipped back before concatenating, or timestep *t* of the backward
  direction lines up with timestep *T-t* of the forward one. That off-by-reversal is
  invisible in the shapes and shows up only as a model that will not learn.

- **Variable-length sequences.** A padding mask, applied so that padded steps neither
  contribute to the loss nor advance the hidden state. The alternative — PyTorch's
  packed-sequence representation — is more efficient and much more machinery; a mask is
  the right first version, and the docstring should say which was chosen and why.

- Gradcheck entries for `GRUCell` unrolled 1 and 4 steps, matching the existing
  `RNNCell` / `LSTMCell` cases, plus one bidirectional case. An invariant asserting that
  a bidirectional layer's two directions see the sequence in opposite orders.

### 7. Attention

- **`scaled_dot_product_attention(q, k, v, mask=None)`** in `pynn/functional/`:
  `softmax(q @ k.T / sqrt(d)) @ v`. Verified expressible today; the work is the API, the
  masking, and the tests.
- **`MultiHeadAttention`** as a Module: project Q, K, V, reshape to
  `(batch, heads, time, head_dim)`, attend, merge back, project out. The split and merge
  are `reshape` + `transpose(axes)`, which round-trip correctly today.
- **Masking.** Support both a causal mask and a padding mask. `masked_fill` already
  exists and is the intended tool: a filled position is overwritten rather than scaled,
  so it receives exactly zero gradient, which is what a mask should mean.
- **Self- versus cross-attention** falls out of letting `k` and `v` differ from `q`.
- Gradcheck entries with and without a mask, and one where the same tensor is passed as
  all three of Q, K, and V — self-attention is the case where one input has three
  consumers, which is exactly the shape a reverse pass that overwrites gets wrong.

### 8. Transformer

- **`TransformerEncoderLayer`**: multi-head self-attention, residual, `LayerNorm`,
  position-wise feed-forward (two `Linear` layers with `GELU` between them), residual,
  `LayerNorm`. Every piece already exists. Offer pre-norm as well as post-norm and say
  which is the default and why — pre-norm trains without a warmup schedule, which matters
  a great deal at this scale.
- **`TransformerEncoder`**: `num_layers` of the above in a `ModuleList`.
- **Positional encoding**: sinusoidal (no parameters) and learned (an `Embedding`).
  Both are cheap; ship both.
- **`TransformerDecoderLayer` / `TransformerDecoder`** with causal masking and
  cross-attention, if a generative example is wanted.
- **An example** is what makes this worth having: a small character-level or
  toy-translation transformer, in the shape of `examples/mnist.ipynb`. Item 9's char-RNN
  notebook would make a natural companion — the same task, two architectures, honestly
  compared.

### Scope note

This is a large body of work and the library does not need it to be complete or
defensible. It is here because it is the natural extension of what already exists, and
because "the autodiff engine is general enough that a transformer is a composition of
what is already in it, not a rewrite" is a claim worth being able to demonstrate rather
than assert.

If only part of it is ever built, **item 7 is the one to build**: attention is the
single most-asked-about architecture, and it is roughly a hundred lines on top of what
is already here.

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
- Differentiable indexing, `concat` / `stack` / `split`, `Tensor.reshape`
- `Embedding`, `RNNCell`, `LSTMCell`, and backpropagation through time
- im2col `conv2d`, and the JIT-compiled `col2im` scatter behind the `numba` extra
- CI across Python 3.10–3.14, coverage floor, `scripts/ci_matrix.sh`
- `docs/DESIGN.md`, benchmarks, the executed MNIST notebook
- `CONTRIBUTING.md` (the how-to-add-a-layer walkthrough), `CHANGELOG.md`, and the
  annotated `v0.1.0` tag — created locally, not pushed
- `py.typed` shipped as package data, and `pynn.__version__` as the single source of the
  version that `pyproject.toml` reads back, guarded by `tests/test_packaging.py`
- A `release` extra plus the build and `twine check` steps, documented in `USAGE.md`
  §11. **Nothing has been uploaded to TestPyPI** — that call is deliberately the
  maintainer's
- ruff and mypy pinned exactly, and `.pre-commit-config.yaml` pinned to the same two
  versions

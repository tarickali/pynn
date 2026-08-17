# TASKS

Work that is queued but not scheduled. The library is green on Python 3.10–3.14 with
982 tests, 415 verification checks, and 98% coverage.

Everything structural is done — the last item of that kind, differentiable indexing, is
what unblocked the recurrent cells — and so is everything in the packaging and process
section that used to lead this file. Items 2-5 are independent and can be picked up in
any order or dropped; the sequence-modelling section at the end is a dependency chain,
and is future work rather than queued work. Finished items move to the bottom **without
renumbering what is left**, since `docs/DESIGN.md` and `tests/core/tensor_test.py` cite
them by number.

**Item 4 is the exception to "nice to have".** It is the one thing here that produces a
wrong number rather than a missing feature. It is confined to an operation the library
does not support and PyTorch refuses by default, and `Tensor.free_graph()` now makes it
unreachable on any graph the caller frees — but it is wrong quietly, which is the
failure mode this codebase is otherwise built to avoid.

**Item 3a is the exception to "independent".** It is a measurement harness, and the two
remaining acceleration candidates cannot honestly be decided without it.

---

## Nice to have

### 2. More layers, losses, and optimizers

Each is small and independent; this is the pile to draw from when time is short.

| Area | Candidates |
| --- | --- |
| Activations | `Mish`, `Hardswish` |
| Losses | ~~`KLDivLoss`~~, ~~`HingeLoss`~~ done |
| Optimizers | ~~`NAdam`~~ done |
| Schedules | warmup. ~~`ReduceLROnPlateau`~~, ~~`OneCycleLR`~~ done — and `OneCycleLR` *is* the warmup, so a standalone one is now marginal |
| Layers | `ConvTranspose2d` (enables an autoencoder example). ~~`Unflatten`~~, ~~`Identity`~~ done |
| Metrics | a `pynn.metrics` module: accuracy, precision / recall / F1, confusion matrix, MSE / MAE / R² |
| Ops | `index_update`, the differentiable write — the one below with an argument behind it |

**`index_update` closes the indexing family.** Reading is differentiable: `x[key]`
gathers, and its reverse scatters the gradient back into the positions it read from.
There is no differentiable write to match it, and `__setitem__` deliberately is not one
— it mutates, and mutating a Tensor an operation already read corrupts that operation's
reverse pass (`docs/DESIGN.md` §14). The functional form has no such problem, because it
produces a new Tensor and leaves the original alone. JAX spells it `x.at[key].set(v)`:

```python
updated = index_update(x, key, value)   # a copy of x, with x[key] replaced
```

The reverse is a genuine scatter: `output.grad` with the written positions zeroed goes
to `x`, and `output.grad` gathered at those positions, unbroadcast, goes to `value`.
`where` and `masked_fill` already cover the mask-shaped cases, but neither expresses an
advanced-index write — `x[[1, 4, 7]] = v` needs a full-shape boolean mask to fake, and
duplicate indices have no meaning in that spelling at all.

Gradcheck entries: a basic slice, an advanced index, **duplicate indices** — where a
reverse that assigns instead of accumulating shows up — and a reused-input variant where
`x` also feeds a second consumer.

**Do the companion guard in the same change.** Assigning a *computed* Tensor into a
buffer is the way anyone coming from NumPy writes an unrolled loop:

```python
buffer = Tensor(np.zeros((batch, window, hidden)))
for t in range(window):
    buffer[:, t] = cell(...)        # accepted today
```

The assignment is not recorded, so nothing upstream of `cell` receives any gradient at
all — measured at exactly 0.0, with no error anywhere. It is the same failure
`__getitem__` used to have, and it is a likelier mistake than the mutation the current
guard catches. `Tensor.__setitem__` already refuses a computed *target*; refusing a
computed *value* is the same check on the other operand. Assigning a leaf Tensor stays
legal, since filling a buffer from data is what the escape hatch is for. The error
should name `stack`, which is what the caller wanted, and `index_update` once it exists.

### 3. Further acceleration, in measured order

`col2im` is compiled and the CNN profile is now flat. These are the remaining
candidates, each timed rather than guessed at. Every one of them costs a *second*
implementation — a NumPy fallback plus a scalar kernel that has to stay equivalent — so
the bar is "worth the duplication", not "faster in a microbenchmark".

| Candidate | Isolated speedup | Share of a step | Verdict |
| --- | --- | --- | --- |
| `col2im` scatter | 2–13x | 42% of a CNN step | **done** |
| Optimizer step | 4.6–7.2x compiled | ~30% → ~15% of a profiled MLP step | **in-place done, not compiling** |
| `stable_sigmoid` | 3.3–3.7x *isolated* | never measured end to end | **blocked on item 3a** |
| `erf` | 7.1–7.5x *isolated* | GELU exact path only | cheap, narrow, same caveat |
| `im2col`'s copy | none | 4.6% of a CNN step | no |

Read the "isolated speedup" column with the warning in **item 3a** in hand. Every number
in it is a microbenchmark, and for the optimizer — the one candidate that has now been
built and measured both ways — the microbenchmark could not be converted into an
end-to-end number at all.

#### The optimizer step: done, in place, no dependency

All six update rules mutate their buffers with `*=` / `+=` / `out=` and apply the step as
`data -= ...`. The arithmetic is **bit-for-bit identical**, checked directly against the
pre-rewrite implementations across 80 flag combinations with `array_equal` rather than
`allclose`. Momentum SGD went from seven full-size allocations per parameter per step to
one; Adam from sixteen to three. Those two facts are countable rather than timed, and they
are the firmest thing here.

The timing that *is* reliable is isolated `update()` on a single parameter, old and new
alternating per call, reproduced across three runs:

| ratio old/new | 10 | 256 | 2,560 | 65,536 | 200,704 |
| --- | --- | --- | --- | --- | --- |
| SGD | 1.23x | 1.29x | 1.51x | 2.14x | 2.60x |
| SGD, momentum 0.9 | 1.39x | 1.49x | 1.66x | 2.09x | 3.08x |
| Adam | 1.14x | 1.23x | 1.22x | 1.45x | 1.92x |
| RMSprop | 1.16x | 1.22x | 1.21x | 1.44x | 1.64x |
| Adagrad | 1.19x | 1.25x | 1.22x | 1.40x | 1.52x |
| Adadelta | 1.17x | 1.26x | 1.25x | 1.31x | 2.05x |

**The gradient of that table is the point.** The win is allocation, so it grows with the
array, and on a small parameter there is almost nothing to save — what is left is
per-call NumPy dispatch, which the in-place form does not reduce and in places increases.
The benchmark MLP has six parameters and *four* of them are in the first three columns
(256, 256, 2,560, 10). That is the honest explanation for why a 2–3x on the function does
not become a 2–3x on the step, and it is a better one than the cache story an earlier
draft of this section gave.

Under cProfile, interleaved and repeated in one process, the optimizer's share of an MLP
step goes from **~30% to ~15%**. Note what that is: a share of profiled time, not of wall
clock. It is quoted because it is the one before/after comparison the machine could make
repeatably.

**Not compiling the remainder**, and item 3a is now most of the reason. A fused `njit`
kernel was measured at 4.6–7.2x by the same isolated method that has just been shown not
to survive contact with a training loop. Even taking it at face value, an update rule that
cost *zero* would save ~15% of a profiled step — for a dual implementation per optimizer,
six of them, each with `maximize` / `nesterov` / `amsgrad` / `centered` variants, against
`col2im`'s single scatter. Revisit only with a trustworthy end-to-end harness and a model
whose parameters are large enough for the right-hand columns above to be the typical case.

### 3a. There is no harness that can measure an end-to-end speedup

**This blocks the rest of item 3**, and it came out of the optimizer work rather than
being anticipated.

Two harnesses were written to convert the isolated numbers above into end-to-end ones,
and neither is usable:

- **A paired in-loop timer** — build two models, run the same forward and backward for
  each, and time only `update()`, alternating per sample with the GC off. It has a
  **demonstrable ordering bias: whichever side of the pair runs first wins, by up to
  1.5x.** Measured old-first, three optimizers came out *slower* after removing thirteen
  allocations, which the size table above says is impossible. Swapping the order reverses
  the verdict. The bias exceeds the signal.
- **Whole-step timing, fresh process per configuration** — the pattern
  `benchmarks/memory.py` already uses. The effect is ~0.1–0.2 ms of a ~3.2 ms step and
  the run-to-run spread is ±0.4 ms, so six alternating runs interleave with no signal.
  This was on a desktop at load average 3.6; a quiet machine may do better, but that is a
  hypothesis rather than a result.

What that leaves is real but narrow: allocation counts, isolated per-size ratios, and a
cProfile share. `stable_sigmoid` and `erf` both need an end-to-end number before anyone
commits to a second implementation, so **building the harness is the prerequisite, not
the follow-up.** What it would need:

- A fresh process per configuration, and both orderings run, with the disagreement between
  them reported rather than averaged away — that disagreement is the error bar.
- Enough repetitions to state a confidence interval instead of a single ratio, and a
  refusal to report a ratio the interval straddles 1.0.
- A model whose parameter sizes span the table above, so a per-size claim can be made
  rather than one number for "an MLP".
- Ideally, a quiet machine, or `taskset`-style pinning — and a recorded load average, so a
  reader knows which of the two situations they are looking at.

`benchmarks/benchmark.py` is not that harness and should not be made into it: its job is
the PyTorch comparison, where the factor is large enough that ±15% does not matter.

**The README's benchmark table is owed a re-measurement for the same reason.** Its rows
were taken in a machine state that could not be reproduced while the optimizer work was
going on: PyTorch's own numbers came out 15–40% away from what the table records (its
LSTM row moved from 14.5 to 17–21 ms), so the whole machine was in a different place
rather than PyNN having changed. The table was left alone on the grounds that replacing a
coherent set of numbers with a noisier one is a loss, and the ±15% caveat under it is
doing real work. Re-take all three rows in one sitting on a quiet machine, with and
without the `numba` extra, and record the load average alongside them.

**`stable_sigmoid`** builds a boolean mask and two fancy-indexed temporaries. A fused
kernel is 3.3–3.7x *in isolation*. It backs `sigmoid`, `softplus`, `silu`, and
BCE-with-logits, so the end-to-end share depends entirely on the model — and, on the
evidence above, on the array sizes as much as on the model.

**`erf`** is the best ratio and the smallest prize. It is `np.frompyfunc(math.erf)` today,
which is a Python call per element — a hidden interpreter loop of exactly the kind a JIT
removes, hence 7x. But it is only on GELU's exact path, and the tanh approximation is the
default. Cheap to add if the compiled module already exists.

**`im2col`'s `ascontiguousarray`** is 4.6% of a CNN step and is not a candidate: pure
memory movement with no arithmetic, where NumPy's copy is already a tuned memcpy. There
is no interpreter overhead to remove. Better addressed by avoiding the transpose than by
compiling the copy.

### 4. A second `backward` over one graph compounds

Found while writing up `free_graph`, by asking what the call it refuses would have done.

Every Tensor carries a `grad`, intermediates included, and every reverse closure reads
its output's **stored** gradient rather than a value handed to it. So a second pass over
one finished graph finds the first pass's gradients still sitting on every intermediate
and propagates them again. It does not double. It compounds, and the overshoot grows
linearly with depth:

| chain depth | one pass | two passes | should be |
| --- | --- | --- | --- |
| 1 | 3.0 | 12.0 | 6.0 |
| 2 | 9.0 | 45.0 | 18.0 |
| 4 | 81.0 | 567.0 | 162.0 |

PyTorch stores gradients **on leaves only** — an intermediate's `.grad` is `None`, and
accessing it warns you it always will be. Its gradients are transient values passed
between `grad_fn` nodes, so a second pass recomputes them from a clean seed and doubles
exactly. Its default refuses the second pass anyway: *"Trying to backward through the
graph a second time"*. `retain_graph=True` is the opt-in.

Scope, so this is not read as worse than it is. Gradient accumulation over
micro-batches — separate forward passes over the same leaves, no `zero_grad` between
them — is **correct** and is what the accumulation contract is about; there is an
invariant and a test for it. What is wrong is re-running one graph, which nothing in
this repository does and which `free_graph` now refuses outright. The behaviour is
pinned by `tests/core/tensor_test.py::test_a_second_backward_over_one_graph_compounds`
so it cannot drift silently, and that test asserts what the library does rather than
what it should.

Three ways to close it:

- **Zero the non-leaf gradients a previous pass filled.** Not every non-leaf, every
  pass: a fresh graph's intermediates are already zero from `Tensor.__init__`, so the
  only ones needing it are the ones a previous `backward` wrote to. Mark each node as
  the reverse pass runs it, and zero only marked nodes on the way in. The first pass
  costs one flag check per node — the same shape as the freed-graph scan `backward`
  already does, measured at 0.03 ms on a 1,234-node graph — and the memset only
  happens on the second pass, which is the one that is currently wrong. This is much
  cheaper than "an allocation per node per backward" makes it sound, and it keeps
  every intermediate's gradient readable afterwards.
- **Store gradients on leaves only**, as PyTorch does, and pass intermediates
  transiently. Correct by construction, and a rewrite of every reverse closure in the
  library — `x.grad += ...` is the idiom `docs/DESIGN.md` §2 is built around. It also
  deletes a real capability rather than merely costing effort: `∂L/∂h_t` at every
  timestep is how you find where an unrolled recurrence stops propagating credit,
  `∂y_c/∂A` at a feature map is Grad-CAM, and reading the gradient either side of a
  fused reverse is how you check one by hand. PyTorch needs `retain_grad()` or a hook
  for all three; here they are `h.grad`. **Not recommended.**
- **Make freeing the default** — `backward(retain_graph=False)`, so the wrong answer
  becomes unreachable rather than merely documented. Cheapest, and the direction
  `free_graph` was already pointing; the objection to it is weaker than it looks, since
  nobody can be relying on a result that has never been right.

The first and third are **complements, not alternatives**, and together they are what
PyTorch does: a retained graph gives correct answers, and you have to ask for one. Take
them in that order. The first is nearly free and makes retention correct, which is what
makes the second safe to ship — flipping the default while `retain_graph=True` still
returns a compounded gradient would just move the wrong number behind a flag.

### 5. A float32 path

**The tape already preserves it.** A float32 array stays float32 through a forward and
backward pass, gradients included — there is an invariant and a test for that. What is
missing is a way to get a whole *model* into float32, and it is two specific things
rather than anything deep:

```
he_normal((4, 3)).dtype    ->  float64      every initializer
Tensor(2.0).dtype          ->  float64      every Python scalar an operator meets
```

1. The eleven initializers in `pynn/functional/initializers.py` call `np.zeros`,
   `Generator.normal` and friends without a dtype, so they return float64 and a layer's
   weights are float64 from the moment it builds. The first matmul promotes every
   activation downstream of it.
2. `convert_tensor_input` wraps a Python scalar in a float64 *array*, so `x * 2.0`
   promotes a float32 tensor. NumPy would not: under NEP 50 a Python scalar is weak and
   `float32_array * 2.0` stays float32. The promotion is this library's, not NumPy's.

Both are mechanical. A `dtype` threaded through the initializers, `initializer_factory`,
and each layer's `build`; or one module-level default in the shape of
`pynn.core.random`'s generator, which is how PyTorch spells it (`set_default_dtype`).
Then either weak-scalar handling in `convert_tensor_input` or inheriting the other
operand's dtype.

Worth it for the memory: every intermediate on the tape carries a data array and a
gradient array of the same size, so a float32 graph is half of a float64 one — directly
against the 860 MB in item 4. The speed is a maybe rather than a promise: `sgemm` has
twice `dgemm`'s SIMD lanes, but whether that reaches a model dominated by Python
dispatch is a measurement, not an inference. Measure before quoting anything.

**`pynn.verify` must stay float64.** Central differences at `eps = 1e-6` subtract two
nearly equal numbers, and float32 carries about 7 decimal digits — the cancellation
would eat most of them and the checker would report failures that are its own
arithmetic rather than the library's. The sweep stays double precision, and a float32
path is verified another way: dtype propagation through every layer, and agreement with
the float64 result to a tolerance float32 can actually meet.

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
  toy-translation transformer, in the shape of `examples/mnist.ipynb`.
  `examples/char_rnn.ipynb` already makes the natural companion — same corpus, same
  vocabulary, same batching, so a transformer notebook could be held to its numbers
  directly rather than compared by anecdote.

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
- `Tensor.free_graph()`, and with it the `gc.collect()` cadence the char-RNN notebook
  used to carry. A finished graph is a reference cycle, since every reverse closure
  references the Tensor it belongs to, so nothing reclaimed it until CPython's collector
  ran — on a schedule set by object counts, not by the megabytes of arrays hanging off
  them. Clearing each node's `children` and `reverse` over the order `backward` walks
  breaks the cycles and lets reference counting take the graph back. Over 400 steps of
  the char-RNN's model: **242.0 ms/step at 3,813 MB peak against 81.0 ms/step at
  860 MB**, and collecting on top of it buys nothing. A method rather than a
  `backward(retain_graph=False)` default, so nothing that worked stopped working, and
  `backward` refuses a freed graph rather than reporting zeros. `python -m
  benchmarks.memory` is the harness
- A second example domain: `examples/char_rnn.ipynb`, a character-level LSTM on Tiny
  Shakespeare (`scripts/download_shakespeare.py`), executed and committed with its
  outputs. It turned up the tape-lifetime problem above and needed no library code
- `CONTRIBUTING.md` (the how-to-add-a-layer walkthrough), `CHANGELOG.md`, and the
  annotated `v0.1.0` tag — created locally, not pushed
- `py.typed` shipped as package data, and `pynn.__version__` as the single source of the
  version that `pyproject.toml` reads back, guarded by `tests/test_packaging.py`
- A `release` extra plus the build and `twine check` steps, documented in `USAGE.md`
  §11. **Nothing has been uploaded to TestPyPI** — that call is deliberately the
  maintainer's
- ruff and mypy pinned exactly, and `.pre-commit-config.yaml` pinned to the same two
  versions
- A Graphviz dump of the tape: `pynn.viz.to_dot`, with `Tensor.to_dot` as a facade, and
  the figure it generates at the top of the README
- **Item 1**, property-based tests over `unbroadcast` and `matrix_multiply_gradients`.
  Hypothesis generates the shape pairs; the properties are the two adjoint identities
  rather than shape assertions, since a gradient of ones survives summing the wrong
  axis and a squeeze that should have been a sum still produces the right shape.
  Mutation-checked against the hand-written list they sit beside: transposing the wrong
  operand inside `matrix_multiply_gradients` passes every parametrized case and fails
  the property

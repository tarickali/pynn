# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the public API may change between minor versions.

## [Unreleased]

### Added

- **Property-based tests over `unbroadcast` and `matrix_multiply_gradients`**, the two
  functions every gradient in the library is routed through. Hypothesis generates
  broadcast-compatible shape pairs and valid matmul operand shapes rather than the
  hand-written list that was there before, and the properties are the two adjoint
  identities: `<broadcast_to(x), g> == <x, unbroadcast(g)>`, and
  `<A @ B, G> == <A, dA> == <B, dB>`. The adjoint is what makes them worth having over
  a shape assertion — a gradient of all ones is invariant under summing the *wrong*
  axis, and a squeeze that should have been a sum produces an array of exactly the
  right shape. Mutation-checked: transposing the wrong operand in
  `matrix_multiply_gradients` is caught only by the property, and passes the entire
  parametrized list. `hypothesis` joins the `dev` extra, configured `derandomize=True`
  so the examples explored are a function of the commit rather than of the clock —
  otherwise the five interpreters in the CI matrix each roll their own dice on every
  push.
- **`Tensor.free_graph()`.** Releases the tape behind a Tensor by clearing each node's
  `children` and `reverse` over the order `backward` walks. Every Tensor holds its
  reverse pass as a closure that references the Tensor it belongs to, so a finished
  graph is a reference cycle: dropping the last name pointing at a loss frees nothing,
  and only CPython's cyclic collector can — which it schedules from object counts rather
  than from the hundreds of megabytes of arrays hanging off those objects. Breaking the
  cycles lets reference counting reclaim the graph immediately. `data` and `grad` are
  untouched, so the parameters still carry the gradients the optimizer is about to read.
  On `examples/char_rnn.ipynb`'s model over 400 steps — batch 32, a 64-step unrolled
  `LSTMCell`, ~150 MB of graph per step — this is **242.0 ms/step at 3,813 MB peak RSS
  without it against 81.0 ms/step at 860 MB with it**: 3.0x the throughput and 4.4x less
  memory. It is a new method rather than a new default for `backward`, which is how
  PyTorch spells the same trade, because re-running a graph is something a caller may
  reasonably expect to work; the default can still be changed later, and cannot be
  changed back quietly. `backward` refuses a graph that has been freed rather than
  walking the stump and reporting zeros, which would be indistinguishable from a
  converged model.
- **`benchmarks/memory.py`.** `python -m benchmarks.memory` measures peak RSS and
  throughput for that model under each way of dealing with the finished graph, in a
  fresh process per configuration with RSS sampled from `ps`. Both of those are
  load-bearing: a heap already grown by an earlier configuration does not shrink back,
  and `resource.getrusage(...).ru_maxrss` reported near-identical peaks for
  configurations whose real peaks differed by more than 2 GB. It defaults to 400 steps
  because the same benchmark over 60 reports the opposite conclusion.
- **`pynn.viz`.** `to_dot(tensor, parameters=None, max_nodes=200)` walks the tape behind
  a Tensor and returns Graphviz DOT — one node per tensor, labelled with the operation
  that produced it and its shape, rounded for a computed node and squared off for a
  leaf, with parameters named from the model and filled differently again. Nodes are
  identified by `id()`, the same way `backward`'s visited set is, so a tensor with two
  consumers appears once with an edge to each. `max_nodes` caps the drawing and a marker
  node reports how much was cut; the walk is breadth-first from the output, so what a cap
  keeps is the part nearest the loss. Emitting text is the whole of it: nothing shells
  out to Graphviz or imports the `graphviz` package, and rendering stays the caller's.
  `Tensor.to_dot()` is a facade over it.
- **A second example domain**, `examples/char_rnn.ipynb`: a character-level language
  model on Tiny Shakespeare, executed and committed with its outputs. `Embedding` →
  `LSTMCell` unrolled over a 64-character window → `Linear`, trained with `AdamW`,
  `CosineAnnealingLR`, and `clip_grad_norm`, with text sampled at six checkpoints, a
  temperature-controlled sampler, and the model's learned character transitions checked
  against the corpus's own bigram statistics. No library code was needed to write it —
  `Embedding`, `LSTMCell`, differentiable slicing, and `SparseCategoricalCrossentropy`
  already covered it. `scripts/download_shakespeare.py` fetches the corpus using nothing
  but the standard library.
- **A tape figure**, `docs/tape.svg`, at the top of the README and in `docs/DESIGN.md`
  §2 — the fifteen nodes a two-layer MLP and a squared-error loss actually leave behind.
  `python scripts/generate_tape_figure.py` regenerates it; only that script needs
  Graphviz, and it writes the DOT either way.

### Fixed

- **The pooling gradient checks passed at the default seed and failed at others.** Both
  drew their inputs from a permutation of `0..n`, which max pooling needs — a probe of
  size `eps` must not be able to change which element wins a window — and average
  pooling does not. At values running to 99 the sum a central difference differences
  reaches the hundreds, and round-off in `f(x + eps) - f(x - eps)` divided by
  `2 * eps = 2e-6` cleared `atol` for the gradient elements near zero. Three of six
  seeds failed. The permutation is now scaled into `[-0.5, 0.5]`, which keeps the gap
  between values four orders of magnitude above `eps`, and average pooling takes the
  same normal inputs as everything else. The whole sweep is now clean across 60 seeds,
  and `tests/test_gradcheck.py` pins the pooling cases at four of them — these were the
  only cases in the sweep whose inputs were not O(1) by construction, which is what
  made them the ones to drift.

### Changed

- **Every optimizer's `update` runs in place.** Profiling an MLP training step put 32%
  of it in `SGD.update`, and none of that was a Python loop — it was allocation. Each
  line of each update rule built a fresh full-size array: seven per parameter per step
  for momentum SGD, sixteen for Adam, two of the seven spent computing
  `grad + 0.0 * data`. The rules now mutate their buffers with `*=` / `+=` / `out=` and
  apply the step as `data -= ...`, which takes momentum SGD to one allocation and Adam
  to three. **The arithmetic is bit-for-bit identical** — the closed-form reference
  transcriptions passed unchanged, and the equality was checked directly against the
  pre-rewrite implementations across 80 flag combinations with `array_equal` rather than
  `allclose`. Momentum SGD's `update` is 3.0x faster measured in isolation and 1.6x
  inside a real training loop, where the forward and backward passes have evicted the
  arrays it touches; it is now ~15% of an MLP step rather than ~32%, and the step itself
  is about 8% faster, inside the benchmark table's stated run-to-run variation. That gap
  between the isolated and in-loop numbers is why the compiled `njit` kernel `TASKS.md`
  measured at 4.6–7.2x was not built: the ceiling on it is now 13% of a step, in
  exchange for a second implementation of six optimizers with four flag variants each.
- **`param.data` is written through rather than rebound.** A consequence of the above,
  and a deliberate one: a caller holding the array — `Tensor.numpy()` returns it — now
  sees training happen, as it would in PyTorch. `state_dict()` and `detach()` copy, so a
  checkpoint is still a snapshot. The new `effective_gradient` helper, which applies
  `maximize` and coupled weight decay, returns a **read-only** array for the same
  reason: when neither applies it is a view of `param.grad` rather than a copy, and an
  update rule that wrote through it would corrupt a gradient the caller still owns.
  `check_invariants` pins both across every flag combination, since neither is visible
  in a trajectory — a rule that consumed `param.grad` takes a correct first step and a
  wrong second one.
- **`Tensor.__setitem__` refuses a Tensor an operation produced.** In-place assignment
  was already documented as non-differentiable and not recorded, but the guard was a
  docstring. A reverse closure reads its inputs' `data` when it runs rather than when
  it was built, so mutating a node after the forward pass takes the gradient at values
  that pass never saw — correct shapes, no error, wrong number. It now raises
  `RuntimeError` naming the operation that produced the Tensor and pointing at `where`,
  `masked_fill`, and `concat`, which are the differentiable spellings. Filling a leaf —
  an input buffer or a parameter — is unaffected, as is assignment into a Tensor built
  under `no_grad`, which has no tape to invalidate. The guard is deliberately partial:
  a leaf that has already been consumed looks exactly like a fresh one, since a Tensor
  knows its children and not its consumers, and catching that would need a version
  counter on every Tensor and a stamp in every closure.
- **`examples/char_rnn.ipynb` calls `loss.free_graph()`** where it used to call
  `gc.collect()` every fourth step, and the paragraph explaining the cadence is now a
  paragraph explaining the call. Collecting *on top* of `free_graph` measures 82.1
  ms/step at 918 MB against 81.0 and 860, so there was nothing left for it to find and
  the cadence went rather than being kept alongside. Re-executed; the losses and the
  sampled text are unchanged to the digit, the wall clock moved from 3.5 to 3.6 minutes.

## [0.1.0] - 2026-08-12

First release. There is no earlier published version, so everything below is new — this
entry describes the library as it stands rather than a set of changes against anything.
Development ran from 2024-08-29 to 2026-08-12; `READ_FILES.md` has the commit-level
history, and `PROJECT_REVIEW.md` records the two review passes that shaped it.

### Added

- **Automatic differentiation.** A define-by-run tape: each output `Tensor` records the
  tensors it was computed from and a closure that pushes its gradient into them. The
  graph is ordinary Python control flow, so a data-dependent loop or an early exit needs
  no special-cased operator. `backward()` orders the graph with an explicit stack rather
  than recursion, so an unrolled recurrence hundreds of steps deep differentiates
  without touching the recursion limit, and it refuses a non-scalar output without an
  explicit `gradient=` rather than silently differentiating `sum(output)` instead.
- **`Tensor`.** NumPy-backed, with operators, comparisons, broadcasting, dtype
  preservation, `detach()`, `requires_grad`, and a separate `trainable` flag that
  freezing sets. Gradients accumulate across passes, so micro-batching is calling
  `backward()` twice. `__array_ufunc__ = None` keeps NumPy from winning operator
  dispatch and silently coercing a `Tensor` into a 0-d object array.
- **Autodiff controls.** `no_grad()`, `enable_grad()`, and `set_grad_enabled()`, gated
  in exactly two places inside `Tensor` so no operation checks the mode itself.
- **Differentiable indexing and shape ops.** Slicing, integer-array and boolean-mask
  gathering, `concat` / `stack` / `split`, `where` / `masked_fill`, `reshape`, and
  `transpose`. Repeated indices scatter-add rather than assign, so a token that appears
  many times in a batch trains as often as it appeared.
- **Module tree.** `Module` owns its parameters and auto-registers child modules on
  attribute assignment; `named_parameters()`, `parameter_groups()`, `state_dict()` /
  `load_state_dict()`, `save` / `load`, `train()` / `eval()`, `freeze()` / `unfreeze()`,
  and `num_parameters()` all walk it recursively. Assigning a plain list, tuple, or dict
  of Modules raises and names the wrapper to use, because the silent version is layers
  that receive gradients and are never stepped by an optimizer.
- **Containers.** `Sequential` (itself a `Module`, so containers nest), `ModuleList`,
  and `ModuleDict`.
- **Layers.** `Linear`, `Conv2d` (im2col plus one gemm), `MaxPool2d`, `AvgPool2d`,
  `Dropout`, `LayerNorm`, `BatchNorm1d`, `BatchNorm2d`, `Embedding`, `RNNCell`,
  `LSTMCell`, `Flatten`, and a generic `Activation` wrapper. Layers build their
  parameters lazily on the first forward pass, so shapes can be inferred from real data.
- **Activations.** Identity, ReLU (with a leaky variant via `alpha`), Sigmoid, Tanh,
  Softmax, LogSoftmax, ELU, SELU, GELU (exact and tanh), SiLU/Swish, SoftPlus, Affine,
  and PReLU — a learnable activation, and therefore a `Module`, so its slope reaches the
  optimizer. Each is checked to stay finite far past where `exp` overflows in float64.
- **Losses.** Binary and categorical cross-entropy in both logits and probability forms,
  sparse categorical cross-entropy, mean squared error, mean absolute error, and
  Huber/SmoothL1, each with `reduction='mean' | 'sum' | 'none'`. The logits forms are
  fused: they take the gradient with respect to the logits directly, which is both the
  correct expression and the numerically stable one.
- **Optimizers.** SGD (momentum, weight decay, Nesterov), Adam, AdamW, RMSprop, Adagrad,
  and Adadelta, plus `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, `clip_grad_norm`,
  and `clip_grad_value`. An optimizer takes the model and reads its parameter groups
  live, so one constructed before the first forward pass still sees lazily built
  parameters.
- **Initializers.** Zeros, ones, constant, random uniform and normal, and the Xavier,
  He, and LeCun variants. Fan-in and fan-out are read from the weight layout, so a
  convolution kernel is scaled by its receptive field rather than by its output-channel
  count.
- **Reproducibility.** One module-level `numpy.random.Generator` behind both weight
  initialization and `Dropout`, so a single `set_seed(n)` makes a whole run reproducible.
- **`pynn.verify`.** A shipped self-verification subpackage — not a test helper — that
  runs against an installed copy with no pytest present: 209 numerical gradient checks,
  107 behavioural invariants, and 24 numerical-stability checks. `check_gradients` is
  public API for verifying your own operations. Runnable as `python -m pynn.verify`,
  which exits non-zero on failure.
- **Optional Numba acceleration.** The `numba` extra compiles `col2im`, the scatter in
  the reverse pass of `conv2d` and the pooling layers and the only Python-level loop
  left in the library, for about 1.27x end-to-end on a small CNN. Results are identical
  either way.
- **Documentation.** `README.md`, `USAGE.md` (every command and what a healthy result
  looks like), `docs/DESIGN.md` (why the library is built the way it is),
  `CONTRIBUTING.md` (a how-to-add-a-layer walkthrough), `READ_FILES.md` (generated
  commit history), and an executed MNIST notebook that renders on GitHub.
- **Packaging and tooling.** A PEP 561 `py.typed` marker so consumers see the
  annotations; `pynn.__version__` as the single source of the version, which
  `pyproject.toml` reads back; extras for every dependency group; ruff and mypy pinned
  exactly; a `.pre-commit-config.yaml` pinned to the same versions; and CI across Python
  3.10–3.14 holding line coverage to a 95% floor.

[Unreleased]: https://github.com/tarickali/pynn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tarickali/pynn/releases/tag/v0.1.0

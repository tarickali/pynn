# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the public API may change between minor versions.

## [Unreleased]

### Added

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
- **A tape figure**, `docs/tape.svg`, at the top of the README and in `docs/DESIGN.md`
  §2 — the fifteen nodes a two-layer MLP and a squared-error loss actually leave behind.
  `python scripts/generate_tape_figure.py` regenerates it; only that script needs
  Graphviz, and it writes the DOT either way.

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

# PyNN — Design Notes

Why the library is built the way it is, what the hard parts were, and what was left out
on purpose. Written for someone reading the code for the first time — or asking about it
in an interview.

---

## 1. Define-by-run, not a static graph

There are two ways to get gradients out of a numerical program. A **static graph** asks
you to describe the computation first (TensorFlow 1, Theano), compiles it, and then runs
it with data. A **define-by-run** tape (PyTorch, autograd, JAX's `grad` on traced
functions) records the operations as they actually execute, so the graph is a byproduct
of the forward pass.

PyNN is define-by-run, for one reason that matters more than the rest: **the graph is
ordinary Python control flow**. A conditional, a loop whose trip count depends on the
data, an early exit — all of them work without a special-cased operator, because the
tape only ever records what really ran. Lazy shape inference (`Linear(16)`, below) falls
out of the same property: the layer can wait until it sees a real input because there is
no separate compile step that needs the shape earlier.

The cost is that there is no whole-graph view to optimize. There is no operator fusion,
no dead-node elimination, no memory planning. Every operation allocates its output and
its closure. That trade is visible in the benchmarks: PyNN is within about 2x of PyTorch
on a matmul-dominated MLP, where BLAS does the work for both, and roughly an order of
magnitude off on a CNN, where PyTorch fuses convolution kernels across threads and PyNN
does not.

## 2. The tape is closures, not an operator registry

Most autodiff libraries define an `Op` class hierarchy: each operation is a class with a
`forward` and a `backward` method, and the graph is a list of `Op` instances with their
saved inputs. PyNN instead records, on each output `Tensor`:

- `children` — the tensors it was computed from, and
- `reverse` — a **closure** that pushes this tensor's gradient into those children.

An operation is therefore just a function:

```python
def tanh(x: Tensor) -> Tensor:
    array = x.data
    data = np.tanh(array)
    output = Tensor(data)
    output.add_children((x,))

    def reverse():
        x.grad += (1 - data**2) * output.grad

    output.forward = "tanh"
    output.reverse = reverse
    return output
```

The closure captures exactly the intermediates the backward pass needs — here `data`,
the forward result, because `d/dx tanh(x) = 1 - tanh(x)²`. Nothing has to be explicitly
declared as "saved for backward"; Python's closure capture *is* the save list. The whole
of `pynn/functional/activations.py` is nine functions in this shape, and a new operation
is one function rather than a class plus a registration.

What it costs: no operator metadata to introspect, no way to serialize a graph, and one
Python closure object allocated per operation. For a library whose point is to be
readable, that is the right side of the trade.

### `grad +=`, never `grad =`

The single most important line in the file above is `x.grad += ...`.

A tensor consumed by more than one downstream operation receives one gradient
contribution per consumer, and they must be **summed**. Assigning keeps only the last
one. The reason this bug survives for a long time is that it is invisible in a plain
feedforward network: an unbranched chain gives every tensor exactly one consumer, so
overwriting and accumulating agree exactly. It surfaces the moment there is a residual
connection, a tied weight, an auxiliary loss, or a regularization term — which is to say,
the moment the model gets interesting.

This is why `pynn.verify` checks **every unary and binary operation twice**: once alone,
and once with its input also feeding a second consumer. The single-consumer form is
exactly correct either way and proves nothing.

## 3. Backward: an iterative topological sort

`Tensor.backward()` does three things.

**It refuses an ambiguous seed.** The gradient of a non-scalar output is not defined
without saying *of what*. Seeding all-ones silently differentiates `sum(output)` instead,
which is a different function than the caller asked for. So a non-scalar output requires
an explicit `gradient=`.

**It orders the graph iteratively.** The reverse pass has to visit a node only after
every consumer of it has run, which is a reverse topological order. The natural
implementation is a recursive depth-first search, and it blows the Python stack at a few
thousand nodes — an unrolled recurrence or a long residual stack reaches that easily. So
the traversal keeps an explicit stack of `(tensor, next-child-index)` frames:

```python
stack = [(self, 0)]
while stack:
    tensor, child_index = stack.pop()
    if child_index < len(tensor.children):
        stack.append((tensor, child_index + 1))     # resume here later
        child = tensor.children[child_index]
        if child not in visited:
            visited.add(child)
            stack.append((child, 0))
    else:
        order.append(tensor)                        # all children done
```

A 5000-node chain is a check in the shipped verification suite.

**It accumulates rather than resets.** `backward()` adds into whatever gradients are
already there and never zeroes them, so gradient accumulation over micro-batches is just
calling it twice. The corollary is that a training loop *must* call `zero_grad()`
between steps.

## 4. Broadcasting, backwards

Broadcasting is where from-scratch autodiff projects usually stop being correct, and it
is worth being precise about why.

When NumPy broadcasts an operand, it implicitly **copies** that operand along the
broadcast axes. `X @ W + b` with `b` of shape `(out,)` and a batch of 32 is really `b`
copied 32 times. The derivative of a copy is a sum, so the reverse pass must sum the
incoming gradient over exactly the axes that were broadcast — otherwise the bias
gradient is 32 times too small, and training still works, just worse.

`pynn/core/utils.py::unbroadcast` does this in two steps, mirroring NumPy's two
broadcasting rules:

1. Sum away the leading axes the operand never had (`(4, 3)` gradient → `(3,)` operand).
2. Sum, keeping dimensions, the axes where the operand had length 1
   (`(4, 3)` gradient → `(1, 3)` operand).

The subtle part is *what* gets reduced. The original implementation expanded the
operand's **already accumulated** `.grad` up to the broadcast shape, added, and summed
back down — which multiplies every previously accumulated value by the broadcast factor.
Two backward passes before a step gave `[4, 4]` and then `[20, 20]` instead of `[8, 8]`.
The rule is: **reduce the incoming gradient, then add.** Never touch what is already
there.

`matmul` needs its own version, because `np.matmul` is not a plain elementwise
broadcast: a 1-D operand is promoted to a matrix for the multiply and the promoted axis
is then dropped from the result, while leading axes broadcast normally.
`matrix_multiply_gradients` promotes both operands to at least 2-D, applies the matrix
rule over the last two axes, and then undoes each adjustment in the right way — a
promoted axis is *squeezed* (it was never real), a broadcast batch axis is *summed*.

## 5. Why softmax and cross-entropy are fused

Composing a `Softmax` layer with a cross-entropy loss is the obvious design and it is
wrong in a way that is easy to miss.

For `L = -Σ y·log(softmax(z)) / B`, the famous gradient is

```
dL/dz = (softmax(z) - y) / B
```

That expression is the gradient **with respect to the logits `z`** — it already has the
softmax Jacobian folded into it. If the loss writes it onto a separate softmax node's
output, that node's own `reverse` then applies the softmax Jacobian a *second* time to a
quantity that has already been through it. The result is off by roughly 3x in magnitude
and, for some entries, has the wrong sign. Training still descends, because the wrong
gradient is positively correlated with the right one — the original MNIST example reached
94% instead of the 98% it reaches now.

So `categorical_crossentropy(..., logits=True)` computes the softmax internally, takes
the gradient with respect to the logits directly, and records only the logits tensor as
its child. The `logits=False` branch is a genuinely different function and gets the
genuinely different gradient, `-y / (p·B)`.

Fusing buys numerical stability too. `binary_crossentropy` from logits uses

```
max(z, 0) - z·y + log1p(exp(-|z|))
```

which is algebraically `-[y·log σ(z) + (1-y)·log(1-σ(z))]` but never evaluates `log(0)`
and never overflows `exp`.

The same reasoning drives the fused reverse passes in `layer_norm` and `batch_norm`. The
mean and the variance depend on *every* element being normalized, so the naive composed
graph re-derives that dependency once per element. Writing the reverse out by hand —

```
dx = (dx̂ - mean(dx̂) - x̂ · mean(dx̂ · x̂)) / std
```

— is one pass over the data instead of N.

Note that `softmax` itself, used standalone, keeps the **full Jacobian-vector product**
rather than assuming a cross-entropy downstream. It is a general-purpose operation, and
it is correct for any axis and any consumer.

## 6. The Module tree

`Module` is the only container abstraction. It owns two things:

- `parameters` — a `dict[str, Tensor]` of its **own** parameters,
- `_modules` — a `dict[str, Module]` of its children, populated automatically when a
  `Module` is assigned to an attribute.

Every collective operation is a recursive walk of that tree: `named_parameters`,
`parameter_groups`, `named_buffers`, `state_dict`, `zero_grad`, `train`/`eval`,
`freeze`/`unfreeze`, `num_parameters`.

The auto-registration is a small `__setattr__` override. It matters because the
alternative failure mode is silent:

```python
class Block(Module):
    def __init__(self, width):
        super().__init__()
        self.first = Linear(width, activation="relu")   # registered here
        self.second = Linear(width)
```

Without it, `Block`'s layers would run in the forward pass, receive gradients, and never
be handed to an optimizer — a model that trains its last layer and nothing else, with no
error anywhere.

### Containers, and the assignment that used to fail quietly

Auto-registration keys on `isinstance(value, Module)`, which leaves one hole: a plain
list.

```python
self.blocks = [Linear(64), Linear(64)]     # not a Module — not registered
```

Those layers run in the forward pass and receive gradients. They are simply absent from
`named_parameters`, so no optimizer ever steps them and no checkpoint ever saves them.
The model trains, the loss falls — the layers that *are* registered take up the slack —
and the ones in the list sit at their initial weights forever. PyTorch has the same hole
and answers it with `ModuleList`; a user who forgets gets no warning.

`ModuleList` and `ModuleDict` are the containers. They register their contents, so
everything recursive reaches them:

```python
self.blocks = ModuleList(Linear(64, activation="relu") for _ in range(depth))
# -> blocks.0.W, blocks.0.b, blocks.1.W, ...
```

Neither has a `forward`: the point is to hold modules whose wiring the enclosing module
decides, and `Sequential` already covers "apply these in order". Calling one raises
rather than guessing.

The second half is the part PyTorch does not do. `Module.__setattr__` **refuses** a
plain list, tuple, or dict that contains a `Module`, naming the wrapper to use. An
assignment that would have silently cost you a layer is now a `TypeError` at the moment
you write it. That trade — a false positive is a one-line fix, a false negative is a
model that never trains a third of itself — is the same one the rest of this library
makes everywhere.

Indices in a `ModuleList` are positions, so inserting renumbers the paths after it and
a checkpoint taken before the insert will not load after one. `ModuleDict` keys become
path segments, so a key containing `.` is refused for the same reason.

### Why `Sequential` had to become a `Module`

`Sequential` used to subclass a separate `Model` base whose `parameters` was a
`list[dict]`, while `Module.parameters` was a `dict`. Nesting one inside another
type-checked, ran the forward pass correctly, and then died inside the optimizer with
`AttributeError: 'list' object has no attribute 'items'`.

Collapsing the two hierarchies into one is the structural change everything else depends
on. `Dropout`, the normalization layers, `state_dict`, `train`/`eval` and per-branch
freezing are all one-liners *given* a tree that composes, and are all special cases
without one. `Model` no longer had anything to do and was deleted rather than left
behind as a base class nothing implements.

### Parameter groups, and why they are live

An optimizer takes the model:

```python
optimizer = SGD(model, learning_rate=0.01)
```

and normalizes it to `model.parameter_groups()` — one group per module in the tree,
holding **the modules' live dictionaries, not copies**. That is deliberate. Layers build
their parameters lazily on the first forward pass, so an optimizer constructed the usual
way (before training starts) is looking at dictionaries that are still empty. Because the
references are live, the parameters simply appear.

Modules with no parameters contribute an empty group. That keeps the list index-aligned
with each optimizer's per-group state as those dictionaries fill in.

Passing the old `model.parameters` — now a plain dict — raises a `TypeError` naming the
fix, rather than iterating a dict's keys and silently stepping nothing.

### Two flags, deliberately

`Tensor` carries both `requires_grad` and `trainable`, and they are not the same thing:

| Flag | Means | Set by |
|---|---|---|
| `requires_grad` | connected to the graph; `backward` can reach it | `no_grad`, `detach` |
| `trainable` | an optimizer may step it | `Module.freeze()` |

A frozen parameter still *receives* a gradient during the backward pass — freezing is not
detaching. It is `Optimizer.trainable_parameters()` that filters it out of the update.
Keeping these separate is what lets you freeze a pretrained trunk while gradients still
flow *through* it to layers below.

### Buffers

Batch normalization's running mean and variance belong to the module but must never be
stepped by an optimizer. They live in `_buffers`, are excluded from `parameter_groups`,
and are included in `state_dict` — because a batch-normalized model that loses its
running statistics evaluates differently after a checkpoint round trip. The weights alone
are not the whole model.

## 7. Training and evaluation modes

`Dropout` is the canonical reason a model needs modes: at training time it zeros a
random fraction of its input; at evaluation it must be exactly the identity. `BatchNorm`
is the other: it normalizes by the current batch's statistics while training and by
running estimates afterwards, so that inference on a single example is well-defined at
all.

`model.eval()` walks the tree and sets `training = False` on every module; `model.train()`
sets it back. Both return `self`, so they chain. A branch can be switched on its own
(`model[0].eval()`), which is what fine-tuning with a frozen, batch-normalized trunk
needs.

**Inverted dropout** is what makes the eval path free: the surviving activations are
divided by the keep probability during training, so the expected value already matches
the input and evaluation needs no compensating factor.

## 8. Turning the tape off

Inference does not need a graph. Building one is wasted work per batch, and it is worse
than waste when the outputs are kept: every `reverse` closure holds the forward pass's
intermediate arrays, so a validation loop that collects predictions retains the entire
graph of every batch it has seen.

```python
model.eval()
with no_grad():
    predictions = [model(Tensor(batch)) for batch in batches]
```

The implementation deliberately has exactly **two gates**, both inside `Tensor`, so that
no operation has to check the mode itself:

- `add_children` drops the edges and marks the output as not requiring gradients;
- the `reverse` **setter** discards the closure instead of storing it.

The second is the one that actually frees memory, and it is why `reverse` is a property
rather than a plain attribute. The alternative — an explicit `record(forward, children,
reverse)` method called by every operation — is arguably cleaner, and would mean touching
all ~25 operation sites; the property keeps the gate in one place.

`backward()` on a tensor produced under `no_grad`, or on one that has been `detach`ed,
raises rather than quietly returning zeros.

## 9. Numerical stability

Four places where the obvious formula is wrong at the edges:

| Function | Naive form | Problem | Used instead |
|---|---|---|---|
| `sigmoid` | `1 / (1 + exp(-x))` | overflows for large negative `x` | branch on the sign, `pynn/core/numeric.py` |
| `softplus` | `log(1 + exp(x))` | `softplus(800)` → `inf` | `np.logaddexp(0, x)` |
| `log` | `log(x + EPSILON)` | `EPSILON ≈ 2.2e-16` is far too small to tame `log(0)`, and it biases the result everywhere | clamp the input |
| `elu` / `selu` | `alpha * (exp(x) - 1)` | catastrophic cancellation near 0 | `np.expm1` on a clamped input |
| `softmax` | `exp(x) / sum(exp(x))` | overflows | subtract the row max first |

`pynn.verify.check_stability` asserts every activation and loss stays finite at `|x|` up
to 1000, with NumPy floating-point warnings promoted to errors so a silent overflow fails
the check rather than printing a warning nobody reads.

## 10. Convolution: im2col and one matmul

The first implementation was a Python double loop over output positions. The current one
unrolls every input patch into a row of a matrix and does a single `gemm`:

```
cols  : (N·out_h·out_w, in_ch·kh·kw)      one row per output position
kernel: (in_ch·kh·kw, out_ch)
                    ↓
out   : (N·out_h·out_w, out_ch)   →   (N, out_ch, out_h, out_w)
```

`im2col` builds `cols` as a **strided view** (`np.lib.stride_tricks.as_strided`) rather
than a Python loop, so the only work that scales with output size is the matmul itself.
It is then made contiguous, deliberately: the view aliases the input, and the reverse
pass reads it after the forward pass has returned.

The reverse pass is two matrix products and a scatter:

- `dK = dOutᵀ @ cols`
- `dX = col2im(dOut @ Kᵀ)`

`col2im` is the adjoint of `im2col`: it scatters rows back into the image, **adding**
where windows overlapped, because a pixel read by several windows contributes to several
outputs. The test for it is the adjoint identity `⟨im2col(x), c⟩ == ⟨x, col2im(c)⟩`,
which is exactly the statement that one is the reverse-mode of the other.

The pooling layers reuse the same machinery, folding channels into the batch axis so that
each row of `cols` is one window of one channel. Max pooling's reverse scatters each
output's gradient to the single position that won its window; average pooling spreads it
evenly. Max pooling pads with `-inf` rather than zero, or a zero pad would beat every
negative input.

## 11. Verification, not just tests

`pynn/verify/` is a shipped subpackage, not a test helper. It runs against an installed
copy of PyNN without pytest (`python -m pynn.verify`), and `tests/` drives it as well so
it cannot rot.

The reason it exists separately: **an autodiff library can be wrong while looking
healthy.** Training loss falls when a gradient is scaled by the batch size. A momentum
buffer with a dropped gradient term still descends, just more slowly. An activation that
overflows to `inf` only poisons a run once the inputs grow large enough. None of these
fail a shape test, and none of them fail a "did the loss go down" test.

Three suites:

- **`check_all_gradients`** — 151 checks. Every operator across broadcasting shape
  combinations, every activation at normal *and* overflow-scale magnitudes, every loss in
  both its logits and probability forms, every module function across strides and
  paddings, and nine graph topologies where a tensor has more than one consumer. Central
  differences, `(f(x+h) - f(x-h)) / 2h`, on float64, with relative error scaled by the sum
  of magnitudes so the check stays meaningful for near-zero gradients.
- **`check_stability`** — finiteness far past where `exp` overflows.
- **`check_invariants`** — behavioral properties of the tape, the module tree, the
  optimizers, and the public factories. Each optimizer is compared against a **closed-form
  transcription of its published update rule**, not against "the loss went down".

The suite was mutation-tested to confirm it is not vacuously green: reintroducing each of
the original bug classes produces failures (reverting accumulation in `core/math.py` → 16
failures; in `functional/activations.py` → 12; removing `unbroadcast` → 39; breaking the
SGD momentum term → 3; restoring the naive sigmoid → 2).

`check_gradients` is public API, so users can verify their own layers:

```python
result = check_gradients(lambda ts: my_loss(ts[0]), [Tensor(x)])
assert result.passed, result
```

## 12. Fan-in is a property of the layout, not of `shape[0]`

The scale a variance-scaling initializer picks is a function of how many inputs and
outputs a weight connects. `he_normal` originally computed it as `sqrt(2 / shape[0])`,
which is fan-in only for a 2-D `(in_features, out_features)` matrix. A `Conv2d` kernel is
shaped `(out_channels, in_channels, kh, kw)`, so that reads the *output* channel count
and drops the receptive field entirely: `Conv2d(16, 32, 3)` came out at a standard
deviation of 0.25 where it should be `sqrt(2 / (16·3·3)) = 0.118`.

Nothing failed. The weights were a plausible size, shapes were right, and every gradient
check passed — an initializer's *scale* is not something a gradient check can see. What
it did was compound with depth: activations grew about 2x per layer, initial logits
reached a standard deviation of 7.8, the first loss was 15.2 rather than `ln(10) ≈ 2.3`,
and a small CNN collapsed into predicting a constant at a learning rate a correctly
initialized one handles comfortably.

`fans(shape)` branches on rank, because the library genuinely has two layouts — a
`Linear` weight is `(in, out)` while a `Conv2d` kernel follows PyTorch's `(out, in, kh,
kw)`. This is the kind of bug worth knowing about: no test fails, no exception is raised,
and the symptom shows up several layers away as "the model won't train".

## 13. Some smaller decisions

**`__array_ufunc__ = None` on `Tensor`.** Without it, NumPy wins the dispatch for
`array + tensor` and coerces the Tensor to a 0-d object array. The result *looks* like an
array, carries no gradient, and every subsequent operation on it is Python-level object
arithmetic. Declining to participate in ufunc dispatch (NEP 13) makes NumPy return
`NotImplemented`, which is what sends Python to `Tensor.__radd__`.

**Comparisons return Tensors, and `__bool__` raises.** `a == b` is elementwise, matching
NumPy and PyTorch, which breaks the `object.__eq__ → bool` contract. Every array library
makes that trade. The consequence is that `if a == b:` cannot mean what it looks like, so
`__bool__` raises for any Tensor with more than one element rather than silently
returning `True` for every non-empty one. Identity semantics survive via `__hash__`,
which the backward pass relies on for its visited set.

**dtype is preserved, mostly.** A float32 array stays float32, and its gradient is
float32 — a gradient is real-valued, so anything non-floating (integers, the booleans a
comparison produces) gets float64. The boundary: the initializers produce float64, and a
Python scalar promoted to a Tensor is float64, so mixing either with a float32 tensor
promotes per NumPy's rules. A fully float32 model would need dtype-aware initializers,
which is not implemented.

**`get_batches` shuffles by default.** Iterating a fixed order every epoch is not
stochastic gradient descent — consecutive steps stay correlated and the gradient noise
that lets SGD escape shallow minima is absent. It yields rather than materializing, so an
epoch never holds a second copy of the dataset.

**One random source.** `pynn.core.random` holds the generator that both weight
initialization and `Dropout` draw from, so a single `set_seed(n)` makes a whole run
reproducible. It is a `numpy.random.Generator`, not the legacy global `numpy.random`,
which cannot be seeded independently of anything else in the process.

## 14. One compiled function, and why only one

The library ships a `numba` extra. It compiles exactly one function, and the reasoning
for *which* one is the interesting part.

The obvious thing to compile is the elementwise primitives — `add`, `multiply`, and so
on. That was the original design and it was a mistake. Those bodies are a single NumPy
call, which is already a vectorized C kernel; there is no Python-level loop for a JIT to
remove. What compiling them adds is per-call dispatch across the Python/JIT boundary
(the whole cost, when the body is one C call), a fresh compile per dtype and rank
combination, and no fusion across calls, since each is invoked separately by the tape.
Measured: **1.9x slower** than plain NumPy on a 2000×2000 add, and a test suite that went
from 1.7s to 17s.

The place that *does* have a Python loop is `col2im`, the scatter in the reverse pass of
`conv2d` and the pooling layers. It cannot be one NumPy call: overlapping windows
contribute to the same input pixel, so the scatter must accumulate, and `+=` on
overlapping slices is not something NumPy vectorizes. Profiling a CNN training step puts
**42% of the time** there.

So the scatter exists twice — a NumPy block loop, and the same logic as explicit scalar
loops that the JIT compiles. In pure Python the scalar version is far slower, which is
the point: compiling removes exactly the loop overhead that made it slow. Measured on a
small CNN:

| | ms/step | vs PyTorch |
|---|---|---|
| NumPy scatter | 75.5 | 9.3x |
| Compiled scatter | 59.3 | 6.5x |

A 1.27x end-to-end speedup from compiling one function. An MLP is unchanged, since it
never touches `col2im`.

Two implementations of one algorithm is a correctness risk, so `tests/utils/array_test.py`
asserts they agree across geometries with and without overlap, and `col2im`'s adjoint
identity is checked either way. The extra stays optional: `llvmlite` is 130 MB, which
should be a choice rather than a condition of installing a NumPy library.

The general lesson is the one worth taking from this: a JIT does not make code fast, it
removes interpreter overhead. Where there is no interpreter overhead to remove — a thin
wrapper over a C kernel — it can only add cost. Profile first, then compile the thing the
profile names.

## 15. What was left out, and why

**Differentiable indexing, `concat`, `stack`, `split`.** `Tensor.__getitem__` returns a
raw NumPy array and drops the graph. Everything currently shipped works on whole tensors,
so nothing needs it yet; an attention mechanism or a `Embedding` layer would.

**Recurrent layers.** Backprop-through-time is the most interesting thing missing. The
tape already handles it — the iterative topological sort exists precisely so that
unrolled recurrences do not blow the stack — but there is no `RNNCell` to exercise it.

**Operator fusion, in-place operations, memory pooling.** These are where the remaining
gap to PyTorch lives, and they are also where the code would stop being readable, which
is the point of the project.

**A graph visualizer.** `Tensor.to_dot()` would be cheap and would make the tape
inspectable. Not written yet.

**Operator fusion beyond `col2im`.** The JIT covers the one Python loop that was
worth compiling (§14). Everything else is already a single NumPy call, and fusing
*across* calls would mean an expression compiler, which is where the code would stop
being readable.

---

## Reading order

| Start here | For |
|---|---|
| `pynn/core/tensor.py` | The tape: `add_children`, `reverse`, `backward`, the topological sort |
| `pynn/core/utils.py` | `unbroadcast` and `matrix_multiply_gradients` |
| `pynn/functional/activations.py` | The closure pattern, nine times |
| `pynn/functional/losses.py` | Why the fused losses look the way they do |
| `pynn/core/module.py` | The module tree and every recursive walk over it |
| `pynn/utils/array.py` | `im2col` / `col2im` |
| `pynn/verify/gradients.py` | What is checked, and the shapes the checks take |

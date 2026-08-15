"""Behavioral invariants of the autodiff engine, optimizers, and public API.

These are the properties that are easy to break without any test going red, because
the library keeps running and training keeps roughly working: a gradient that is
overwritten instead of accumulated, a momentum buffer that decays to nothing, an
optimizer that silently ignores a flag.

Each optimizer is compared against a closed-form transcription of its published update
rule rather than against a "loss went down" assertion, since a broken momentum buffer
still descends, just more slowly.

The module-tree checks are here for the same reason: a container that loses track of a
nested child does not raise, it just reports fewer parameters, and the layers it dropped
silently never train.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np

import pynn.core.math as pmath
import pynn.functional as F
from pynn.core import Module, Tensor, concat, is_grad_enabled, no_grad, split
from pynn.functional.initializers import fans, he_normal
from pynn.nn import (
    BatchNorm1d,
    Dropout,
    Embedding,
    Linear,
    ModuleDict,
    ModuleList,
    RNNCell,
    Sequential,
)
from pynn.nn.factories import activation_factory, initializer_factory
from pynn.nn.losses import MeanSquaredError
from pynn.optim import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    CosineAnnealingLR,
    ExponentialLR,
    RMSprop,
    StepLR,
    clip_grad_norm,
)
from pynn.verify.report import CheckReport

__all__ = [
    "check_api",
    "check_autodiff",
    "check_invariants",
    "check_modules",
    "check_optimizers",
    "check_training",
]

ACTIVATION_NAMES = [
    "affine",
    "elu",
    "gelu",
    "identity",
    "log_softmax",
    "prelu",
    "relu",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "swish",
    "tanh",
]

INITIALIZER_NAMES = [
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "ones",
    "random_normal",
    "random_uniform",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]

ALL_OPTIMIZERS = [SGD, Adam, RMSprop, Adagrad, Adadelta]


def check_autodiff() -> CheckReport:
    """Verify graph construction, gradient accumulation, and backward's contract."""

    report = CheckReport(name="autodiff")

    # Indexing has to stay on the tape. Returning a raw array drops the graph with no
    # error at all, so a model that slices a sequence trains nothing upstream of it.
    tensor = Tensor(np.arange(12.0).reshape(3, 4))
    try:
        row = tensor[0]
        tensor[1] = np.zeros(4)
        report.add(
            "Tensor supports indexing and assignment",
            isinstance(row, Tensor)
            and row.data.tolist() == [0.0, 1.0, 2.0, 3.0]
            and tensor[1].data.tolist() == [0.0] * 4,
        )
    except Exception as error:
        report.add(
            "Tensor supports indexing and assignment",
            False,
            f"raised {type(error).__name__}: {error}",
        )

    indexed = Tensor(np.arange(6.0).reshape(3, 2))
    pmath.sum(indexed[[0, 0, 2]]).backward()
    report.add(
        "indexing scatters its gradient back, accumulating on repeats",
        bool(np.array_equal(indexed.grad, [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]])),
        f"{indexed.grad.tolist()} (row 0 was read twice)",
    )

    # concat has many inputs and one output; split has one input and many outputs.
    # Both are where a reverse pass sends a gradient to the wrong place quietly.
    head, tail = Tensor(np.ones((2, 3))), Tensor(np.zeros((2, 4)))
    pmath.sum(concat([head, tail], axis=1) * np.arange(7.0)).backward()
    report.add(
        "concat splits its gradient back to each input",
        bool(
            np.array_equal(head.grad, np.tile(np.arange(3.0), (2, 1)))
            and np.array_equal(tail.grad, np.tile(np.arange(3.0, 7.0), (2, 1)))
        ),
    )

    whole = Tensor(np.ones((6, 2)))
    top, _, bottom = split(whole, 3)
    (pmath.sum(top) + pmath.sum(bottom) * 2.0).backward()
    report.add(
        "split routes each piece's gradient to its own slice",
        bool(np.array_equal(whole.grad[:, 0], [1.0, 1.0, 0.0, 0.0, 2.0, 2.0])),
        f"{whole.grad[:, 0].tolist()}",
    )

    shared = Tensor(np.ones((2, 2)))
    pmath.sum(concat([shared, shared], axis=0)).backward()
    report.add(
        "a tensor concatenated with itself receives both gradients",
        bool(np.allclose(shared.grad, 2.0)),
        f"max {shared.grad.max()}",
    )

    # Transposing must stay on the tape. Returning a detached Tensor here yields a
    # zero gradient with no error at all.
    x = Tensor(np.random.default_rng(0).standard_normal((3, 4)))
    y = Tensor(np.random.default_rng(1).standard_normal((3, 2)))
    pmath.sum(x.T @ y).backward()
    report.add(
        "transpose participates in the graph",
        bool(np.any(x.grad != 0.0)),
        f"max |grad| = {np.abs(x.grad).max():.3g}",
    )

    # Two backward passes without zero_grad must double the gradient. Reducing a
    # broadcast operand by mutating its accumulated gradient instead of the incoming
    # one scales the old value by the batch size.
    rng = np.random.default_rng(11)
    X = Tensor(rng.standard_normal((4, 3)))
    W = Tensor(rng.standard_normal((3, 2)))
    b = Tensor(rng.standard_normal((2,)))
    pmath.sum(X @ W + b).backward()
    first = b.grad.copy()
    pmath.sum(X @ W + b).backward()
    report.add(
        "backward accumulates across calls",
        bool(np.allclose(b.grad, 2 * first)),
        f"{first.tolist()} then {b.grad.tolist()}",
    )

    b.zero_grad()
    report.add("zero_grad clears gradients", bool(np.all(b.grad == 0.0)))

    # A tensor feeding two consumers must sum both contributions.
    shared = Tensor(np.array([[1.0, -2.0]]))
    (pmath.sum(shared) + pmath.sum(shared * 2.0)).backward()
    report.add(
        "gradients from multiple consumers are summed",
        bool(np.allclose(shared.grad, 3.0)),
        f"grad={shared.grad.tolist()} (expected 3.0)",
    )

    # The topological sort must not recurse, or deep graphs blow the Python stack.
    deep = Tensor(np.array([1.0]))
    accumulator = deep
    for _ in range(5000):
        accumulator = accumulator + 1.0
    try:
        accumulator.backward()
        report.add(
            "backward handles a 5000-node chain",
            bool(np.allclose(deep.grad, 1.0)),
        )
    except RecursionError:
        report.add("backward handles a 5000-node chain", False, "RecursionError")

    # Seeding a non-scalar output with ones silently differentiates a different
    # function than the caller asked for, so it must be refused.
    non_scalar = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]])) * 2.0
    try:
        non_scalar.backward()
        report.add("backward rejects a non-scalar output", False, "no error raised")
    except ValueError:
        report.add("backward rejects a non-scalar output", True)

    explicit = Tensor(np.array([[1.0, 2.0]]))
    (explicit * 2.0).backward(gradient=np.ones((1, 2)))
    report.add(
        "backward accepts an explicit seed gradient",
        bool(np.allclose(explicit.grad, 2.0)),
    )

    try:
        (Tensor(np.array([1.0, 2.0])) * 2.0).backward(gradient=np.ones((3,)))
        report.add("backward rejects a mismatched seed", False, "no error raised")
    except ValueError:
        report.add("backward rejects a mismatched seed", True)

    # NumPy would otherwise handle `array + tensor` itself, coercing the Tensor to a
    # 0-d object array: the result looks like an array, has no gradient, and every
    # operation on it afterwards is Python-level object arithmetic.
    left = np.full((2, 3), 3.0)
    right = Tensor(np.ones((2, 3)))
    product = left * right
    report.add(
        "an ndarray on the left still produces a Tensor",
        isinstance(product, Tensor) and product.dtype == np.float64,
        f"got {type(product).__name__}",
    )
    pmath.sum(product).backward()
    report.add(
        "an ndarray on the left stays on the tape",
        bool(np.allclose(right.grad, 3.0)),
    )

    # Recording is what inference does not need. Without a gate, every evaluation
    # batch builds a graph, and holding on to the outputs holds on to all of it.
    x = Tensor(np.ones((2, 3)))
    with no_grad():
        inference = pmath.sum(F.tanh(x) * 2.0)
    report.add(
        "no_grad records no children",
        inference.children == () and not inference.requires_grad,
    )
    try:
        inference.backward()
        report.add("backward rejects a tensor built under no_grad", False, "no error")
    except RuntimeError:
        report.add("backward rejects a tensor built under no_grad", True)

    report.add("grad mode is restored after the block", is_grad_enabled())

    detached = Tensor(np.ones((2, 3)))
    pmath.sum(detached.detach() * 2.0).backward()
    report.add(
        "detach stops the gradient",
        bool(np.all(detached.grad == 0.0)),
        f"max |grad| = {np.abs(detached.grad).max():.3g}",
    )

    # `no_grad` above is how the tape is never built; this is how it is given back. A
    # finished graph is a reference cycle, since every reverse closure references the
    # Tensor it belongs to, so dropping the last name pointing at a loss frees nothing.
    # `free_graph` breaks the cycles, and the property worth checking is that reference
    # counting alone then reclaims the graph — waiting for the cyclic collector is the
    # whole problem, and CPython schedules it from object counts rather than from the
    # hundreds of megabytes of arrays hanging off those objects.
    leaf = Tensor(np.ones((2, 2)))
    projected = leaf * 2.0
    finished = pmath.sum(F.tanh(projected) * projected)
    finished.backward()
    earned = leaf.grad.copy()

    # Turned off so that reference counting is what the next check observes: a
    # collection triggered by unrelated allocation would satisfy it for the wrong
    # reason.
    collecting = gc.isenabled()
    gc.disable()
    try:
        intermediate = weakref.ref(projected)
        del projected
        retained = intermediate() is not None
        finished.free_graph()
        report.add(
            "free_graph reclaims the tape without the cyclic collector",
            retained and intermediate() is None,
            f"held before free_graph: {retained}, "
            f"reclaimed after: {intermediate() is None}",
        )
    finally:
        if collecting:
            gc.enable()

    report.add(
        "free_graph empties the freed output's children",
        finished.children == (),
        f"{len(finished.children)} children remain",
    )
    # Freeing the tape must not take the gradients with it — the optimizer reads them
    # after the call, and zeros here would look exactly like a converged model.
    report.add(
        "free_graph leaves the gradients backward computed",
        bool(np.any(earned != 0.0) and np.array_equal(leaf.grad, earned)),
        f"{leaf.grad.tolist()}",
    )
    try:
        finished.backward()
        report.add(
            "backward refuses a graph free_graph released", False, "no error raised"
        )
    except RuntimeError:
        report.add("backward refuses a graph free_graph released", True)

    # A float32 input used to be upcast on the way in, and every gradient was float64
    # regardless of the data — twice the memory, silently.
    single = Tensor(np.ones((4, 3), dtype=np.float32))
    weights = Tensor(np.ones((3, 2), dtype=np.float32))
    output = single @ weights
    pmath.sum(output).backward()
    report.add(
        "float32 survives a forward and backward pass",
        single.dtype == np.float32
        and output.dtype == np.float32
        and single.grad.dtype == np.float32,
        f"data {single.dtype}, output {output.dtype}, grad {single.grad.dtype}",
    )

    # A picture of the tape is only evidence of anything if it is a picture of the
    # graph backward walks. Two independent traversals — the breadth-first walk in
    # pynn/viz.py and the topological sort here — have to reach the same tensors, and
    # the shape that separates them is a tensor with two consumers, which a walk with
    # no visited set draws twice.
    shared = Tensor(np.ones((2, 2)))
    residual = pmath.sum(F.relu(shared) + shared)
    drawn = residual.to_dot().count("[label=")
    walked = len(residual._topological_order())
    report.add(
        "to_dot draws exactly the tensors backward walks",
        drawn == walked,
        f"{drawn} drawn, {walked} on the tape",
    )

    return report


def _trajectory(optimizer_cls, steps: int = 6, gradient: float = 0.7, **kwargs):
    """Step one scalar parameter with a constant gradient, returning its values."""
    param = Tensor(np.array([1.0]))
    optimizer = optimizer_cls([{"w": param}], **kwargs)

    values = []
    for _ in range(steps):
        param.grad = np.array([gradient])
        optimizer.update()
        values.append(float(param.data[0]))
    return values


def check_optimizers() -> CheckReport:
    """Verify each optimizer against a closed-form reference implementation."""

    report = CheckReport(name="optimizers")
    steps, grad, start = 6, 0.7, 1.0

    # --- SGD, plain ------------------------------------------------------- #
    lr = 0.1
    expected, value = [], start
    for _ in range(steps):
        value -= lr * grad
        expected.append(value)
    report.add(
        "SGD matches the reference update",
        bool(np.allclose(_trajectory(SGD, learning_rate=lr), expected)),
    )

    # --- SGD with momentum ------------------------------------------------ #
    # buf = momentum * buf + (1 - dampening) * grad. Dropping the gradient term
    # makes the buffer decay to zero, so momentum silently does nothing.
    momentum = 0.9
    expected, value, buffer = [], start, None
    for _ in range(steps):
        buffer = grad if buffer is None else momentum * buffer + grad
        value -= lr * buffer
        expected.append(value)
    actual = _trajectory(SGD, learning_rate=lr, momentum=momentum)
    report.add(
        "SGD momentum matches the reference update",
        bool(np.allclose(actual, expected)),
        f"got {np.round(actual, 5).tolist()}",
    )

    step_sizes = -np.diff([start, *actual])
    report.add(
        "SGD momentum accelerates under a constant gradient",
        bool(np.all(np.diff(step_sizes) > 0)),
        f"step sizes {np.round(step_sizes, 5).tolist()}",
    )

    # --- SGD, Nesterov ---------------------------------------------------- #
    expected, value, buffer = [], start, None
    for _ in range(steps):
        buffer = grad if buffer is None else momentum * buffer + grad
        value -= lr * (grad + momentum * buffer)
        expected.append(value)
    report.add(
        "SGD Nesterov matches the reference update",
        bool(
            np.allclose(
                _trajectory(SGD, learning_rate=lr, momentum=momentum, nesterov=True),
                expected,
            )
        ),
    )

    # --- Adam ------------------------------------------------------------- #
    lr, beta_1, beta_2, eps = 0.01, 0.9, 0.999, 1e-8
    expected, value, m, v = [], start, 0.0, 0.0
    for step in range(1, steps + 1):
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad**2
        value -= lr * (m / (1 - beta_1**step)) / (np.sqrt(v / (1 - beta_2**step)) + eps)
        expected.append(value)
    report.add(
        "Adam matches the reference update",
        bool(np.allclose(_trajectory(Adam, learning_rate=lr), expected)),
    )

    # --- RMSprop ---------------------------------------------------------- #
    lr, alpha, eps = 0.01, 0.99, 1e-10
    expected, value, square_average = [], start, 0.0
    for _ in range(steps):
        square_average = alpha * square_average + (1 - alpha) * grad**2
        value -= lr * grad / (np.sqrt(square_average) + eps)
        expected.append(value)
    report.add(
        "RMSprop matches the reference update",
        bool(np.allclose(_trajectory(RMSprop, learning_rate=lr), expected)),
    )

    # --- Adagrad ---------------------------------------------------------- #
    expected, value, total = [], start, 0.0
    for _ in range(steps):
        total += grad**2
        value -= lr * grad / (np.sqrt(total) + eps)
        expected.append(value)
    report.add(
        "Adagrad matches the reference update",
        bool(np.allclose(_trajectory(Adagrad, learning_rate=lr), expected)),
    )

    # --- Adadelta --------------------------------------------------------- #
    lr, rho = 1.0, 0.9
    expected, value, average, accumulator = [], start, 0.0, 0.0
    for _ in range(steps):
        average = rho * average + (1 - rho) * grad**2
        delta = np.sqrt((accumulator + eps) / (average + eps)) * grad
        accumulator = rho * accumulator + (1 - rho) * delta**2
        value -= lr * delta
        expected.append(value)
    report.add(
        "Adadelta matches the reference update",
        bool(np.allclose(_trajectory(Adadelta, learning_rate=lr), expected)),
    )

    # --- AdamW ------------------------------------------------------------ #
    # Adam folds weight decay into the gradient, so it goes through the same
    # 1/sqrt(v) rescaling as everything else and a parameter with a large second
    # moment gets less decay. AdamW applies it to the parameter directly.
    lr, decay = 0.01, 0.1
    expected, value, m, v = [], start, 0.0, 0.0
    for step in range(1, steps + 1):
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad**2
        value -= lr * decay * value
        value -= lr * (m / (1 - beta_1**step)) / (np.sqrt(v / (1 - beta_2**step)) + eps)
        expected.append(value)
    report.add(
        "AdamW matches the decoupled reference update",
        bool(
            np.allclose(
                _trajectory(AdamW, learning_rate=lr, weight_decay=decay), expected
            )
        ),
    )
    report.add(
        "AdamW decay is not rescaled by the second moment",
        not np.allclose(
            _trajectory(Adam, learning_rate=lr, weight_decay=decay),
            _trajectory(AdamW, learning_rate=lr, weight_decay=decay),
        ),
    )

    # --- flags and shared behavior ---------------------------------------- #
    for optimizer_cls in ALL_OPTIMIZERS:
        name = optimizer_cls.__name__
        ascending = _trajectory(optimizer_cls, learning_rate=0.01, maximize=True)
        report.add(
            f"{name} honors maximize",
            bool(np.all(np.diff([start, *ascending]) > 0)),
        )

        # Layers build their parameters on the first forward pass, so the optimizer
        # is constructed against dictionaries that are still empty.
        model = Sequential([Linear(4, 3)])
        optimizer = optimizer_cls(model, learning_rate=0.1)
        X = Tensor(np.random.default_rng(0).standard_normal((5, 4)))
        loss = MeanSquaredError()(Tensor(np.zeros((5, 3))), model(X))
        model.zero_grad()
        loss.backward()
        before = model.modules[0].parameters["W"].data.copy()
        optimizer.update()
        report.add(
            f"{name} updates lazily built parameters",
            not np.allclose(before, model.modules[0].parameters["W"].data),
        )

        rng = np.random.default_rng(3)
        model = Sequential([Linear(4, 8, activation="tanh"), Linear(8, 1)])
        loss_fn = MeanSquaredError()
        optimizer = optimizer_cls(model, learning_rate=0.05)
        X = Tensor(rng.standard_normal((16, 4)))
        y = Tensor(rng.standard_normal((16, 1)))
        first_loss = float(loss_fn(y, model(X)).item())
        for _ in range(50):
            loss = loss_fn(y, model(X))
            model.zero_grad()
            loss.backward()
            optimizer.update()
        last_loss = float(loss_fn(y, model(X)).item())
        report.add(
            f"{name} reduces the loss",
            last_loss < first_loss,
            f"{first_loss:.4f} -> {last_loss:.4f}",
        )

    return report


def check_training() -> CheckReport:
    """Verify the learning-rate schedules and gradient clipping."""

    report = CheckReport(name="training")

    def rates(scheduler_cls, epochs: int = 6, **kwargs) -> list[float]:
        optimizer = SGD([{"w": Tensor(np.array([1.0]))}], learning_rate=0.1)
        scheduler = scheduler_cls(optimizer, **kwargs)
        values = [optimizer.learning_rate]
        for _ in range(epochs):
            scheduler.step()
            values.append(optimizer.learning_rate)
        return values

    report.add(
        "StepLR drops on a staircase",
        bool(
            np.allclose(
                rates(StepLR, step_size=3, gamma=0.5),
                [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025],
            )
        ),
    )
    report.add(
        "ExponentialLR decays every epoch",
        bool(
            np.allclose(
                rates(ExponentialLR, gamma=0.9), [0.1 * 0.9**e for e in range(7)]
            )
        ),
    )
    annealed = rates(CosineAnnealingLR, epochs=10, T_max=10, eta_min=0.001)
    report.add(
        "CosineAnnealingLR reaches eta_min at T_max",
        bool(np.isclose(annealed[0], 0.1) and np.isclose(annealed[-1], 0.001)),
        f"{annealed[0]:.4f} -> {annealed[-1]:.4f}",
    )

    # A schedule that computed rates nobody read would satisfy every check above.
    optimizer = SGD([{"w": Tensor(np.array([1.0]))}], learning_rate=0.1)
    parameter = optimizer.parameters[0]["w"]
    StepLR(optimizer, step_size=1, gamma=0.0).step()
    parameter.grad = np.array([0.7])
    optimizer.update()
    report.add(
        "a schedule's rate reaches the optimizer",
        bool(np.allclose(parameter.data, 1.0)),
        f"a zero learning rate left the parameter at {parameter.data.tolist()}",
    )

    # Clipping shortens the step; it must not turn it.
    module = Linear(2, 1)
    module.parameters["a"] = Tensor(np.array([3.0, 4.0]))
    module.parameters["b"] = Tensor(np.array([12.0, 0.0]))
    for name in ("a", "b"):
        module.parameters[name].grad = module.parameters[name].data.copy()
    before = [module.parameters[name].grad.copy() for name in ("a", "b")]

    reported = clip_grad_norm(module, max_norm=1.0)
    after = [module.parameters[name].grad for name in ("a", "b")]
    total = float(np.sqrt(sum(float((g**2).sum()) for g in after)))

    report.add(
        "clip_grad_norm reports the norm before clipping",
        bool(np.isclose(reported, 13.0)),
        f"reported {reported:.4f}, expected 13.0",
    )
    report.add(
        "clip_grad_norm brings the total norm to the limit",
        bool(np.isclose(total, 1.0, rtol=1e-3)),
        f"{total:.6f}",
    )
    report.add(
        "clip_grad_norm preserves the step direction",
        all(
            bool(
                np.isclose(
                    float(np.dot(original, clipped))
                    / float(np.linalg.norm(original) * np.linalg.norm(clipped)),
                    1.0,
                )
            )
            for original, clipped in zip(before, after, strict=True)
        ),
    )

    return report


def check_api() -> CheckReport:
    """Verify that every documented factory name and layer shape path works."""

    report = CheckReport(name="api")

    # Every name the factory advertises must construct without arguments. `elu`
    # used to raise TypeError here because its alpha had no default.
    for name in ACTIVATION_NAMES:
        try:
            activation = activation_factory(name)
            report.add(f"activation_factory({name!r})", activation is not None)
        except Exception as error:
            report.add(
                f"activation_factory({name!r})",
                False,
                f"raised {type(error).__name__}: {error}",
            )

    for name in INITIALIZER_NAMES:
        try:
            initializer = initializer_factory(name)
            shape = initializer((4, 3)).shape
            report.add(
                f"initializer_factory({name!r})", shape == (4, 3), f"shape {shape}"
            )
        except Exception as error:
            report.add(
                f"initializer_factory({name!r})",
                False,
                f"raised {type(error).__name__}: {error}",
            )

    for name in ["constant", "random_normal", "random_uniform"]:
        params = {"value": 0.5} if name == "constant" else {}
        try:
            initializer = initializer_factory({"name": name, "params": params})
            report.add(
                f"initializer_factory dict form for {name!r}", initializer is not None
            )
        except Exception as error:
            report.add(
                f"initializer_factory dict form for {name!r}",
                False,
                f"raised {type(error).__name__}: {error}",
            )

    # Lazy shape inference: Linear(out_features) resolves in_features on first call.
    lazy = Linear(6)
    output = lazy(Tensor(np.zeros((5, 4))))
    report.add(
        "Linear infers in_features on first forward",
        lazy.in_features == 4 and output.shape == (5, 6),
        f"in_features={lazy.in_features}, output {output.shape}",
    )

    # The scale a variance-scaling initializer picks is a function of the fan-in and
    # fan-out. Reading them off the wrong axes leaves the weights at a plausible
    # magnitude, so nothing looks wrong until a convolutional stack will not train.
    report.add(
        "fans reads a Linear weight as (in, out)",
        fans((784, 256)) == (784, 256),
        f"{fans((784, 256))}",
    )
    report.add(
        "fans reads a Conv2d kernel through its receptive field",
        fans((32, 16, 3, 3)) == (144, 288),
        f"{fans((32, 16, 3, 3))} (expected (144, 288))",
    )
    kernel = he_normal((32, 16, 3, 3), rng=np.random.default_rng(0)).data
    report.add(
        "he_normal scales a conv kernel by its fan-in",
        bool(np.isclose(kernel.std(), np.sqrt(2.0 / 144), rtol=0.05)),
        f"std {kernel.std():.4f}, expected {np.sqrt(2.0 / 144):.4f}, "
        f"the out-channel reading would give {np.sqrt(2.0 / 32):.4f}",
    )

    unknown_rejected = True
    for factory in (activation_factory, initializer_factory):
        try:
            factory("not_a_real_name")
            unknown_rejected = False
        except ValueError:
            pass
    report.add("factories reject unknown names", unknown_rejected)

    return report


def check_modules() -> CheckReport:
    """Verify that the module tree composes: nesting, freezing, modes, checkpoints."""

    report = CheckReport(name="modules")
    X = Tensor(np.random.default_rng(0).standard_normal((5, 4)))

    def build() -> Sequential:
        return Sequential([Sequential([Linear(4, 3, activation="tanh")]), Linear(3, 2)])

    flat = Sequential([Linear(4, 3, activation="tanh"), Linear(3, 2)])
    deep = build()
    flat(X)
    deep(X)

    # A container that loses a nested child reports fewer parameters and trains fewer
    # layers, without raising anywhere.
    report.add(
        "a nested container reports every parameter",
        flat.num_parameters() == deep.num_parameters() == 23,
        f"flat {flat.num_parameters()}, nested {deep.num_parameters()}",
    )
    report.add(
        "named_parameters uses dotted paths",
        sorted(deep.named_parameters()) == ["0.0.W", "0.0.b", "1.W", "1.b"],
        f"{sorted(deep.named_parameters())}",
    )

    # Nesting used to type-check and run the forward pass, then fail here.
    try:
        model = build()
        optimizer = SGD(model, learning_rate=0.1)
        loss_fn = MeanSquaredError()
        y = Tensor(np.zeros((5, 2)))
        first = float(loss_fn(y, model(X)).item())
        for _ in range(20):
            loss = loss_fn(y, model(X))
            model.zero_grad()
            loss.backward()
            optimizer.update()
        last = float(loss_fn(y, model(X)).item())
        report.add(
            "a nested container trains", last < first, f"{first:.4f} -> {last:.4f}"
        )
    except Exception as error:
        report.add(
            "a nested container trains",
            False,
            f"raised {type(error).__name__}: {error}",
        )

    # Freezing has to reach descendants, and the optimizer has to honor it.
    model = build()
    model(X)
    model[0].freeze()
    before = model.state_dict()
    optimizer = SGD(model, learning_rate=0.5)
    loss = MeanSquaredError()(Tensor(np.zeros((5, 2))), model(X))
    model.zero_grad()
    loss.backward()
    optimizer.update()
    after = model.state_dict()
    report.add(
        "freezing a branch excludes it from the update",
        bool(np.array_equal(before["0.0.W"], after["0.0.W"]))
        and not np.array_equal(before["1.W"], after["1.W"]),
    )
    report.add(
        "a frozen parameter still receives a gradient",
        bool(np.any(model.named_parameters()["0.0.W"].grad != 0.0)),
    )

    # Modes propagate, or Dropout and the normalization layers keep training behavior
    # during evaluation.
    model = build()
    model.eval()
    report.add(
        "eval propagates through the tree",
        all(not module.training for _, module in model.named_modules()),
    )
    model.train()
    report.add(
        "train propagates through the tree",
        all(module.training for _, module in model.named_modules()),
    )

    # A checkpoint has to reproduce the model's outputs exactly.
    source, target = build(), build()
    source(X)
    target(X)
    target.load_state_dict(source.state_dict())
    report.add(
        "a state dict round trips",
        bool(np.allclose(source(X).data, target(X).data)),
    )

    # A layer that ignores the mode gives a different answer on every evaluation call,
    # and nothing raises.
    stochastic = Sequential([Linear(4, 6, activation="relu"), Dropout(0.5), Linear(2)])
    report.add(
        "dropout varies during training",
        not np.array_equal(stochastic(X).data, stochastic(X).data),
    )
    stochastic.eval()
    report.add(
        "dropout is deterministic at evaluation",
        bool(np.array_equal(stochastic(X).data, stochastic(X).data)),
    )

    normalized = BatchNorm1d(4)
    scaled = Tensor(np.random.default_rng(1).standard_normal((32, 4)) * 3.0 + 5.0)
    for _ in range(200):
        normalized(scaled)
    running = normalized.named_buffers()["running_mean"].copy()
    report.add(
        "batch norm tracks the running mean",
        bool(np.allclose(running, scaled.data.mean(axis=0), atol=1e-3)),
        f"{np.round(running, 3).tolist()}",
    )

    normalized.eval()
    single = Tensor(scaled.data[:1])
    expected = (single.data - running) / np.sqrt(
        normalized.named_buffers()["running_var"] + 1e-5
    )
    report.add(
        "batch norm uses the running statistics at evaluation",
        bool(np.allclose(normalized(single).data, expected)),
    )
    normalized(Tensor(scaled.data + 100.0))
    report.add(
        "batch norm does not update its statistics at evaluation",
        bool(np.array_equal(normalized.named_buffers()["running_mean"], running)),
    )

    # A plain list of modules is not registered, so its layers receive gradients and
    # are never stepped. The container types exist to prevent that, and the assignment
    # itself is refused so the mistake cannot be made quietly.
    holder = ModuleList([Linear(4, 3), Linear(3, 2)])
    report.add(
        "a ModuleList registers its contents",
        [name for name, _ in holder.named_children()] == ["0", "1"],
    )

    class Held(Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = ModuleList([Linear(4, 4, activation="relu"), Linear(4, 2)])

        def forward(self, inputs: Tensor) -> Tensor:
            for block in self.blocks:
                inputs = block(inputs)
            return inputs

        @property
        def hyperparameters(self) -> dict[str, object]:
            return {}

    held = Held()
    held(X)
    report.add(
        "a ModuleList's parameters reach the enclosing tree",
        sorted(held.named_parameters())
        == ["blocks.0.W", "blocks.0.b", "blocks.1.W", "blocks.1.b"],
        f"{sorted(held.named_parameters())}",
    )

    before = held.state_dict()
    optimizer = SGD(held, learning_rate=0.1)
    for _ in range(5):
        loss = MeanSquaredError()(Tensor(np.zeros((5, 2))), held(X))
        held.zero_grad()
        loss.backward()
        optimizer.update()
    after = held.state_dict()
    report.add(
        "every module in a ModuleList is stepped",
        all(not np.array_equal(before[name], after[name]) for name in before),
        f"unmoved: {[n for n in before if np.array_equal(before[n], after[n])]}",
    )

    keyed = ModuleDict({"value": Linear(4, 1), "policy": Linear(4, 3)})
    report.add(
        "a ModuleDict registers its contents under its keys",
        [name for name, _ in keyed.named_children()] == ["value", "policy"],
    )

    try:
        held.blocks = [Linear(4, 4)]  # type: ignore[assignment]
        report.add("a plain list of Modules is refused", False, "no error raised")
    except TypeError as error:
        report.add(
            "a plain list of Modules is refused", "ModuleList" in str(error), str(error)
        )

    # An unrolled recurrence is the graph shape the tape was built for: the same
    # weights on the tape once per timestep, and thousands of nodes deep.
    cell = RNNCell(3, 4)
    step_input = Tensor(np.full((2, 3), 0.05))
    state = cell(step_input)
    for _ in range(299):
        state = cell(step_input, state)
    pmath.sum(state).backward()
    report.add(
        "a 300-step unrolled cell differentiates without recursion",
        bool(np.all(np.isfinite(cell.parameters["W_hh"].grad))),
    )
    report.add(
        "an unrolled cell shares one set of weights",
        sorted(cell.named_parameters()) == ["W_hh", "W_ih", "b_hh", "b_ih"],
        f"{sorted(cell.named_parameters())}",
    )

    # An embedding's reverse is a scatter-add. Writing instead of adding would train a
    # frequent token as though it appeared once.
    table = Embedding(4, 2)
    pmath.sum(table(np.array([1, 1, 1, 3]))).backward()
    report.add(
        "an embedding accumulates a repeated token's gradient",
        table.parameters["W"].grad[:, 0].tolist() == [0.0, 3.0, 0.0, 1.0],
        f"{table.parameters['W'].grad[:, 0].tolist()}",
    )

    # Running statistics are buffers, not parameters: never stepped, always saved.
    report.add(
        "running statistics are excluded from the optimizer",
        all(
            "running" not in name
            for group in normalized.parameter_groups()
            for name in group
        ),
    )
    report.add(
        "running statistics are included in the checkpoint",
        "running_mean" in normalized.state_dict(),
    )

    return report


def check_invariants() -> CheckReport:
    """Run the autodiff, optimizer, module, and API invariant checks together."""

    report = CheckReport(name="invariants")
    for suite in (
        check_autodiff(),
        check_optimizers(),
        check_modules(),
        check_training(),
        check_api(),
    ):
        report.extend(suite)
    return report

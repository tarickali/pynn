"""Numerical gradient checking for PyNN's automatic differentiation.

The functions here compare the gradients produced by reverse-mode autodiff against
central-difference approximations. They are part of the public API so that users can
verify custom layers and operations, not just the ones shipped with the library:

    from pynn.core import Tensor, concat, masked_fill, split, stack, where
    from pynn.verify import check_gradients

    result = check_gradients(lambda ts: my_loss(ts[0]), [Tensor(x)])
    assert result.passed, result

`check_all_gradients` sweeps every operation the library ships with, and is what
`pynn.verify.run_all` calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

import pynn.core.math as pmath
import pynn.functional as F
from pynn.core import Tensor, concat, masked_fill, split, stack, where
from pynn.core.types import Array
from pynn.functional.losses import (
    binary_crossentropy,
    categorical_crossentropy,
    huber,
    mean_absolute_error,
    mean_squared_error,
    sparse_categorical_crossentropy,
)
from pynn.functional.modules import (
    avg_pool2d,
    batch_norm,
    conv2d,
    dropout,
    embedding,
    flatten,
    layer_norm,
    linear,
    max_pool2d,
    unflatten,
)
from pynn.nn import LSTMCell, RNNCell
from pynn.verify.report import CheckReport, CheckResult

__all__ = [
    "DEFAULT_SEED",
    "GradCheckResult",
    "GradientCase",
    "InputCheck",
    "analytic_gradient",
    "check_all_gradients",
    "check_case",
    "check_gradients",
    "gradient_cases",
    "numerical_gradient",
]

#: A function mapping a list of Tensors to a single scalar (size-1) Tensor.
ScalarFn = Callable[[list[Tensor]], Tensor]
#: A single-argument tensor op, or a factory producing random inputs for one.
TensorFn = Callable[..., Tensor]
#: Seed for `gradient_cases`. Fixed so that a failure is reproducible.
DEFAULT_SEED = 20240605


@dataclass(frozen=True)
class GradientCase:
    """One named gradient check: a scalar function and the inputs to differentiate.

    `gradient_cases` returns these so that callers can drive the sweep themselves —
    `tests/test_gradcheck.py` turns each into a separate pytest case so that a failure
    names the operation, rather than reporting one failure for the whole sweep.
    """

    name: str
    fn: ScalarFn
    inputs: list[Tensor]
    #: Central-difference step. Cases with large-magnitude inputs need a larger step,
    #: because the difference of two nearby large values loses significant digits.
    eps: float = 1e-6
    rtol: float = 1e-5

    @property
    def id(self) -> str:
        """The name as a pytest-friendly identifier.

        >>> GradientCase("add (3, 4)+(4,)", lambda ts: ts[0], []).id
        'add-(3,4)+(4,)'
        """
        return self.name.replace(", ", ",").replace(" ", "-")


@dataclass
class InputCheck:
    """Comparison of analytic and numerical gradients for one input."""

    index: int
    analytic: Array
    numerical: Array
    max_relative_error: float
    max_absolute_error: float
    passed: bool

    def worst_index(self) -> tuple[int, ...]:
        """Index of the element with the largest absolute discrepancy."""
        diff = np.abs(self.analytic - self.numerical)
        return tuple(
            int(axis) for axis in np.unravel_index(int(np.argmax(diff)), diff.shape)
        )


@dataclass
class GradCheckResult:
    """Aggregate result of a gradient check across all inputs."""

    inputs: list[InputCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.inputs)

    @property
    def max_relative_error(self) -> float:
        if not self.inputs:
            return 0.0
        return max(check.max_relative_error for check in self.inputs)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"gradient check {status} "
            f"(max relative error {self.max_relative_error:.3e})"
        ]
        for check in self.inputs:
            mark = "ok" if check.passed else "FAIL"
            lines.append(
                f"  [{mark}] input {check.index} shape={check.analytic.shape} "
                f"rel={check.max_relative_error:.3e} abs={check.max_absolute_error:.3e}"
            )
            if not check.passed:
                idx = check.worst_index()
                lines.append(
                    f"         worst at {idx}: "
                    f"analytic={check.analytic[idx]:+.8f} "
                    f"numerical={check.numerical[idx]:+.8f}"
                )
        return "\n".join(lines)


def _as_scalar(output: Tensor) -> float:
    if output.size != 1:
        raise ValueError(
            "gradient checking requires a scalar-valued function, "
            f"but the function returned a Tensor of shape {output.shape}"
        )
    return float(output.data.reshape(()))


def numerical_gradient(
    fn: ScalarFn,
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
) -> list[Array]:
    """Approximate d fn / d input for each input using central differences.

    Central differences, ``(f(x + eps) - f(x - eps)) / (2 * eps)``, have error
    ``O(eps**2)`` rather than the ``O(eps)`` of a forward difference, which matters
    because ``eps`` cannot be made arbitrarily small in floating point.

    Parameters
    ----------
    fn : ScalarFn
        Maps a list of Tensors to a scalar Tensor. Called ``2 * n`` times, where
        ``n`` is the total number of elements across all inputs, so keep inputs small.
    inputs : Sequence[Tensor]
        Tensors to differentiate with respect to. Their data is restored afterwards.
    eps : float, default 1e-6
        Perturbation size. Suited to float64; use a larger value for float32.

    Returns
    -------
    list[Array]
        One gradient array per input, matching that input's shape.
    """

    tensors = list(inputs)
    gradients = []

    for target in tensors:
        gradient = np.zeros(target.data.shape, dtype=np.float64)
        iterator = np.nditer(target.data, flags=["multi_index"])
        while not iterator.finished:
            index = iterator.multi_index
            original = target.data[index]

            target.data[index] = original + eps
            plus = _as_scalar(fn(tensors))

            target.data[index] = original - eps
            minus = _as_scalar(fn(tensors))

            target.data[index] = original
            gradient[index] = (plus - minus) / (2 * eps)
            iterator.iternext()
        gradients.append(gradient)

    return gradients


def analytic_gradient(fn: ScalarFn, inputs: Sequence[Tensor]) -> list[Array]:
    """Gradients from a single reverse-mode pass, with gradients zeroed first."""
    tensors = list(inputs)
    for tensor in tensors:
        tensor.zero_grad()

    output = fn(tensors)
    _as_scalar(output)
    output.backward()

    return [tensor.grad.copy() for tensor in tensors]


def check_gradients(
    fn: ScalarFn,
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> GradCheckResult:
    """Compare autodiff gradients against central-difference approximations.

    Parameters
    ----------
    fn : ScalarFn
        Maps a list of Tensors to a scalar Tensor. Must be called repeatedly, so it
        should be free of side effects and must rebuild its graph on every call.
    inputs : Sequence[Tensor]
        Tensors to differentiate with respect to.
    eps : float, default 1e-6
        Perturbation size for the numerical gradient.
    rtol, atol : float
        An element passes when
        ``|analytic - numerical| <= atol + rtol * (|analytic| + |numerical|)``.
        Scaling by the sum of magnitudes keeps the check meaningful for gradients
        that are near zero.

    Returns
    -------
    GradCheckResult
        Use ``result.passed``; ``str(result)`` reports the worst offending element,
        which makes assertion failures diagnosable.
    """

    tensors = list(inputs)
    analytic = analytic_gradient(fn, tensors)
    numerical = numerical_gradient(fn, tensors, eps=eps)

    result = GradCheckResult()
    for index, (a, n) in enumerate(zip(analytic, numerical, strict=True)):
        if a.shape != n.shape:
            raise ValueError(
                f"input {index}: autodiff produced a gradient of shape {a.shape} "
                f"for a tensor of shape {n.shape}"
            )
        absolute = np.abs(a - n)
        scale = atol + rtol * (np.abs(a) + np.abs(n))
        relative = absolute / np.maximum(
            np.abs(a) + np.abs(n), np.finfo(np.float64).tiny
        )
        result.inputs.append(
            InputCheck(
                index=index,
                analytic=a,
                numerical=n,
                max_relative_error=float(np.max(relative)) if relative.size else 0.0,
                max_absolute_error=float(np.max(absolute)) if absolute.size else 0.0,
                passed=bool(np.all(absolute <= scale)),
            )
        )

    return result


def _contract(op: TensorFn) -> ScalarFn:
    """``sum(op(x) * c)`` — contracts the output against a second input tensor."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(op(ts[0]) * ts[1])

    return scalar


def _contract_with_reuse(op: TensorFn) -> ScalarFn:
    """``sum(op(x) * c) + sum(x * d)`` — gives ``x`` a second consumer.

    A reverse pass that assigns to ``x.grad`` rather than accumulating into it loses one
    of the two contributions here, while producing an exactly correct gradient in the
    single-consumer form.
    """

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(op(ts[0]) * ts[1]) + pmath.sum(ts[0] * ts[2])

    return scalar


def _contract_binary(op: TensorFn) -> ScalarFn:
    """``sum(op(a, b))`` for a binary operator."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(op(ts[0], ts[1]))

    return scalar


def _contract_binary_with_reuse(op: TensorFn) -> ScalarFn:
    """``sum(op(a, b) * c) + sum(a * d)`` — the reuse check for a binary operator."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(op(ts[0], ts[1]) * ts[2]) + pmath.sum(ts[0] * ts[3])

    return scalar


def _conv2d_case(stride: tuple[int, int], padding: tuple[int, int]) -> ScalarFn:
    """``sum(conv2d(x, kernel))`` for one stride/padding combination."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(conv2d(ts[0], ts[1], None, stride, padding))

    return scalar


def _layer_norm_case(normalized_shape: tuple[int, ...]) -> ScalarFn:
    """``sum(layer_norm(x, gamma, beta) * c)`` over one set of trailing axes."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(layer_norm(ts[0], ts[1], ts[2], normalized_shape) * ts[3])

    return scalar


def _batch_norm_case(features: int | None) -> ScalarFn:
    """``sum(batch_norm(x, gamma, beta) * c)``, training when `features` is None.

    Evaluation is a genuinely different function: it normalizes by fixed statistics
    rather than by ones derived from `x`, so its gradient does not go through the mean
    and the variance at all.
    """

    def scalar(ts: list[Tensor]) -> Tensor:
        if features is None:
            normalized = batch_norm(ts[0], ts[1], ts[2], training=True)
        else:
            normalized = batch_norm(
                ts[0],
                ts[1],
                ts[2],
                running_mean=np.zeros(features),
                running_var=np.ones(features),
                training=False,
            )
        return pmath.sum(normalized * ts[3])

    return scalar


def _pool_case(
    pool: TensorFn,
    kernel: tuple[int, int],
    stride: tuple[int, int] | None,
    padding: tuple[int, int],
) -> ScalarFn:
    """``sum(pool(x) * c)`` for one pooling geometry."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(pool(ts[0], kernel, stride, padding) * ts[1])

    return scalar


def _pool_reuse_case(pool: TensorFn) -> ScalarFn:
    """``sum(pool(x)) + sum(x * c)`` — the reuse check for a pooling op."""

    def scalar(ts: list[Tensor]) -> Tensor:
        return pmath.sum(pool(ts[0])) + pmath.sum(ts[0] * ts[1])

    return scalar


def gradient_cases(seed: int = DEFAULT_SEED) -> list[GradientCase]:
    """Build the gradient-check sweep over every operation the library ships with.

    Covers each operator across broadcasting and matmul shape combinations, every math
    function and activation, every loss in both its logits and probability forms, the
    module functions across strides and paddings, and graph topologies where a tensor
    has more than one consumer.

    Every unary and binary op appears twice: once in isolation, and once with its input
    also feeding a second consumer. The second form is the one that catches a reverse
    pass which assigns to `grad` instead of accumulating into it, since the
    single-consumer form yields an exactly correct gradient either way.

    Parameters
    ----------
    seed : int, default DEFAULT_SEED
        Seed for the random inputs.

    Returns
    -------
    list[GradientCase]
        Independent cases. Inputs may be reused across `check_gradients` calls, which
        zeroes gradients before each pass and restores perturbed data afterwards.
    """

    rng = np.random.default_rng(seed)

    def normal(*shape: int) -> Tensor:
        return Tensor(rng.standard_normal(shape))

    def away_from_zero(*shape: int) -> Tensor:
        """Values bounded away from 0, where relu and abs have no derivative."""
        magnitude = rng.uniform(0.5, 2.0, shape)
        return Tensor(magnitude * rng.choice([-1.0, 1.0], shape))

    def positive(*shape: int) -> Tensor:
        return Tensor(rng.uniform(0.5, 3.0, shape))

    cases: list[tuple[str, ScalarFn, list[Tensor]]] = []
    # Collected separately because they need a larger central-difference step.
    scale_cases: list[tuple[str, ScalarFn, list[Tensor]]] = []

    # Operators, including the broadcasting shape combinations.
    for left, right in [
        ((3, 4), (3, 4)),
        ((3, 4), (4,)),
        ((3, 4), (1, 4)),
        ((3, 1), (1, 4)),
        ((2, 3, 4), (3, 4)),
        ((2, 3, 4), (1, 1, 4)),
        ((5,), ()),
    ]:
        shapes = f"{left}+{right}"
        for op_name, op in [
            ("add", lambda a, b: a + b),
            ("sub", lambda a, b: a - b),
            ("mul", lambda a, b: a * b),
            # Several ops over the same two operands, so both broadcast reductions and
            # accumulation across operators have to be right simultaneously.
            ("mixed", lambda a, b: a + b * a - b),
        ]:
            cases.append(
                (
                    f"{op_name} {shapes}",
                    _contract_binary(op),
                    [normal(*left), normal(*right)],
                )
            )

    # matmul, covering vector promotion and batch broadcasting.
    for left, right in [
        ((3, 4), (4, 2)),
        ((1, 4), (4, 1)),
        ((4,), (4, 2)),
        ((3, 4), (4,)),
        ((4,), (4,)),
        ((2, 3, 4), (4, 5)),
        ((2, 3, 4), (2, 4, 5)),
        ((2, 3, 4), (4,)),
    ]:
        cases.append(
            (
                f"matmul {left}@{right}",
                lambda ts: pmath.sum(ts[0] @ ts[1]),
                [normal(*left), normal(*right)],
            )
        )

    cases += [
        (
            "truediv broadcasting",
            lambda ts: pmath.sum(ts[0] / ts[1]),
            [normal(3, 4), positive(4)],
        ),
        (
            "rtruediv by a scalar",
            lambda ts: pmath.sum(2.0 / ts[0]),
            [positive(3, 4)],
        ),
        (
            "transpose into matmul",
            lambda ts: pmath.sum(ts[0].T @ ts[1]),
            [normal(3, 4), normal(3, 2)],
        ),
        (
            "transpose with explicit axes",
            lambda ts: pmath.sum(ts[0].transpose((2, 0, 1)) * ts[1]),
            [normal(2, 3, 4), normal(4, 2, 3)],
        ),
    ]

    # Every unary op, in two graph shapes. The plain shape catches a wrong Jacobian;
    # the reused shape additionally catches a reverse pass that assigns to `x.grad`
    # instead of accumulating into it, which is invisible when the input has a single
    # consumer. Each is contracted against a random tensor so that a wrong Jacobian
    # cannot hide behind an upstream gradient of all ones.
    # A square shape keeps every reduction axis broadcast-compatible with the tensor
    # the result is contracted against.
    square = (4, 4)
    unary: list[tuple[str, TensorFn, TensorFn]] = [
        ("exp", pmath.exp, normal),
        ("log", pmath.log, positive),
        ("abs", pmath.abs, away_from_zero),
        ("identity", F.identity, normal),
        ("affine", lambda x: F.affine(x, 2.0, 3.0), normal),
        ("relu", F.relu, away_from_zero),
        ("leaky relu", lambda x: F.relu(x, 0.2), away_from_zero),
        ("sigmoid", F.sigmoid, normal),
        ("tanh", F.tanh, normal),
        ("elu", F.elu, away_from_zero),
        ("elu alpha=0.5", lambda x: F.elu(x, 0.5), away_from_zero),
        ("selu", F.selu, away_from_zero),
        ("softplus", F.softplus, normal),
        ("softmax", F.softmax, normal),
        ("softmax axis=0", lambda x: F.softmax(x, axis=0), normal),
        ("gelu", F.gelu, normal),
        # The exact path goes through erf and has a different derivative expression
        # from the tanh approximation, so it is a separate function to check.
        ("gelu exact", lambda x: F.gelu(x, approximate="none"), normal),
        ("silu", F.silu, normal),
        ("log softmax", F.log_softmax, normal),
        ("log softmax axis=0", lambda x: F.log_softmax(x, axis=0), normal),
        ("neg", lambda x: -x, normal),
        ("transpose", lambda x: x.T, normal),
        ("pow 3", lambda x: x**3, positive),
        ("pow -2", lambda x: x**-2, positive),
    ]
    for axis in [None, 0, 1, (0, 1), -1]:
        unary.append(
            (f"sum axis={axis}", lambda x, a=axis: pmath.sum(x, axis=a), normal)
        )
        unary.append(
            (f"mean axis={axis}", lambda x, a=axis: pmath.mean(x, axis=a), normal)
        )

    for name, op, domain in unary:
        cases.append((name, _contract(op), [domain(*square), normal(*square)]))
        cases.append(
            (
                f"{name} with a reused input",
                _contract_with_reuse(op),
                [domain(*square), normal(*square), normal(*square)],
            )
        )

    # The activations again at magnitudes where a naive implementation overflows. The
    # stability suite checks the results are finite; these check they are also *right*,
    # which a saturating approximation would not be.
    def at_scale(*shape: int) -> Tensor:
        magnitude = rng.uniform(20.0, 60.0, shape)
        return Tensor(magnitude * rng.choice([-1.0, 1.0], shape))

    for name, activation in [
        ("sigmoid", F.sigmoid),
        ("tanh", F.tanh),
        ("elu", F.elu),
        ("selu", F.selu),
        ("softplus", F.softplus),
        ("softmax", F.softmax),
        ("silu", F.silu),
        ("gelu", F.gelu),
        ("log softmax", F.log_softmax),
    ]:
        scale_cases.append(
            (
                f"{name} at large magnitudes",
                _contract(activation),
                [at_scale(*square), normal(*square)],
            )
        )

    # The same reuse check for the binary operators and the module functions: feed one
    # operand to a second consumer so both gradient contributions must survive.
    binary: list[tuple[str, TensorFn, TensorFn]] = [
        ("add", lambda a, b: a + b, normal),
        ("sub", lambda a, b: a - b, normal),
        ("mul", lambda a, b: a * b, normal),
        ("truediv", lambda a, b: a / b, positive),
        ("matmul", lambda a, b: a @ b, normal),
    ]
    for name, op, domain in binary:
        cases.append(
            (
                f"{name} with a reused operand",
                _contract_binary_with_reuse(op),
                [domain(*square), domain(*square), normal(*square), normal(*square)],
            )
        )

    # PReLU learns its slope, so alpha is a second differentiable input rather than a
    # Python float — the only activation here whose gradient has two destinations.
    cases += [
        (
            "prelu shared slope",
            lambda ts: pmath.sum(F.prelu(ts[0], ts[1]) * ts[2]),
            [away_from_zero(4, 4), Tensor(np.array([0.25])), normal(4, 4)],
        ),
        (
            "prelu per-channel slope",
            lambda ts: pmath.sum(F.prelu(ts[0], ts[1]) * ts[2]),
            [away_from_zero(3, 4), Tensor(rng.uniform(0.1, 0.5, 4)), normal(3, 4)],
        ),
        (
            "prelu per-channel slope on images",
            lambda ts: pmath.sum(F.prelu(ts[0], ts[1]) * ts[2]),
            [
                away_from_zero(2, 3, 4, 4),
                Tensor(rng.uniform(0.1, 0.5, 3)),
                normal(2, 3, 4, 4),
            ],
        ),
        (
            "prelu with a reused input",
            lambda ts: pmath.sum(F.prelu(ts[0], ts[1])) + pmath.sum(ts[0] * ts[2]),
            [away_from_zero(4, 4), Tensor(np.array([0.25])), normal(4, 4)],
        ),
    ]

    # Losses.
    binary_targets = Tensor(rng.integers(0, 2, (6, 1)).astype(np.float64))
    class_targets = Tensor(np.eye(4)[rng.integers(0, 4, 6)])
    cases += [
        (
            "loss mse",
            lambda ts: mean_squared_error(ts[0], ts[1]),
            [normal(4, 3), normal(4, 3)],
        ),
        (
            "loss mse reduction=sum",
            lambda ts: mean_squared_error(ts[0], ts[1], reduction="sum"),
            [normal(4, 3), normal(4, 3)],
        ),
        (
            "loss mae",
            lambda ts: mean_absolute_error(ts[0], ts[1]),
            [Tensor(np.zeros((4, 3))), away_from_zero(4, 3)],
        ),
        (
            "loss bce from logits",
            lambda ts: binary_crossentropy(binary_targets, ts[0], logits=True),
            [normal(6, 1)],
        ),
        (
            "loss bce from probabilities",
            lambda ts: binary_crossentropy(binary_targets, ts[0], logits=False),
            [Tensor(rng.uniform(0.15, 0.85, (6, 1)))],
        ),
        (
            "loss categorical ce from logits",
            lambda ts: categorical_crossentropy(class_targets, ts[0], logits=True),
            [normal(6, 4)],
        ),
        (
            "loss categorical ce from probabilities",
            lambda ts: categorical_crossentropy(class_targets, ts[0], logits=False),
            [Tensor(rng.uniform(0.1, 0.9, (6, 4)))],
        ),
    ]

    # Huber switches formula at |r| = delta. Residuals are kept clear of the join,
    # where the second derivative jumps and a central difference straddles both pieces.
    huber_targets = Tensor(np.zeros((4, 3)))
    cases += [
        (
            "loss huber inside delta",
            lambda ts: huber(huber_targets, ts[0], delta=2.0),
            [Tensor(rng.uniform(0.2, 0.8, (4, 3)) * rng.choice([-1.0, 1.0], (4, 3)))],
        ),
        (
            "loss huber beyond delta",
            lambda ts: huber(huber_targets, ts[0], delta=0.5),
            [Tensor(rng.uniform(2.0, 4.0, (4, 3)) * rng.choice([-1.0, 1.0], (4, 3)))],
        ),
        (
            "loss huber reduction=sum",
            lambda ts: huber(huber_targets, ts[0], delta=1.0, reduction="sum"),
            [Tensor(rng.uniform(0.1, 0.5, (4, 3)))],
        ),
    ]

    # The sparse form has a different reverse pass from the one-hot form — it scatters
    # into the label positions rather than subtracting a dense target.
    sparse_labels = rng.integers(0, 4, 6)
    cases += [
        (
            "loss sparse categorical ce from logits",
            lambda ts: sparse_categorical_crossentropy(sparse_labels, ts[0]),
            [normal(6, 4)],
        ),
        (
            "loss sparse categorical ce from probabilities",
            lambda ts: sparse_categorical_crossentropy(
                sparse_labels, ts[0], logits=False
            ),
            [Tensor(rng.uniform(0.1, 0.9, (6, 4)))],
        ),
        (
            "loss sparse categorical ce reduction=sum",
            lambda ts: sparse_categorical_crossentropy(
                sparse_labels, ts[0], reduction="sum"
            ),
            [normal(6, 4)],
        ),
    ]

    # An unreduced loss is not a scalar, so it is contracted against a second tensor to
    # give the check something to differentiate — and that contraction is exactly how a
    # caller weighting examples would use it.
    weights = normal(6, 1)
    cases += [
        (
            "loss bce reduction=none",
            lambda ts: pmath.sum(
                binary_crossentropy(binary_targets, ts[0], reduction="none") * ts[1]
            ),
            [normal(6, 1), weights],
        ),
        (
            "loss categorical ce reduction=none",
            lambda ts: pmath.sum(
                categorical_crossentropy(class_targets, ts[0], reduction="none") * ts[1]
            ),
            [normal(6, 4), normal(6)],
        ),
        (
            "loss mse reduction=none",
            lambda ts: pmath.sum(
                mean_squared_error(ts[0], ts[1], reduction="none") * ts[2]
            ),
            [normal(4, 3), normal(4, 3), normal(4, 3)],
        ),
        (
            "loss huber reduction=none",
            lambda ts: pmath.sum(huber(huber_targets, ts[0], reduction="none") * ts[1]),
            [Tensor(rng.uniform(0.1, 0.6, (4, 3))), normal(4, 3)],
        ),
    ]

    # Module functions.
    cases += [
        (
            "linear",
            lambda ts: pmath.sum(linear(ts[0], ts[1], ts[2])),
            [normal(5, 3), normal(3, 2), normal(2)],
        ),
        (
            "linear without bias",
            lambda ts: pmath.sum(linear(ts[0], ts[1], None)),
            [normal(5, 3), normal(3, 2)],
        ),
        (
            "flatten",
            lambda ts: pmath.sum(flatten(ts[0]) * ts[1]),
            [normal(2, 3, 4), normal(2, 12)],
        ),
        (
            "flatten with a reused input",
            lambda ts: pmath.sum(flatten(ts[0]) * ts[1]) + pmath.sum(ts[0] * ts[2]),
            [normal(2, 3, 4), normal(2, 12), normal(2, 3, 4)],
        ),
        (
            "unflatten",
            lambda ts: pmath.sum(unflatten(ts[0], (3, 4)) * ts[1]),
            [normal(2, 12), normal(2, 3, 4)],
        ),
        (
            "unflatten with an inferred axis",
            lambda ts: pmath.sum(unflatten(ts[0], (3, -1)) * ts[1]),
            [normal(2, 12), normal(2, 3, 4)],
        ),
        (
            "unflatten with a reused input",
            lambda ts: (
                pmath.sum(unflatten(ts[0], (3, 4)) * ts[1]) + pmath.sum(ts[0] * ts[2])
            ),
            [normal(2, 12), normal(2, 3, 4), normal(2, 12)],
        ),
        # The round trip, which is the claim the pair makes: a gradient that goes in
        # comes back through both layouts unchanged.
        (
            "flatten then unflatten",
            lambda ts: pmath.sum(unflatten(flatten(ts[0]), (3, 4)) * ts[1]),
            [normal(2, 3, 4), normal(2, 3, 4)],
        ),
    ]
    for stride, padding in [
        ((1, 1), (0, 0)),
        ((2, 2), (0, 0)),
        ((1, 1), (1, 1)),
        ((2, 2), (1, 2)),
    ]:
        cases.append(
            (
                f"conv2d stride={stride} padding={padding}",
                _conv2d_case(stride, padding),
                [normal(2, 2, 5, 5), normal(3, 2, 3, 3)],
            )
        )
    # The bias is per output channel and broadcasts over the spatial axes, so its
    # gradient has to sum over batch *and* position.
    cases.append(
        (
            "conv2d with a per-channel bias",
            lambda ts: pmath.sum(conv2d(ts[0], ts[1], ts[2])),
            [normal(2, 2, 4, 4), normal(3, 2, 3, 3), normal(3, 1, 1)],
        )
    )

    # Normalization. The statistics depend on every element being normalized, so the
    # gradient does not factor elementwise: a formulation that forgets either the mean
    # or the variance path still produces plausible, wrong numbers.
    for label, shape, normalized in [
        ("layer_norm (4, 5)", (4, 5), (5,)),
        ("layer_norm over two axes", (3, 4, 5), (4, 5)),
    ]:
        cases.append(
            (
                label,
                _layer_norm_case(normalized),
                [
                    normal(*shape),
                    Tensor(rng.uniform(0.5, 1.5, normalized)),
                    normal(*normalized),
                    normal(*shape),
                ],
            )
        )
    cases += [
        (
            "layer_norm without affine parameters",
            lambda ts: pmath.sum(layer_norm(ts[0]) * ts[1]),
            [normal(4, 5), normal(4, 5)],
        ),
        (
            "layer_norm with a reused input",
            lambda ts: pmath.sum(layer_norm(ts[0]) * ts[1]) + pmath.sum(ts[0] * ts[2]),
            [normal(4, 5), normal(4, 5), normal(4, 5)],
        ),
    ]

    for label, shape, statistics in [
        ("batch_norm (6, 3)", (6, 3), (3,)),
        ("batch_norm (4, 3, 2, 2)", (4, 3, 2, 2), (3, 1, 1)),
    ]:
        inputs = [
            normal(*shape),
            Tensor(rng.uniform(0.5, 1.5, statistics)),
            normal(*statistics),
            normal(*shape),
        ]
        cases.append((f"{label} training", _batch_norm_case(None), inputs))
        # Evaluation reads fixed statistics, which makes it an affine map of the input
        # and its gradient a different expression entirely.
        cases.append(
            (f"{label} evaluation", _batch_norm_case(statistics[0]), list(inputs))
        )
    cases.append(
        (
            "batch_norm with a reused input",
            lambda ts: pmath.sum(batch_norm(ts[0]) * ts[1]) + pmath.sum(ts[0] * ts[2]),
            [normal(5, 3), normal(5, 3), normal(5, 3)],
        )
    )

    # Pooling. Max pooling needs values drawn far enough apart that a probe of size
    # `eps` cannot change which element wins a window, where the function is not
    # differentiable. A permutation supplies that; dividing it through keeps the gap at
    # `1/n` — still four orders of magnitude above `eps` — while holding the values
    # inside [-0.5, 0.5].
    #
    # The scaling is the point, not incidental. A bare permutation runs to 99, and
    # round-off in `f(x + eps) - f(x - eps)` scales with `|f|`: divided by
    # `2 * eps = 2e-6` it reached 1e-7, past `atol`, for gradient elements that are
    # near zero. Both pooling checks then passed at the default seed and failed at
    # others, which is the worst way for a check to be wrong.
    def separated(*shape: int) -> Tensor:
        count = int(np.prod(shape))
        spread = (rng.permutation(count) - (count - 1) / 2) / count
        return Tensor(spread.reshape(shape))

    # Average pooling is smooth, so it needs no separation at all and takes the same
    # normal inputs as everything else.
    for pool_name, pool, domain in [
        ("max_pool2d", max_pool2d, separated),
        ("avg_pool2d", avg_pool2d, normal),
    ]:
        for kernel, step, pad in [
            ((2, 2), None, (0, 0)),
            ((2, 2), (1, 1), (0, 0)),
            ((3, 3), (2, 2), (1, 1)),
        ]:
            # The tensor the output is contracted against has to match the pooled
            # shape, so ask the op itself rather than recomputing the arithmetic.
            pooled = pool(Tensor(np.zeros((2, 2, 5, 5))), kernel, step, pad).shape
            cases.append(
                (
                    f"{pool_name} kernel={kernel} stride={step} padding={pad}",
                    _pool_case(pool, kernel, step, pad),
                    [domain(2, 2, 5, 5), normal(*pooled)],
                )
            )
        cases.append(
            (
                f"{pool_name} with a reused input",
                _pool_reuse_case(pool),
                [domain(2, 2, 4, 4), normal(2, 2, 4, 4)],
            )
        )

    # Dropout's mask is drawn once per call, so the check has to see the same mask on
    # every probe; a fixed seed is what makes the function deterministic enough to
    # difference at all.
    cases.append(
        (
            "dropout",
            lambda ts: pmath.sum(dropout(ts[0], 0.5, training=True, rng=0) * ts[1]),
            [normal(4, 5), normal(4, 5)],
        )
    )
    cases.append(
        (
            "dropout at evaluation",
            lambda ts: pmath.sum(dropout(ts[0], 0.5, training=False) * ts[1]),
            [normal(4, 5), normal(4, 5)],
        )
    )

    # Indexing and shape operations. The reverse of a gather is a scatter, and the
    # scatter has to accumulate: an index read twice contributes twice, which a plain
    # assignment into the gradient would silently drop.
    cases += [
        (
            "getitem row",
            lambda ts: pmath.sum(ts[0][1] * ts[1]),
            [normal(4, 4), normal(4)],
        ),
        (
            "getitem slice",
            lambda ts: pmath.sum(ts[0][1:3] * ts[1]),
            [normal(4, 4), normal(2, 4)],
        ),
        (
            "getitem column",
            lambda ts: pmath.sum(ts[0][:, 2] * ts[1]),
            [normal(4, 4), normal(4)],
        ),
        (
            "getitem fancy",
            lambda ts: pmath.sum(ts[0][[0, 2, 3]] * ts[1]),
            [normal(4, 4), normal(3, 4)],
        ),
        # The one that catches a scatter which assigns instead of accumulating.
        (
            "getitem repeated index",
            lambda ts: pmath.sum(ts[0][[1, 1, 1]] * ts[1]),
            [normal(4, 4), normal(3, 4)],
        ),
        (
            "getitem boolean mask",
            lambda ts: pmath.sum(ts[0][np.array([True, False, True, False])] * ts[1]),
            [normal(4, 4), normal(2, 4)],
        ),
        (
            "getitem with a reused input",
            lambda ts: pmath.sum(ts[0][0]) + pmath.sum(ts[0] * ts[1]),
            [normal(4, 4), normal(4, 4)],
        ),
        (
            "reshape",
            lambda ts: pmath.sum(ts[0].reshape(2, 8) * ts[1]),
            [normal(4, 4), normal(2, 8)],
        ),
        (
            "concat axis=0",
            lambda ts: pmath.sum(concat([ts[0], ts[1]]) * ts[2]),
            [normal(2, 3), normal(4, 3), normal(6, 3)],
        ),
        (
            "concat axis=1",
            lambda ts: pmath.sum(concat([ts[0], ts[1]], axis=1) * ts[2]),
            [normal(3, 2), normal(3, 4), normal(3, 6)],
        ),
        # The same tensor joined to itself: both halves of the gradient must land.
        (
            "concat with a tensor twice",
            lambda ts: pmath.sum(concat([ts[0], ts[0]]) * ts[1]),
            [normal(2, 3), normal(4, 3)],
        ),
        (
            "stack",
            lambda ts: pmath.sum(stack([ts[0], ts[1], ts[2]]) * ts[3]),
            [normal(3, 2), normal(3, 2), normal(3, 2), normal(3, 3, 2)],
        ),
        (
            "stack axis=1",
            lambda ts: pmath.sum(stack([ts[0], ts[1]], axis=1) * ts[2]),
            [normal(3, 2), normal(3, 2), normal(3, 2, 2)],
        ),
        (
            "split",
            lambda ts: pmath.sum(
                sum(
                    pmath.sum(piece * weight)
                    for piece, weight in zip(
                        split(ts[0], 3), [1.0, 2.0, 3.0], strict=True
                    )
                )
            ),
            [normal(6, 2)],
        ),
        # Only some pieces used: the unused slice must still end up at zero rather
        # than uninitialized.
        (
            "split with an unused piece",
            lambda ts: (
                pmath.sum(split(ts[0], 3)[0] * ts[1])
                + pmath.sum(split(ts[0], 3)[2] * ts[1])
            ),
            [normal(6, 2), normal(2, 2)],
        ),
        (
            "split uneven sizes",
            lambda ts: pmath.sum(concat(list(split(ts[0], [1, 3, 2]))) * ts[1]),
            [normal(6, 2), normal(6, 2)],
        ),
    ]

    # where and masked_fill route the gradient per element rather than transforming it.
    # The reused-branch case is the one that matters: `where(c, x, x)` gives one tensor
    # two consumers, and the two halves have to sum to the whole gradient.
    selector = rng.random((4, 4)) > 0.5
    cases += [
        (
            "where",
            lambda ts: pmath.sum(where(selector, ts[0], ts[1]) * ts[2]),
            [normal(4, 4), normal(4, 4), normal(4, 4)],
        ),
        (
            "where against a scalar",
            lambda ts: pmath.sum(where(selector, ts[0], 0.0) * ts[1]),
            [normal(4, 4), normal(4, 4)],
        ),
        (
            "where broadcasting a row",
            lambda ts: pmath.sum(where(selector, ts[0], ts[1]) * ts[2]),
            [normal(4, 4), normal(4), normal(4, 4)],
        ),
        (
            "where with the same tensor in both branches",
            lambda ts: pmath.sum(where(selector, ts[0], ts[0]) * ts[1]),
            [normal(4, 4), normal(4, 4)],
        ),
        (
            "where with a reused input",
            lambda ts: (
                pmath.sum(where(selector, ts[0], ts[1])) + pmath.sum(ts[0] * ts[2])
            ),
            [normal(4, 4), normal(4, 4), normal(4, 4)],
        ),
        (
            "masked_fill",
            lambda ts: pmath.sum(masked_fill(ts[0], selector, -1.0) * ts[1]),
            [normal(4, 4), normal(4, 4)],
        ),
        # The case it exists for: a causal mask ahead of a softmax. A filled position is
        # overwritten rather than scaled, so its gradient is exactly zero.
        (
            "masked_fill into a causal softmax",
            lambda ts: pmath.sum(
                F.softmax(masked_fill(ts[0], np.triu(np.ones((4, 4), bool), 1), -1e9))
                * ts[1]
            ),
            [normal(4, 4), normal(4, 4)],
        ),
        (
            "masked_fill with a reused input",
            lambda ts: (
                pmath.sum(masked_fill(ts[0], selector, 0.5)) + pmath.sum(ts[0] * ts[1])
            ),
            [normal(4, 4), normal(4, 4)],
        ),
    ]

    # Embedding, whose reverse is a scatter-add. A repeated index is the whole point:
    # a common token appears many times in a batch and every occurrence contributes.
    table_indices = rng.integers(0, 5, (3, 4))
    cases += [
        (
            "embedding",
            lambda ts: pmath.sum(embedding(ts[0], table_indices) * ts[1]),
            [normal(5, 3), normal(3, 4, 3)],
        ),
        (
            "embedding with a repeated index",
            lambda ts: pmath.sum(embedding(ts[0], np.array([1, 1, 1, 2]))),
            [normal(5, 3)],
        ),
        (
            "embedding with a reused table",
            lambda ts: pmath.sum(embedding(ts[0], np.array([0, 2]))) + pmath.sum(ts[0]),
            [normal(5, 3)],
        ),
    ]

    # A recurrent cell unrolled over several steps: the same weights appear once per
    # step, so every parameter's gradient is a sum over the whole sequence. This is the
    # shape that catches a reverse pass which assigns rather than accumulates — it
    # would train on the last timestep only, and still descend.
    def _unrolled(cell_type, steps: int) -> ScalarFn:
        def scalar(ts: list[Tensor]) -> Tensor:
            cell = cell_type(3, 4)
            cell.parameters["W_ih"] = ts[0]
            cell.parameters["W_hh"] = ts[1]
            cell.parameters["b_ih"] = ts[2]
            cell.parameters["b_hh"] = ts[3]
            cell.input_size, cell.initialized = 3, True
            # First step separately: it takes no incoming state, and writing it that
            # way keeps `state` a Tensor (or a pair) rather than an optional one.
            state = cell(ts[4])
            for _ in range(steps - 1):
                state = cell(ts[4], state)
            hidden = state[0] if isinstance(state, tuple) else state
            return pmath.sum(hidden * ts[5])

        return scalar

    for label, cell_type, width in [
        ("RNNCell", RNNCell, 4),
        ("LSTMCell", LSTMCell, 16),
    ]:
        for steps in (1, 4):
            cases.append(
                (
                    f"{label} unrolled {steps} steps",
                    _unrolled(cell_type, steps),
                    [
                        normal(3, width),
                        normal(4, width),
                        normal(width),
                        normal(width),
                        normal(2, 3),
                        normal(2, 4),
                    ],
                )
            )

    # Graph topologies where a tensor feeds more than one consumer. A plain
    # feedforward chain never exercises these, and they are where gradient
    # accumulation bugs surface.
    cases += [
        (
            "graph input used twice",
            lambda ts: pmath.sum(ts[0]) + pmath.sum(ts[0] * 2.0),
            [normal(3, 4)],
        ),
        (
            "graph branch rejoins",
            lambda ts: pmath.sum(F.relu(ts[0])) + pmath.sum(ts[0]),
            [away_from_zero(3, 4)],
        ),
        (
            "graph diamond",
            lambda ts: pmath.sum(pmath.exp(ts[0]) * pmath.log(ts[0] + 3.0)),
            [Tensor(rng.uniform(-0.5, 0.5, (3, 4)))],
        ),
        ("graph self product", lambda ts: pmath.sum(ts[0] * ts[0]), [normal(3, 4)]),
        (
            "graph tied weights",
            lambda ts: pmath.sum(F.tanh(F.tanh(ts[0] @ ts[1]) @ ts[1])),
            [normal(3, 4), normal(4, 4)],
        ),
        (
            "graph residual connection",
            lambda ts: pmath.sum(ts[0] + F.tanh(ts[0] @ ts[1])),
            [normal(4, 3), normal(3, 3)],
        ),
    ]

    def two_heads(ts):
        trunk = F.tanh(ts[0] @ ts[1])
        return pmath.sum(trunk @ ts[2]) + pmath.sum(F.sigmoid(trunk @ ts[3]))

    cases.append(
        (
            "graph two heads share a trunk",
            two_heads,
            [normal(4, 3), normal(3, 3), normal(3, 2), normal(3, 2)],
        )
    )

    aux_targets = Tensor(rng.standard_normal((4, 2)))

    def auxiliary_loss(ts):
        hidden = F.tanh(ts[0] @ ts[1])
        return mean_squared_error(aux_targets, hidden) + 0.5 * mean_absolute_error(
            aux_targets, hidden
        )

    cases.append(("graph auxiliary loss", auxiliary_loss, [normal(4, 3), normal(3, 2)]))

    built = [GradientCase(name, fn, inputs) for name, fn, inputs in cases]
    # Differencing two values around x=50 with eps=1e-6 leaves almost no significant
    # digits, so the large-magnitude cases get a coarser step and looser tolerance.
    built += [
        GradientCase(name, fn, inputs, eps=1e-4, rtol=1e-4)
        for name, fn, inputs in scale_cases
    ]
    return built


def check_case(case: GradientCase) -> CheckResult:
    """Run one `GradientCase`, converting a raised exception into a failed result."""

    try:
        result = check_gradients(case.fn, case.inputs, eps=case.eps, rtol=case.rtol)
    except Exception as error:
        # Broad on purpose: a raised exception is reported as a failed check, not
        # re-raised, so the rest of the sweep can still run.
        return CheckResult(case.name, False, f"raised {type(error).__name__}: {error}")
    return CheckResult(
        case.name,
        result.passed,
        f"max relative error {result.max_relative_error:.2e}",
    )


def check_all_gradients(seed: int = DEFAULT_SEED) -> CheckReport:
    """Gradient-check every operation the library ships with.

    Parameters
    ----------
    seed : int, default DEFAULT_SEED
        Seed for the random inputs, so a failure is reproducible.

    Returns
    -------
    CheckReport
        One result per case, named after the operation being checked.

    See Also
    --------
    gradient_cases : The cases this runs, for driving them individually.
    """

    report = CheckReport(name="gradients")
    report.results.extend(check_case(case) for case in gradient_cases(seed))
    return report

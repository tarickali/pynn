"""The two functions the reverse pass routes every gradient through.

The hand-written shape lists below are the cases somebody thought of. The property
tests after them are the ones nobody did: `unbroadcast` composes two different
reduction rules and `matrix_multiply_gradients` layers matmul's vector promotion on
top of batch broadcasting, and both are shape plumbing, where the failure mode is a
plausible array of the right size holding the wrong numbers.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pynn.core.utils import matrix_multiply_gradients, unbroadcast


@pytest.mark.parametrize(
    "operand_shape,broadcast_shape",
    [
        ((3,), (2, 3)),
        ((2, 3), (2, 3)),
        ((1, 3), (2, 3)),
        ((2, 1), (2, 3)),
        ((3,), (32, 2, 3)),
        ((2, 3), (32, 2, 3)),
        ((1, 1), (4, 5)),
        ((), (4, 5)),
    ],
    ids=str,
)
def test_unbroadcast_restores_operand_shape(operand_shape, broadcast_shape):
    gradient = np.ones(broadcast_shape)
    reduced = unbroadcast(gradient, operand_shape)

    assert reduced.shape == operand_shape
    # Summing a gradient of ones must conserve the total, since every element of the
    # broadcast result is a copy of exactly one element of the operand.
    assert reduced.sum() == pytest.approx(gradient.sum())


def test_unbroadcast_is_a_no_op_for_matching_shapes():
    gradient = np.arange(6.0).reshape(2, 3)
    assert unbroadcast(gradient, (2, 3)) is gradient


def test_unbroadcast_sums_the_correct_axis():
    gradient = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    assert unbroadcast(gradient, (3,)).tolist() == [11.0, 22.0, 33.0]
    assert unbroadcast(gradient, (2, 1)).tolist() == [[6.0], [60.0]]


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((3, 4), (4, 2)),
        ((4,), (4, 2)),
        ((3, 4), (4,)),
        ((4,), (4,)),
        ((2, 3, 4), (4, 5)),
        ((2, 3, 4), (2, 4, 5)),
        ((2, 3, 4), (4,)),
    ],
    ids=str,
)
def test_matrix_multiply_gradients_shapes(left_shape, right_shape):
    rng = np.random.default_rng(0)
    left = rng.standard_normal(left_shape)
    right = rng.standard_normal(right_shape)
    gradient = np.ones_like(left @ right)

    left_gradient, right_gradient = matrix_multiply_gradients(gradient, left, right)

    assert left_gradient.shape == left.shape
    assert right_gradient.shape == right.shape


# --------------------------------------------------------------------------- #
# Property-based tests
#
# `derandomize=True` on purpose. Hypothesis normally seeds itself from the clock, so
# the same commit explores different shapes on every run and on each of the five
# interpreters in the CI matrix — five chances per push for a red that has nothing to
# do with the change. Derandomizing makes these exactly as reproducible as the
# parametrized lists above, with the search widened by raising `max_examples` rather
# than by rolling the dice again. `deadline=None` for the same reason: the per-example
# time limit is a property of the machine, not of the code under test.
# --------------------------------------------------------------------------- #

MAX_RANK = 4
MAX_DIM = 4
#: Enough shapes to cover the rank and vector-promotion combinations several times
#: over, and small enough that the whole property suite stays around a second.
PROPERTY = settings(max_examples=250, derandomize=True, deadline=None)


@st.composite
def broadcastable_to(draw, target: tuple[int, ...]) -> tuple[int, ...]:
    """A shape that NumPy broadcasts up to `target`.

    Both of NumPy's rules, and nothing else: an operand may drop leading axes
    entirely, and may carry a 1 wherever the target has any length. Those are exactly
    the two reductions `unbroadcast` has to undo.
    """
    rank = draw(st.integers(0, len(target)))
    suffix = target[len(target) - rank :]
    return tuple(draw(st.sampled_from((1, size))) for size in suffix)


@st.composite
def broadcast_pair(draw) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A broadcast result shape, and an operand shape that broadcasts to it."""
    target = tuple(
        draw(st.lists(st.integers(1, MAX_DIM), max_size=MAX_RANK, min_size=0))
    )
    return target, draw(broadcastable_to(target))


@st.composite
def matmul_shapes(draw) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Operand shapes for a valid `left @ right`.

    Covers both of `np.matmul`'s special cases: either operand may be 1-D, in which
    case it is promoted for the multiply and the promoted axis is dropped from the
    result, and leading axes on the operands that are *not* promoted broadcast against
    each other. A 1-D operand carries no batch axes, so the two features compose
    rather than multiplying out.
    """
    m, k, n = (draw(st.integers(1, MAX_DIM)) for _ in range(3))
    left_vector = draw(st.booleans())
    right_vector = draw(st.booleans())

    left = (k,) if left_vector else (m, k)
    right = (k,) if right_vector else (k, n)

    batch = tuple(draw(st.lists(st.integers(1, 3), max_size=2)))
    if not left_vector:
        left = draw(broadcastable_to(batch)) + left
    if not right_vector:
        right = draw(broadcastable_to(batch)) + right
    return left, right


def _arrays(shapes: tuple[tuple[int, ...], ...], seed: int) -> list[np.ndarray]:
    """Deterministic standard-normal arrays, one per shape."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape) for shape in shapes]


@given(broadcast_pair())
@PROPERTY
def test_unbroadcast_counts_the_replications(shapes):
    """Every element of the reduced gradient is the copy count of the operand.

    NumPy broadcasting replicates an operand a fixed number of times — the ratio of
    the two element counts — and the gradient of a copy is a sum, so a gradient of
    ones must come back as that ratio, uniformly.
    """
    target, operand_shape = shapes
    reduced = unbroadcast(np.ones(target), operand_shape)

    replications = math.prod(target) // math.prod(operand_shape)
    assert reduced.shape == operand_shape
    assert np.array_equal(reduced, np.full(operand_shape, float(replications)))


@given(broadcast_pair(), st.integers(0, 2**32 - 1))
@PROPERTY
def test_unbroadcast_is_the_adjoint_of_broadcasting(shapes, seed):
    """`<broadcast_to(x), g> == <x, unbroadcast(g)>`, for every compatible pair.

    Stronger than the replication count above, and it is the property that matters:
    a gradient of ones is invariant under summing the *wrong* axis, so the count alone
    passes for a reduction that conserves the total while scrambling which operand
    element each contribution belongs to.
    """
    target, operand_shape = shapes
    gradient, operand = _arrays((target, operand_shape), seed)

    broadcast = float(np.sum(np.broadcast_to(operand, target) * gradient))
    reduced = float(np.sum(operand * unbroadcast(gradient, operand_shape)))

    assert reduced == pytest.approx(broadcast, rel=1e-9, abs=1e-12)


@given(matmul_shapes(), st.integers(0, 2**32 - 1))
@PROPERTY
def test_matrix_multiply_gradients_satisfy_the_adjoint_identity(shapes, seed):
    """`<A @ B, G> == <A, dA> == <B, dB>`, for every valid pair of operand shapes.

    `A -> <A @ B, G>` is linear in `A`, so its gradient pairs with `A` to give the
    value back. That is the definition of the reverse-mode adjoint, and unlike a shape
    assertion it fails for a squeeze that should have been a sum — which is precisely
    the distinction matmul's promoted axes and broadcast batch axes turn on.
    """
    left_shape, right_shape = shapes
    left, right = _arrays((left_shape, right_shape), seed)
    output = left @ right
    (gradient,) = _arrays((output.shape,), seed + 1)

    left_gradient, right_gradient = matrix_multiply_gradients(gradient, left, right)

    assert left_gradient.shape == left.shape
    assert right_gradient.shape == right.shape

    pairing = float(np.sum(output * gradient))
    assert float(np.sum(left * left_gradient)) == pytest.approx(
        pairing, rel=1e-9, abs=1e-12
    )
    assert float(np.sum(right * right_gradient)) == pytest.approx(
        pairing, rel=1e-9, abs=1e-12
    )

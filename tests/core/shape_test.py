"""Indexing, reshaping, and the join/split operations.

Gradients are checked against central differences by the verify sweep. What is here is
the behaviour a gradient check cannot see on its own: that indexing returns something
still attached to the graph, that a scatter accumulates rather than overwrites, and that
a tensor used twice in a join receives both contributions.
"""

import numpy as np
import pytest

import pynn.core.math as pmath
from pynn.core import Tensor, concat, masked_fill, split, stack, where


@pytest.fixture
def x() -> Tensor:
    return Tensor(np.arange(12.0).reshape(3, 4))


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def test_indexing_returns_a_tensor(x):
    """It used to return a raw array, which dropped the graph with no error."""
    row = x[1]

    assert isinstance(row, Tensor)
    assert row.data.tolist() == [4.0, 5.0, 6.0, 7.0]


@pytest.mark.parametrize(
    "key,expected_shape",
    [
        (1, (4,)),
        (slice(1, 3), (2, 4)),
        ((0, 2), ()),
        ((slice(None), 1), (3,)),
        ((Ellipsis, 0), (3,)),
        ([0, 2], (2, 4)),
        (np.array([True, False, True]), (2, 4)),
    ],
    ids=["int", "slice", "pair", "column", "ellipsis", "fancy", "mask"],
)
def test_every_index_form_works(x, key, expected_shape):
    selected = x[key]

    assert isinstance(selected, Tensor)
    assert selected.shape == expected_shape
    assert np.array_equal(selected.data, x.data[key])


def test_a_slice_stays_on_the_tape(x):
    pmath.sum(x[1] * 2.0).backward()

    expected = np.zeros((3, 4))
    expected[1] = 2.0
    assert np.array_equal(x.grad, expected)


def test_a_repeated_index_accumulates(x):
    """The case a scatter that assigns instead of adding gets wrong."""
    pmath.sum(x[[1, 1, 1]]).backward()

    expected = np.zeros((3, 4))
    expected[1] = 3.0
    assert np.array_equal(x.grad, expected)


def test_a_boolean_mask_routes_to_the_selected_rows(x):
    pmath.sum(x[np.array([True, False, True])]).backward()

    assert np.array_equal(x.grad, [[1.0] * 4, [0.0] * 4, [1.0] * 4])


def test_indexing_composes_with_other_consumers(x):
    """The slice and the whole tensor both contribute to the same gradient."""
    (pmath.sum(x[0]) + pmath.sum(x)).backward()

    expected = np.ones((3, 4))
    expected[0] = 2.0
    assert np.array_equal(x.grad, expected)


def test_a_tensor_may_be_used_as_an_index():
    values = Tensor(np.arange(6.0).reshape(3, 2))
    indices = Tensor(np.array([0, 2]), dtype=np.intp)

    assert np.array_equal(values[indices].data, [[0.0, 1.0], [4.0, 5.0]])


def test_a_float_tensor_index_is_rejected_with_a_useful_message():
    """Tensors default to float64, so this is the mistake people will actually make."""
    values = Tensor(np.arange(6.0).reshape(3, 2))

    with pytest.raises(ValueError, match="integers or booleans"):
        values[Tensor([0, 2])]


def test_setitem_assigns_without_recording(x):
    """Documented as non-differentiable: the tape holds what produced a value."""
    x[0] = np.zeros(4)
    assert x.data[0].tolist() == [0.0] * 4

    x[1] = Tensor(np.ones(4))
    assert x.data[1].tolist() == [1.0] * 4


# --------------------------------------------------------------------------- #
# reshape
# --------------------------------------------------------------------------- #


def test_reshape_accepts_both_call_forms(x):
    assert x.reshape(2, 6).shape == (2, 6)
    assert x.reshape((2, 6)).shape == (2, 6)
    assert x.reshape(-1).shape == (12,)


def test_reshape_gradient_returns_to_the_original_shape(x):
    pmath.sum(x.reshape(12) * 3.0).backward()

    assert x.grad.shape == (3, 4)
    assert np.allclose(x.grad, 3.0)


# --------------------------------------------------------------------------- #
# concat
# --------------------------------------------------------------------------- #


def test_concat_joins_along_an_existing_axis():
    a, b = Tensor(np.ones((2, 3))), Tensor(np.zeros((2, 4)))

    assert concat([a, b], axis=1).shape == (2, 7)
    assert concat([a, Tensor(np.zeros((5, 3)))]).shape == (7, 3)


def test_concat_splits_its_gradient_back_to_each_input():
    a, b = Tensor(np.ones((2, 3))), Tensor(np.zeros((2, 2)))

    pmath.sum(concat([a, b], axis=1) * np.arange(5.0)).backward()

    assert np.array_equal(a.grad, np.tile([0.0, 1.0, 2.0], (2, 1)))
    assert np.array_equal(b.grad, np.tile([3.0, 4.0], (2, 1)))


def test_a_tensor_concatenated_with_itself_gets_both_gradients():
    shared = Tensor(np.ones((2, 2)))

    pmath.sum(concat([shared, shared])).backward()

    assert np.allclose(shared.grad, 2.0)


def test_concat_accepts_a_negative_axis():
    a, b = Tensor(np.ones((2, 3))), Tensor(np.ones((2, 1)))

    joined = concat([a, b], axis=-1)
    pmath.sum(joined).backward()

    assert joined.shape == (2, 4)
    assert np.allclose(a.grad, 1.0)


def test_concat_of_one_tensor_is_that_tensor():
    a = Tensor(np.ones((2, 3)))
    assert np.array_equal(concat([a]).data, a.data)


def test_concat_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one tensor"):
        concat([])


# --------------------------------------------------------------------------- #
# stack
# --------------------------------------------------------------------------- #


def test_stack_adds_an_axis():
    tensors = [Tensor(np.ones(3)), Tensor(np.zeros(3))]

    assert stack(tensors).shape == (2, 3)
    assert stack(tensors, axis=1).shape == (3, 2)


def test_stack_routes_each_slice_back():
    a, b = Tensor(np.ones(3)), Tensor(np.ones(3))

    pmath.sum(stack([a, b]) * np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).backward()

    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 2.0)


def test_stack_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one tensor"):
        stack([])


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #


def test_split_into_equal_sections():
    pieces = split(Tensor(np.ones((6, 2))), 3)

    assert len(pieces) == 3
    assert all(piece.shape == (2, 2) for piece in pieces)


def test_split_with_explicit_sizes():
    pieces = split(Tensor(np.ones((6, 2))), [1, 3, 2])

    assert [piece.shape[0] for piece in pieces] == [1, 3, 2]


def test_each_piece_routes_its_gradient_to_its_own_slice():
    whole = Tensor(np.ones((6, 2)))
    first, second, third = split(whole, 3)

    (pmath.sum(first) + pmath.sum(second) * 2.0 + pmath.sum(third) * 3.0).backward()

    assert whole.grad[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_an_unused_piece_leaves_its_slice_at_zero():
    whole = Tensor(np.ones((6, 2)))
    first, _, third = split(whole, 3)

    (pmath.sum(first) + pmath.sum(third)).backward()

    assert whole.grad[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]


def test_split_along_a_different_axis():
    pieces = split(Tensor(np.ones((2, 6))), 3, axis=1)

    assert all(piece.shape == (2, 2) for piece in pieces)


def test_split_then_concat_is_the_identity():
    whole = Tensor(np.arange(12.0).reshape(6, 2))

    rejoined = concat(list(split(whole, 3)))
    pmath.sum(rejoined).backward()

    assert np.array_equal(rejoined.data, whole.data)
    assert np.allclose(whole.grad, 1.0)


@pytest.mark.parametrize(
    "sections,message",
    [
        (0, "sections must be positive"),
        (4, "into 4 equal sections"),
        ([1, 2], "sum to 3"),
    ],
    ids=str,
)
def test_split_rejects_impossible_requests(sections, message):
    with pytest.raises(ValueError, match=message):
        split(Tensor(np.ones((6, 2))), sections)


# --------------------------------------------------------------------------- #
# where and masked_fill
#
# The reverse pass routes rather than transforms: each element of the incoming gradient
# belongs to exactly one input, and the other gets zero there. The cases that matter are
# the ones where "exactly one" is not obvious — a broadcast branch, a scalar branch, and
# the same tensor on both sides.
# --------------------------------------------------------------------------- #


def test_where_selects_per_element():
    values = Tensor(np.array([-1.0, 2.0, -3.0, 4.0]))
    other = Tensor(np.array([10.0, 20.0, 30.0, 40.0]))

    assert where(values.data > 0, values, other).data.tolist() == [10.0, 2.0, 30.0, 4.0]


def test_where_routes_each_element_to_one_branch():
    a = Tensor(np.ones(4))
    b = Tensor(np.ones(4))
    mask = np.array([True, False, True, False])

    pmath.sum(where(mask, a, b)).backward()

    assert a.grad.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert b.grad.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert (a.grad + b.grad).tolist() == [1.0] * 4, "every element goes somewhere"


def test_where_accepts_a_scalar_branch():
    values = Tensor(np.array([1.0, 2.0]))

    result = where(np.array([True, False]), values, 0.0)
    pmath.sum(result).backward()

    assert result.data.tolist() == [1.0, 0.0]
    assert values.grad.tolist() == [1.0, 0.0]


def test_where_unbroadcasts_a_lower_rank_branch():
    """A replicated branch's gradient is a sum over the axes it was copied along."""
    row = Tensor(np.zeros(3))
    mask = np.array([[True, False, True], [False, False, True]])

    pmath.sum(where(mask, Tensor(np.zeros((2, 3))), row)).backward()

    assert row.grad.tolist() == [1.0, 2.0, 0.0]


def test_where_with_the_same_tensor_in_both_branches():
    """One tensor, two consumers: the halves must sum to the whole gradient."""
    values = Tensor(np.ones(4))

    pmath.sum(where(np.array([True, False, True, False]), values, values)).backward()

    assert values.grad.tolist() == [1.0] * 4


def test_where_accepts_a_tensor_condition():
    values = Tensor(np.array([1.0, 2.0]))
    condition = Tensor(np.array([True, False]), dtype=bool)

    assert where(condition, values, 0.0).data.tolist() == [1.0, 0.0]


def test_where_rejects_an_unconvertible_branch():
    with pytest.raises(TypeError, match="Cannot perform operation"):
        where(np.array([True]), Tensor(np.ones(1)), "nope")


def test_masked_fill_replaces_marked_positions():
    values = Tensor(np.array([1.0, 2.0, 3.0]))

    filled = masked_fill(values, np.array([False, True, False]), -9.0)

    assert filled.data.tolist() == [1.0, -9.0, 3.0]


def test_a_filled_position_gets_exactly_zero_gradient():
    """It was overwritten, not scaled, so it had no influence on the output."""
    values = Tensor(np.array([1.0, 2.0, 3.0]))

    pmath.sum(masked_fill(values, np.array([False, True, False]), -9.0)).backward()

    assert values.grad.tolist() == [1.0, 0.0, 1.0]


def test_masked_fill_keeps_the_constant_off_the_tape():
    """`where(mask, value, x)` would make the constant a leaf holding a gradient."""
    values = Tensor(np.array([1.0, 2.0]))

    filled = masked_fill(values, np.array([True, False]), -1e9)

    assert len(filled.children) == 1
    assert filled.children[0] is values


def test_masked_fill_drives_a_causal_attention_mask():
    """The case it exists for: each position sees only itself and the past."""
    import pynn.functional as F

    scores = Tensor(np.zeros((1, 4, 4)))
    future = np.triu(np.ones((4, 4), dtype=bool), k=1)

    weights = F.softmax(masked_fill(scores, future, -1e9), axis=-1)

    assert np.allclose(weights.data[0, 0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(weights.data[0, 3], [0.25, 0.25, 0.25, 0.25])
    assert np.allclose(weights.data.sum(axis=-1), 1.0)

    pmath.sum(weights).backward()
    assert np.all(scores.grad[0][future] == 0.0)

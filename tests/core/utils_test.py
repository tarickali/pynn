import numpy as np
import pytest

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

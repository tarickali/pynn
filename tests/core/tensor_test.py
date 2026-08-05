import numpy as np
import pytest

from pynn.core import Tensor
from pynn.core.types import Array


def test_init(rng):
    # test : init with number
    data = 0
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    data = 1.0
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    # test : init with lists
    data = [0]
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    data = [[0.0, 1.0, 2.0], [1, 2, 3]]
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    # test : init with nddata
    data = rng.standard_normal((2, 3))
    tensor = Tensor(data)
    assert np.allclose(tensor.data, data)
    assert tensor.dtype == data.dtype
    assert isinstance(tensor.data, Array)

    data = np.zeros((2, 3))
    tensor = Tensor(data, np.float32)
    assert np.all(tensor.data == data)
    assert tensor.dtype == np.float32
    assert isinstance(tensor.data, Array)


def test_binary_operations(rng):
    #####----- test : add -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a + b
    x = Tensor(a)
    y = Tensor(b)
    z = x + y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a + b
    x = Tensor(a)
    y = Tensor(b)
    z = x + y
    assert isinstance(z, Tensor)
    assert z.shape == c.shape
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : sub -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : mul -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : matmul -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 2))
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : truediv -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)


def test_unary_operations(rng):
    #####----- test : pow -----#####
    a = rng.standard_normal((2, 3))
    b = a**2
    x = Tensor(a)
    y = x**2
    assert isinstance(y, Tensor)
    assert np.allclose(y.data, b)
    assert isinstance(y.data, Array)

    #####----- test : neg -----#####
    a = rng.standard_normal((2, 3))
    b = -a
    x = Tensor(a)
    y = -x
    assert isinstance(y, Tensor)
    assert np.allclose(y.data, b)
    assert isinstance(y.data, Array)


def test_comparison_operations(rng):
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    x = Tensor(a)
    y = Tensor(b)

    for op in [
        lambda u, v: u == v,
        lambda u, v: u != v,
        lambda u, v: u <= v,
        lambda u, v: u < v,
        lambda u, v: u >= v,
        lambda u, v: u > v,
    ]:
        expected = op(a, b)
        result = op(x, y)
        assert isinstance(result, Tensor)
        assert result.dtype == bool
        assert np.array_equal(result.data, expected)


def test_bool_rejects_a_multi_element_tensor():
    """`if a == b:` must not silently return True for every non-empty Tensor."""
    equal = Tensor([1.0, 2.0]) == Tensor([1.0, 2.0])
    unequal = Tensor([1.0, 2.0]) == Tensor([1.0, 3.0])

    assert equal.dtype == bool
    assert equal.all()
    assert not unequal.all()
    assert unequal.any()

    with pytest.raises(ValueError, match="ambiguous"):
        bool(equal)
    with pytest.raises(ValueError, match="ambiguous"):
        bool(unequal)
    with pytest.raises(ValueError, match="ambiguous"):
        if equal:
            pass


def test_bool_of_a_scalar_tensor():
    assert bool(Tensor(1.0)) is True
    assert bool(Tensor(0.0)) is False
    assert bool(Tensor([True], dtype=bool)) is True
    assert bool(Tensor([False], dtype=bool)) is False


def test_cast():
    x = Tensor([0, 1, 2])
    x.cast(int)
    assert x.dtype == int
    x.cast(np.float32)
    assert x.dtype == np.float32


# --------------------------------------------------------------------------- #
# Reflected operators
#
# `2.0 - x` reaches `__rsub__`, which is a different code path from `x - 2.0` and
# gets the operand order wrong if it just delegates to `__sub__`.
# --------------------------------------------------------------------------- #


def test_reflected_operators_use_the_right_operand_order(rng):
    a = rng.standard_normal((2, 3))
    x = Tensor(a)

    assert np.allclose((2.0 + x).data, 2.0 + a)
    assert np.allclose((2.0 - x).data, 2.0 - a)
    assert np.allclose((2.0 * x).data, 2.0 * a)
    assert np.allclose((2.0 / x).data, 2.0 / a)


def test_reflected_matmul_uses_the_right_operand_order(rng):
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 4))

    assert np.allclose((a @ Tensor(b)).data, a @ b)


def test_an_ndarray_on_the_left_still_produces_a_tensor(rng):
    """NumPy used to win the dispatch and return an object-dtype array of Tensors.

    That is silently wrong: the result looks like an array, has no gradient, and
    every subsequent operation on it is Python-level object arithmetic.
    """
    a = rng.standard_normal((2, 3))
    x = Tensor(rng.standard_normal((2, 3)))

    for result in [a + x, a - x, a * x, a / x]:
        assert isinstance(result, Tensor)
        assert result.dtype == np.float64


def test_an_ndarray_on_the_left_stays_on_the_tape(rng):
    import pynn.core.math as pmath

    x = Tensor(rng.standard_normal((2, 3)))
    pmath.sum(np.full((2, 3), 3.0) * x).backward()

    assert np.allclose(x.grad, 3.0)


# --------------------------------------------------------------------------- #
# Accessors and rejection paths
# --------------------------------------------------------------------------- #


def test_numpy_returns_the_underlying_array(rng):
    a = rng.standard_normal((2, 3))
    tensor = Tensor(a)

    assert isinstance(tensor.numpy(), Array)
    assert np.allclose(tensor.numpy(), a)


def test_item_returns_a_python_scalar():
    assert Tensor([[2.5]]).item() == 2.5


def test_shape_properties():
    tensor = Tensor(np.zeros((2, 3, 4)))

    assert tensor.shape == (2, 3, 4)
    assert tensor.ndim == 3
    assert tensor.size == 24


def test_repr_names_the_dtype_and_shape():
    text = repr(Tensor(np.zeros((2, 3))))

    assert text.startswith("Tensor(")
    assert "float64" in text
    assert "(2, 3)" in text


def test_pow_rejects_a_tensor_exponent():
    with pytest.raises(TypeError, match="Cannot perform operation"):
        Tensor([1.0, 2.0]) ** Tensor([2.0, 2.0])


def test_operations_reject_an_unconvertible_operand():
    with pytest.raises(TypeError, match="Cannot perform operation"):
        Tensor([1.0]) + "two"

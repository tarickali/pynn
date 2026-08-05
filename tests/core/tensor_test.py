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

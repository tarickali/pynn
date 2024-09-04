import numpy as np
from pynn.core.types import Array
from pynn.core import Tensor


def test_init():
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
    data = np.random.randn(2, 3)
    tensor = Tensor(data)
    assert np.allclose(tensor.data, data)
    assert tensor.dtype == data.dtype
    assert isinstance(tensor.data, Array)

    data = np.zeros((2, 3))
    tensor = Tensor(data, np.float32)
    assert np.all(tensor.data == data)
    assert tensor.dtype == np.float32
    assert isinstance(tensor.data, Array)


def test_binary_operations():
    #####----- test : add -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a + b
    x = Tensor(a)
    y = Tensor(b)
    z = x + y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
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
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : mul -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : matmul -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 2)
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : truediv -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)


def test_unary_operations():
    #####----- test : pow -----#####
    a = np.random.randn(2, 3)
    b = a**2
    x = Tensor(a)
    y = x**2
    assert isinstance(y, Tensor)
    assert np.allclose(y.data, b)
    assert isinstance(y.data, Array)

    #####----- test : neg -----#####
    a = np.random.randn(2, 3)
    b = -a
    x = Tensor(a)
    y = -x
    assert isinstance(y, Tensor)
    assert np.allclose(y.data, b)
    assert isinstance(y.data, Array)


def test_comparison_operations():
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    x = Tensor(a)
    y = Tensor(b)

    c = a == b
    z = x == y
    assert isinstance(z, Tensor)
    assert np.all(z.data == c)

    c = a != b
    z = x != y
    assert isinstance(z, Tensor)
    assert np.all(z.data == c)

    c = a <= b
    z = x <= y
    assert isinstance(z, Tensor)
    assert np.all(z.data == c)

    c = a < b
    z = x < y
    assert isinstance(z, Tensor)
    assert np.all(z.data == c)

    c = a >= b
    z = x >= y
    assert isinstance(z, Tensor)
    assert np.all(z.data == c)

    c = a > b
    z = x > y
    assert isinstance(z, Tensor)
    assert np.all(z.data == c)


def test_cast():
    x = Tensor([0, 1, 2])
    x.cast(int)
    assert x.dtype == int
    x.cast(np.float32)
    assert x.dtype == np.float32

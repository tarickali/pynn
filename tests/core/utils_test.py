import numpy as np
from pynn.core.utils import expand_array, shrink_array


def test_tensor_utils():
    # test : expand and shrink tensors with ndim=2
    x = np.random.randn(2, 3)
    y = np.random.randn(3)
    z = x + y
    assert z.shape == (2, 3)

    x_expand = expand_array(x, z.shape)
    y_expand = expand_array(y, z.shape)
    assert x_expand.shape == z.shape
    assert y_expand.shape == z.shape

    x_shrink = shrink_array(x_expand, x.shape)
    y_shrink = shrink_array(y_expand, y.shape)
    print(x_shrink)
    assert x_shrink.shape == x.shape
    assert y_shrink.shape == y.shape

    # test : expand and shrink tensors with ndim=3
    x = np.random.randn(32, 2, 3)
    y = np.random.randn(2, 3)
    z = x + y
    assert z.shape == (32, 2, 3)

    x_expand = expand_array(x, z.shape)
    y_expand = expand_array(y, z.shape)
    assert x_expand.shape == z.shape
    assert y_expand.shape == z.shape

    x_shrink = shrink_array(x_expand, x.shape)
    y_shrink = shrink_array(y_expand, y.shape)
    assert x_shrink.shape == x.shape
    assert y_shrink.shape == y.shape

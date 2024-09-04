"""
title : test_tensor_utils.py
create : @tarickali 23/12/14
update : @tarickali 23/12/14
"""

import numpy as np
from pynn.core.tensor import Tensor
from pynn.core.utils import expand_tensor, shrink_tensor


def test_tensor_utils():
    # test : expand and shrink tensors with ndim=2
    x = Tensor(np.random.randn(2, 3))
    y = Tensor(np.random.randn(3))
    z = x + y
    assert z.shape == (2, 3)

    x_expand = expand_tensor(x, z.shape)
    y_expand = expand_tensor(y, z.shape)
    assert x_expand.shape == z.shape
    assert y_expand.shape == z.shape

    x_shrink = shrink_tensor(x_expand, x.shape)
    y_shrink = shrink_tensor(y_expand, y.shape)
    print(x_shrink)
    assert x_shrink.shape == x.shape
    assert y_shrink.shape == y.shape

    # test : expand and shrink tensors with ndim=3
    x = Tensor(np.random.randn(32, 2, 3))
    y = Tensor(np.random.randn(2, 3))
    z = x + y
    assert z.shape == (32, 2, 3)

    x_expand = expand_tensor(x, z.shape)
    y_expand = expand_tensor(y, z.shape)
    assert x_expand.shape == z.shape
    assert y_expand.shape == z.shape

    x_shrink = shrink_tensor(x_expand, x.shape)
    y_shrink = shrink_tensor(y_expand, y.shape)
    assert x_shrink.shape == x.shape
    assert y_shrink.shape == y.shape

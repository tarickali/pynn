import numpy as np
import torch
from pynn.core import Tensor
import pynn.core.math.functions as pmath


def test_log():
    W = np.random.uniform(0.1, 10.0, (10, 16))
    x = np.random.uniform(0.1, 10.0, (32, 10))
    b = np.zeros(16)
    t = np.random.uniform(10.0, 20.0, (32, 16))

    tensor_W = Tensor(W)
    tensor_x = Tensor(x)
    tensor_b = Tensor(b)
    tensor_t = Tensor(t)
    tensor_y = pmath.log(tensor_x @ tensor_W + tensor_b)
    tensor_loss = pmath.sum(tensor_t - tensor_y) / 32
    tensor_loss.backward()

    torch_W = torch.Tensor(W)
    torch_W.requires_grad = True
    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    torch_b = torch.Tensor(b)
    torch_b.requires_grad = True
    torch_t = torch.Tensor(t)
    torch_t.requires_grad = True
    torch_y = torch.log(torch_x @ torch_W + torch_b)
    torch_loss = torch.sum(torch_t - torch_y) / 32
    torch_loss.backward()

    for tensor, torch_tensor in [
        (tensor_W, torch_W),
        (tensor_x, torch_x),
        (tensor_b, torch_b),
    ]:
        assert np.allclose(tensor.grad, torch_tensor.grad.numpy())

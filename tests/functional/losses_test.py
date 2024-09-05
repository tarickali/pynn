import numpy as np
import torch
import torch.nn.functional as G
from pynn.core import Tensor
import pynn.functional as F
import pynn.nn.losses as L

from tensorflow import keras


def get_data(input_dim: int = 32, output_dim: int = 1):
    np.random.seed()
    x = np.random.randn(32, input_dim)
    W = np.random.randn(input_dim, output_dim)
    b = np.zeros(output_dim)
    y = np.random.randn(32, output_dim)

    ptensor_W = Tensor(W)
    ptensor_x = Tensor(x)
    ptensor_b = Tensor(b)
    ptensor_z = ptensor_x @ ptensor_W + ptensor_b

    torch_W = torch.Tensor(W)
    torch_W.requires_grad = True
    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    torch_b = torch.Tensor(b)
    torch_b.requires_grad = True
    torch_z = torch_x @ torch_W + torch_b

    return (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    )


def test_regression_losses():
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 16)

    ptensor_o = F.relu(ptensor_z)
    ptensor_loss = L.MeanAbsoluteError()(Tensor(y), ptensor_o)
    ptensor_loss.backward()

    torch_o = G.relu(torch_z)
    torch_loss = torch.nn.L1Loss()(torch.Tensor(y), torch_o)
    torch_loss.backward()

    for i, (ptensor, tensor) in enumerate(
        [(ptensor_W, torch_W), (ptensor_x, torch_x), (ptensor_b, torch_b)]
    ):
        assert np.allclose(ptensor.data, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)


def test_binary_loss():
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 1)

    y = np.random.randint(0, 2, (32, 1))

    ptensor_o = F.sigmoid(ptensor_z)
    ptensor_loss = L.BinaryCrossentropy(logits=False)(Tensor(y), ptensor_o)
    ptensor_loss.backward()

    torch_o = G.sigmoid(torch_z)
    torch_o.retain_grad()
    torch_loss = torch.nn.BCELoss()(torch_o, torch.Tensor(y))
    torch_loss.retain_grad()
    torch_loss.backward()

    keras_loss = keras.losses.BinaryCrossentropy(False)(y, ptensor_o.data)

    for i, (ptensor, tensor) in enumerate(
        [(ptensor_W, torch_W), (ptensor_x, torch_x), (ptensor_b, torch_b)]
    ):
        assert np.allclose(ptensor.data, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)


def test_multi_loss():
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 10)

    y = np.eye(10)[np.random.choice(10, 32)]

    ptensor_o = ptensor_z
    ptensor_loss = L.CategoricalCrossentropy(logits=True)(Tensor(y), ptensor_o)
    ptensor_loss.backward()

    torch_o = torch_z
    torch_o.retain_grad()
    torch_loss = torch.nn.CrossEntropyLoss()(torch_o, torch.Tensor(y))
    torch_loss.retain_grad()
    torch_loss.backward()

    for i, (ptensor, tensor) in enumerate(
        [(ptensor_W, torch_W), (ptensor_x, torch_x), (ptensor_b, torch_b)]
    ):
        assert np.allclose(ptensor.data, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)

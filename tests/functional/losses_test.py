import numpy as np
import pytest

import pynn.functional as F
import pynn.nn.losses as L
from pynn.core import Tensor

torch = pytest.importorskip("torch", reason="comparison against PyTorch is optional")
import torch.nn.functional as G  # noqa: E402

pytestmark = pytest.mark.external


def get_data(input_dim: int = 32, output_dim: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((32, input_dim))
    W = rng.standard_normal((input_dim, output_dim))
    b = np.zeros(output_dim)
    y = rng.standard_normal((32, output_dim))

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

    for ptensor, tensor in [
        (ptensor_W, torch_W),
        (ptensor_x, torch_x),
        (ptensor_b, torch_b),
    ]:
        assert np.allclose(ptensor.data, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)


def test_binary_loss():
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 1, seed=1)

    y = np.random.default_rng(1).integers(0, 2, (32, 1)).astype(np.float64)

    ptensor_o = F.sigmoid(ptensor_z)
    ptensor_loss = L.BinaryCrossentropy(logits=False)(Tensor(y), ptensor_o)
    ptensor_loss.backward()

    torch_o = G.sigmoid(torch_z)
    torch_o.retain_grad()
    torch_loss = torch.nn.BCELoss()(torch_o, torch.Tensor(y))
    torch_loss.backward()

    for ptensor, tensor in [
        (ptensor_W, torch_W),
        (ptensor_x, torch_x),
        (ptensor_b, torch_b),
    ]:
        assert np.allclose(ptensor.data, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)


def test_binary_loss_from_logits():
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 1, seed=2)

    y = np.random.default_rng(2).integers(0, 2, (32, 1)).astype(np.float64)

    L.BinaryCrossentropy(logits=True)(Tensor(y), ptensor_z).backward()

    torch.nn.BCEWithLogitsLoss()(torch_z, torch.Tensor(y)).backward()

    for ptensor, tensor in [
        (ptensor_W, torch_W),
        (ptensor_x, torch_x),
        (ptensor_b, torch_b),
    ]:
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)


def test_multi_loss():
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 10, seed=3)

    y = np.eye(10)[np.random.default_rng(3).choice(10, 32)]

    L.CategoricalCrossentropy(logits=True)(Tensor(y), ptensor_z).backward()

    torch.nn.CrossEntropyLoss()(torch_z, torch.Tensor(y)).backward()

    for ptensor, tensor in [
        (ptensor_W, torch_W),
        (ptensor_x, torch_x),
        (ptensor_b, torch_b),
    ]:
        assert np.allclose(ptensor.data, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)


def test_multi_loss_from_probabilities():
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 10, seed=4)

    y = np.eye(10)[np.random.default_rng(4).choice(10, 32)]

    L.CategoricalCrossentropy(logits=False)(Tensor(y), F.softmax(ptensor_z)).backward()

    torch.nn.CrossEntropyLoss()(torch_z, torch.Tensor(y)).backward()

    for ptensor, tensor in [
        (ptensor_W, torch_W),
        (ptensor_x, torch_x),
        (ptensor_b, torch_b),
    ]:
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)


def test_kl_divergence_matches_torch():
    """Against `batchmean`, which is the reduction that is actually a KL divergence.

    PyTorch's `KLDivLoss` also takes log-probabilities where this one takes logits and
    fuses the `log_softmax` in, so the call sites differ on purpose — the values and
    the gradients they produce must not.
    """
    (
        (ptensor_x, ptensor_W, ptensor_b, ptensor_z),
        (torch_x, torch_W, torch_b, torch_z),
        _,
    ) = get_data(32, 10, seed=5)

    y = np.random.default_rng(5).dirichlet(np.ones(10), 32)

    ptensor_loss = L.KLDivLoss(logits=True)(Tensor(y), ptensor_z)
    ptensor_loss.backward()

    torch_loss = torch.nn.KLDivLoss(reduction="batchmean")(
        G.log_softmax(torch_z, dim=-1), torch.Tensor(y)
    )
    torch_loss.backward()

    assert np.allclose(ptensor_loss.data, torch_loss.detach().numpy(), atol=1e-6)
    for ptensor, tensor in [
        (ptensor_W, torch_W),
        (ptensor_x, torch_x),
        (ptensor_b, torch_b),
    ]:
        assert np.allclose(ptensor.grad, tensor.grad.numpy(), atol=1e-6)

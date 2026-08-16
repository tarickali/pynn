"""NAdam against `torch.optim.NAdam`, and where the two legitimately disagree.

The closed-form reference in `optimizers_test.py` is the primary check — it is the
published rule, transcribed independently. This one is the cross-check against a
second implementation, and it is worth reading for what it had to tolerate.

PyTorch keeps NAdam's `mu_product` and `step` as **float32** tensors regardless of the
parameter's dtype. On a float64 parameter that shows up as a relative error of about
1e-8 in the step, which compounds to roughly 1e-10 in the parameter over a few
iterations. This library keeps `mu_product` as a Python float, matches the float64
reference exactly, and therefore cannot match torch to float64 precision. The
tolerance below is sized for torch's error, not for this library's.
"""

import numpy as np
import pytest

from pynn.core import Tensor
from pynn.optim import NAdam

torch = pytest.importorskip("torch", reason="comparison against PyTorch is optional")

pytestmark = pytest.mark.external

STEPS = 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"weight_decay": 0.1},
        {"weight_decay": 0.1, "decoupled_weight_decay": True},
        {"momentum_decay": 0.02},
        {"maximize": True},
        {"beta_1": 0.8, "beta_2": 0.99},
    ],
    ids=str,
)
def test_nadam_matches_torch(kwargs):
    rng = np.random.default_rng(0)
    start = rng.standard_normal(5)
    gradients = [rng.standard_normal(5) for _ in range(STEPS)]

    param = Tensor(start.copy())
    optimizer = NAdam([{"w": param}], learning_rate=0.01, **kwargs)

    torch_kwargs = dict(kwargs)
    betas = (torch_kwargs.pop("beta_1", 0.9), torch_kwargs.pop("beta_2", 0.999))
    torch_param = torch.tensor(start.copy(), requires_grad=True)
    torch_optimizer = torch.optim.NAdam(
        [torch_param], lr=0.01, betas=betas, **torch_kwargs
    )

    for gradient in gradients:
        param.grad = gradient.copy()
        optimizer.update()

        torch_param.grad = torch.tensor(gradient.copy())
        torch_optimizer.step()

        assert np.allclose(param.data, torch_param.detach().numpy(), atol=1e-8)

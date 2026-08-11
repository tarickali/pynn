"""PyNN: A PyTorch-like neural network library.

Usage (PyTorch-style):
    import pynn
    from pynn.nn import Linear, ReLU, CrossEntropyLoss
    from pynn.optim import SGD

    model = pynn.nn.Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 10),
    ])
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model, learning_rate=0.01)
"""

from pynn.core import (
    Loss,
    Module,
    Tensor,
    enable_grad,
    is_grad_enabled,
    no_grad,
    set_grad_enabled,
)
from pynn.optim import SGD, Adadelta, Adagrad, Adam, Optimizer, RMSprop

#: The one place the version is written. `pyproject.toml` declares
#: `dynamic = ["version"]` and reads this attribute, so a release is a single edit here
#: and there is no second literal to disagree with it. Keep it a plain string:
#: setuptools parses this file rather than importing it, and importing it would need
#: numpy, which an isolated build environment does not have.
__version__ = "0.1.0"

__all__ = [
    "SGD",
    "Adadelta",
    "Adagrad",
    "Adam",
    "Loss",
    "Module",
    "Optimizer",
    "RMSprop",
    "Tensor",
    "enable_grad",
    "is_grad_enabled",
    "no_grad",
    "set_grad_enabled",
]

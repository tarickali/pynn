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
    optimizer = SGD(model.parameters, learning_rate=0.01)
"""

from pynn.core import Tensor, Module, Model, Loss
from pynn.optim import Optimizer, SGD, Adam, RMSprop, Adagrad, Adadelta

__all__ = [
    "Tensor",
    "Module",
    "Model",
    "Loss",
    "Optimizer",
    "SGD",
    "Adam",
    "RMSprop",
    "Adagrad",
    "Adadelta",
]

from pynn.core import Module, Model
from pynn.nn.modules import Linear, Conv2d, Flatten, Activation
from pynn.nn.models import Sequential
from pynn.nn.activations import (
    Identity,
    Affine,
    ReLU,
    Sigmoid,
    Tanh,
    ELU,
    SELU,
    SoftPlus,
    Softmax,
)
from pynn.nn.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    MeanSquaredError,
    MeanAbsoluteError,
)

# PyTorch-style aliases
BCELoss = BinaryCrossentropy
CrossEntropyLoss = CategoricalCrossentropy
MSELoss = MeanSquaredError
L1Loss = MeanAbsoluteError

__all__ = [
    # Core
    "Module",
    "Model",
    # Layers / containers
    "Linear",
    "Conv2d",
    "Flatten",
    "Activation",
    "Sequential",
    # Activations
    "Identity",
    "Affine",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "ELU",
    "SELU",
    "SoftPlus",
    # Losses
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "MeanSquaredError",
    "MeanAbsoluteError",
    # PyTorch-style loss aliases
    "BCELoss",
    "CrossEntropyLoss",
    "MSELoss",
    "L1Loss",
]

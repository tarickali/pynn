from pynn.core import Model, Module
from pynn.nn.activations import (
    ELU,
    SELU,
    Affine,
    Identity,
    ReLU,
    Sigmoid,
    Softmax,
    SoftPlus,
    Tanh,
)
from pynn.nn.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    MeanAbsoluteError,
    MeanSquaredError,
)
from pynn.nn.models import Sequential
from pynn.nn.modules import Activation, Conv2d, Flatten, Linear

# PyTorch-style aliases
BCELoss = BinaryCrossentropy
CrossEntropyLoss = CategoricalCrossentropy
MSELoss = MeanSquaredError
L1Loss = MeanAbsoluteError

__all__ = [
    "ELU",
    "SELU",
    "Activation",
    "Affine",
    # PyTorch-style loss aliases
    "BCELoss",
    # Losses
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "Conv2d",
    "CrossEntropyLoss",
    "Flatten",
    # Activations
    "Identity",
    "L1Loss",
    # Layers / containers
    "Linear",
    "MSELoss",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "Model",
    # Core
    "Module",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "SoftPlus",
    "Softmax",
    "Tanh",
]

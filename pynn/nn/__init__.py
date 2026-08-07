from pynn.core import Module
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
from pynn.nn.containers import ModuleDict, ModuleList
from pynn.nn.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    MeanAbsoluteError,
    MeanSquaredError,
)
from pynn.nn.models import Sequential
from pynn.nn.modules import (
    Activation,
    AvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2d,
)

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
    "AvgPool2d",
    # PyTorch-style loss aliases
    "BCELoss",
    "BatchNorm1d",
    "BatchNorm2d",
    # Losses
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Flatten",
    # Activations
    "Identity",
    "L1Loss",
    "LayerNorm",
    # Layers / containers
    "Linear",
    "MSELoss",
    "MaxPool2d",
    "MeanAbsoluteError",
    "MeanSquaredError",
    # Core
    "Module",
    "ModuleDict",
    "ModuleList",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "SoftPlus",
    "Softmax",
    "Tanh",
]

from pynn.core import Module
from pynn.nn.activations import (
    ELU,
    GELU,
    SELU,
    Affine,
    Identity,
    LogSoftmax,
    PReLU,
    ReLU,
    Sigmoid,
    SiLU,
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
    "GELU",
    "SELU",
    "Activation",
    "Affine",
    "AvgPool2d",
    "BCELoss",
    "BatchNorm1d",
    "BatchNorm2d",
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Flatten",
    "Identity",
    "L1Loss",
    "LayerNorm",
    "Linear",
    "LogSoftmax",
    "MSELoss",
    "MaxPool2d",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "Module",
    "ModuleDict",
    "ModuleList",
    "PReLU",
    "ReLU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SoftPlus",
    "Softmax",
    "Tanh",
]

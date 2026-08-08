"""Embedding lookup and recurrent cells.

A cell is applied once per timestep with the *same* weights, so the loop that unrolls it
puts every parameter on the tape once per step. That shape is what backpropagation
through time is, and it is the one a reverse pass that assigns instead of accumulating
gets wrong — a 30-step sequence would train on the last step only, and still converge
slowly enough to look like a learning-rate problem.

It is also why `Tensor.backward` uses an explicit stack: an unrolled recurrence is a
graph thousands of nodes deep, and a recursive traversal exhausts the interpreter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import pynn.functional as F
from pynn.core import Module, Tensor
from pynn.core.types import Array, Shape
from pynn.functional.modules import embedding, linear
from pynn.nn.factories import initializer_factory

__all__ = ["Embedding", "LSTMCell", "RNNCell"]


class Embedding(Module):
    """A lookup table mapping integer indices to dense vectors.

    Equivalent to multiplying a one-hot matrix by a weight matrix, without ever forming
    the one-hot: for a 50,000-token vocabulary that matrix is 50,000 columns of zeros
    per token, and every operation on it is a multiply by zero.

    Parameters
    ----------
    num_embeddings : int
        Number of rows — vocabulary size.
    embedding_dim : int
        Width of each vector.
    initializer : str | dict[str, Any]
        How to fill the table. Normal with unit variance by default, matching PyTorch.

    Examples
    --------
    >>> import numpy as np
    >>> table = Embedding(10, 4)
    >>> table(np.array([[1, 2], [3, 1]])).shape
    (2, 2, 4)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        initializer: str | dict[str, Any] = "random_normal",
        name: str = "Embedding",
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.initializer = initializer
        self.name = name

        self.weight_init = initializer_factory(initializer)
        self.register_parameter("W", self.weight_init((num_embeddings, embedding_dim)))
        self.initialized = True

    def forward(self, X: Array | Tensor) -> Tensor:  # type: ignore[override]
        """Look up `X`, which holds indices rather than activations."""
        return embedding(self.parameters["W"], X)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "initializer": self.initializer,
        }


class _Cell(Module):
    """Shared plumbing for the recurrent cells.

    Both hold input-to-hidden and hidden-to-hidden weights, infer `input_size` on the
    first call, and need a zero hidden state when none is given.
    """

    #: Hidden-state multiple the gates need: 1 for a plain RNN, 4 for an LSTM.
    gates: int = 1

    def __init__(
        self,
        *dims: int,
        include_bias: bool = True,
        weight_initializer: str | dict[str, Any] = "xavier_uniform",
        name: str | None = None,
    ) -> None:
        super().__init__()
        if len(dims) == 1:
            self.input_size: int | None = None
            self.hidden_size = dims[0]
        elif len(dims) == 2:
            self.input_size, self.hidden_size = dims
        else:
            raise TypeError(
                f"{type(self).__name__}() takes 1 or 2 positional dimension arguments "
                f"(hidden_size, or input_size and hidden_size), got {len(dims)}"
            )
        self.include_bias = include_bias
        self.weight_initializer = weight_initializer
        if name is not None:
            self.name = name

        self.weight_init = initializer_factory(weight_initializer)

    def build(self, input_shape: Shape) -> None:
        if len(input_shape) != 2:
            raise ValueError(
                f"{self.name} expects input of shape (batch, input_size), got "
                f"{input_shape}"
            )
        if self.input_size is None:
            self.input_size = input_shape[1]
        elif input_shape[1] != self.input_size:
            raise ValueError(
                f"expected input_size {self.input_size}, got {input_shape[1]}"
            )

        width = self.gates * self.hidden_size
        self.register_parameter("W_ih", self.weight_init((self.input_size, width)))
        self.register_parameter("W_hh", self.weight_init((self.hidden_size, width)))
        if self.include_bias:
            self.register_parameter("b_ih", np.zeros(width))
            self.register_parameter("b_hh", np.zeros(width))
        self.initialized = True

    def _zeros(self, batch: int) -> Tensor:
        return Tensor(np.zeros((batch, self.hidden_size)))

    def _gates(self, X: Tensor, h: Tensor) -> Tensor:
        """`X @ W_ih + h @ W_hh`, plus biases — the affine part every cell shares."""
        parameters = self.parameters
        return linear(X, parameters["W_ih"], parameters.get("b_ih")) + linear(
            h, parameters["W_hh"], parameters.get("b_hh")
        )

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "include_bias": self.include_bias,
            "weight_initializer": self.weight_initializer,
        }


class RNNCell(_Cell):
    """One step of an Elman RNN: `h' = tanh(X @ W_ih + h @ W_hh + b)`.

    A cell, not a layer: it takes and returns a hidden state, and the caller writes the
    loop over timesteps. That is deliberate — the loop is where variable-length
    sequences, teacher forcing, and truncated backpropagation live, and hiding it inside
    a layer would mean re-exposing all three as flags.

    Shape can be given or inferred: `RNNCell(hidden_size)` reads `input_size` from the
    first batch, `RNNCell(input_size, hidden_size)` states both.

    Parameters
    ----------
    *dims : int
        `hidden_size`, or `input_size` and `hidden_size`.
    nonlinearity : str, default "tanh"
        Activation applied to the gate sum. "tanh" or "relu".
    include_bias : bool, default True
    weight_initializer : str | dict[str, Any]

    Examples
    --------
    >>> import numpy as np
    >>> cell = RNNCell(4, 8)
    >>> h = None
    >>> for step in range(3):                      # the loop is yours
    ...     h = cell(Tensor(np.zeros((2, 4))), h)
    >>> h.shape
    (2, 8)
    """

    def __init__(
        self,
        *dims: int,
        nonlinearity: str = "tanh",
        include_bias: bool = True,
        weight_initializer: str | dict[str, Any] = "xavier_uniform",
        name: str = "RNNCell",
    ) -> None:
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(
                f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}"
            )
        super().__init__(
            *dims,
            include_bias=include_bias,
            weight_initializer=weight_initializer,
            name=name,
        )
        # After super().__init__(), which is what Module.__setattr__ requires.
        self.nonlinearity = nonlinearity

    def forward(self, X: Tensor, h: Tensor | None = None) -> Tensor:  # type: ignore[override]
        """Advance one timestep, returning the new hidden state."""
        if not self.initialized:
            self.build(X.shape)
        if h is None:
            h = self._zeros(X.shape[0])

        gates = self._gates(X, h)
        return F.tanh(gates) if self.nonlinearity == "tanh" else F.relu(gates)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {**super().hyperparameters, "nonlinearity": self.nonlinearity}


class LSTMCell(_Cell):
    """One step of an LSTM, returning `(h, c)`.

    The four gates are computed as a single matrix multiply into a `4 * hidden_size`
    block and then split, which is one `gemm` per step instead of four. The order —
    input, forget, cell, output — matches PyTorch, so a checkpoint's weights mean the
    same thing in both.

    The cell state `c` is the reason an LSTM holds information across many steps: it is
    updated additively, so a gradient flowing back through it is multiplied by the
    forget gate rather than by a saturating derivative each step.

    Parameters
    ----------
    *dims : int
        `hidden_size`, or `input_size` and `hidden_size`.
    include_bias : bool, default True
    weight_initializer : str | dict[str, Any]

    Examples
    --------
    >>> import numpy as np
    >>> cell = LSTMCell(4, 8)
    >>> h, c = cell(Tensor(np.zeros((2, 4))))
    >>> h.shape, c.shape
    ((2, 8), (2, 8))
    """

    gates = 4

    def __init__(
        self,
        *dims: int,
        include_bias: bool = True,
        weight_initializer: str | dict[str, Any] = "xavier_uniform",
        name: str = "LSTMCell",
    ) -> None:
        super().__init__(
            *dims,
            include_bias=include_bias,
            weight_initializer=weight_initializer,
            name=name,
        )

    def forward(  # type: ignore[override]
        self, X: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Advance one timestep, returning `(hidden, cell)`.

        Returns a pair rather than a Tensor, which is why this overrides `forward`'s
        signature: an LSTM carries two states, and folding them into one would mean
        splitting them again at every call site.
        """
        if not self.initialized:
            self.build(X.shape)
        if state is None:
            h, c = self._zeros(X.shape[0]), self._zeros(X.shape[0])
        else:
            h, c = state

        combined = self._gates(X, h)
        size = self.hidden_size
        # One gemm, then split — rather than four multiplies against four matrices.
        input_gate = F.sigmoid(combined[:, :size])
        forget_gate = F.sigmoid(combined[:, size : 2 * size])
        candidate = F.tanh(combined[:, 2 * size : 3 * size])
        output_gate = F.sigmoid(combined[:, 3 * size :])

        next_c = forget_gate * c + input_gate * candidate
        next_h = output_gate * F.tanh(next_c)
        return next_h, next_c

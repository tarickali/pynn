"""Embedding and the recurrent cells.

Gradients are checked against central differences by the verify sweep, including the
cells unrolled over several steps. What is here is the behaviour around them: that the
weights really are shared across timesteps, that a deep unroll does not exhaust the
interpreter, and that an embedding's scatter accumulates on repeated tokens.
"""

import numpy as np
import pytest

import pynn.core.math as pmath
from pynn.core import Tensor
from pynn.core.random import set_seed
from pynn.nn import Embedding, Linear, LSTMCell, RNNCell
from pynn.nn.losses import SparseCategoricalCrossentropy
from pynn.optim import Adam

# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


def test_embedding_looks_up_rows():
    table = Embedding(5, 3)
    table.parameters["W"].data = np.arange(15.0).reshape(5, 3)

    assert np.array_equal(table(np.array([0, 4])).data, [[0.0, 1.0, 2.0], [12, 13, 14]])


def test_embedding_preserves_the_index_shape():
    table = Embedding(10, 4)

    assert table(np.array([1, 2])).shape == (2, 4)
    assert table(np.array([[1, 2], [3, 4]])).shape == (2, 2, 4)


def test_embedding_registers_its_table():
    table = Embedding(10, 4)

    assert sorted(table.named_parameters()) == ["W"]
    assert table.num_parameters() == 40
    assert "W" in table.state_dict()


def test_a_repeated_token_accumulates_its_gradient():
    """The whole point of an embedding: common tokens appear many times per batch."""
    table = Embedding(4, 2)

    pmath.sum(table(np.array([1, 1, 1, 3]))).backward()

    assert table.parameters["W"].grad[:, 0].tolist() == [0.0, 3.0, 0.0, 1.0]


def test_embedding_accepts_a_tensor_of_indices():
    table = Embedding(5, 3)
    indices = Tensor(np.array([0, 2]), dtype=np.intp)

    assert table(indices).shape == (2, 3)


@pytest.mark.parametrize("indices", [[-1, 0], [0, 5]], ids=["negative", "too-large"])
def test_embedding_rejects_an_out_of_range_index(indices):
    with pytest.raises(ValueError, match=r"indices must be in \[0, 5\)"):
        Embedding(5, 3)(np.array(indices))


def test_embedding_matches_a_one_hot_matmul(rng):
    """It is that product, computed without ever forming the one-hot."""
    table = Embedding(6, 4)
    labels = np.array([0, 3, 3, 5])

    looked_up = table(labels).data
    one_hot = np.eye(6)[labels] @ table.parameters["W"].data

    assert np.allclose(looked_up, one_hot)


# --------------------------------------------------------------------------- #
# RNNCell
# --------------------------------------------------------------------------- #


def test_rnn_cell_returns_a_hidden_state(rng):
    cell = RNNCell(4, 8)

    h = cell(Tensor(rng.standard_normal((3, 4))))

    assert isinstance(h, Tensor)
    assert h.shape == (3, 8)


def test_rnn_cell_defaults_to_a_zero_hidden_state():
    cell = RNNCell(4, 8)
    X = Tensor(np.zeros((2, 4)))

    implicit = cell(X)
    explicit = cell(X, Tensor(np.zeros((2, 8))))

    assert np.allclose(implicit.data, explicit.data)


def test_rnn_cell_infers_its_input_size():
    cell = RNNCell(8)
    assert cell.input_size is None

    cell(Tensor(np.zeros((2, 5))))

    assert cell.input_size == 5
    assert cell.parameters["W_ih"].shape == (5, 8)
    assert cell.parameters["W_hh"].shape == (8, 8)


def test_rnn_cell_matches_its_definition(rng):
    cell = RNNCell(3, 4)
    X = Tensor(rng.standard_normal((2, 3)))
    h = Tensor(rng.standard_normal((2, 4)))
    cell(X, h)  # build

    expected = np.tanh(
        X.data @ cell.parameters["W_ih"].data
        + cell.parameters["b_ih"].data
        + h.data @ cell.parameters["W_hh"].data
        + cell.parameters["b_hh"].data
    )
    assert np.allclose(cell(X, h).data, expected)


def test_rnn_cell_relu_nonlinearity(rng):
    cell = RNNCell(3, 4, nonlinearity="relu")

    assert np.all(cell(Tensor(rng.standard_normal((5, 3)))).data >= 0.0)


def test_rnn_cell_rejects_an_unknown_nonlinearity():
    with pytest.raises(ValueError, match="'tanh' or 'relu'"):
        RNNCell(3, 4, nonlinearity="sigmoid")


def test_a_cell_rejects_a_non_matrix_input():
    with pytest.raises(ValueError, match=r"\(batch, input_size\)"):
        RNNCell(3, 4)(Tensor(np.zeros((2, 3, 4))))


def test_a_cell_rejects_too_many_dimension_arguments():
    with pytest.raises(TypeError, match="1 or 2 positional"):
        RNNCell(1, 2, 3)


# --------------------------------------------------------------------------- #
# LSTMCell
# --------------------------------------------------------------------------- #


def test_lstm_cell_returns_a_hidden_and_a_cell_state(rng):
    cell = LSTMCell(4, 8)

    h, c = cell(Tensor(rng.standard_normal((3, 4))))

    assert h.shape == (3, 8)
    assert c.shape == (3, 8)


def test_lstm_gates_are_one_matrix(rng):
    """Four gates, one gemm: the weights are 4 * hidden_size wide."""
    cell = LSTMCell(4, 8)
    cell(Tensor(rng.standard_normal((2, 4))))

    assert cell.parameters["W_ih"].shape == (4, 32)
    assert cell.parameters["W_hh"].shape == (8, 32)


def test_lstm_cell_state_is_updated_additively(rng):
    """Why it holds information: c is gated and added to, not squashed each step."""
    cell = LSTMCell(3, 4)
    X = Tensor(np.zeros((2, 3)))
    # Force the forget gate wide open and the input gate shut.
    cell(X)  # build
    cell.parameters["b_ih"].data[:] = np.concatenate(
        [np.full(4, -20.0), np.full(4, 20.0), np.zeros(4), np.zeros(4)]
    )

    carried = Tensor(np.full((2, 4), 5.0))
    _, next_c = cell(X, (Tensor(np.zeros((2, 4))), carried))

    assert np.allclose(next_c.data, 5.0, atol=1e-6)


def test_lstm_cell_takes_the_state_it_returns(rng):
    cell = LSTMCell(3, 4)
    X = Tensor(rng.standard_normal((2, 3)))

    state = cell(X)
    again = cell(X, state)

    assert len(again) == 2
    assert again[0].shape == (2, 4)


# --------------------------------------------------------------------------- #
# Backpropagation through time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cell_type", [RNNCell, LSTMCell], ids=lambda c: c.__name__)
def test_weights_are_shared_across_timesteps(cell_type, rng):
    """One set of parameters, applied `steps` times — not one set per step."""
    cell = cell_type(3, 4)
    X = Tensor(rng.standard_normal((2, 3)))

    state = None
    for _ in range(5):
        state = cell(X) if state is None else cell(X, state)

    assert sorted(cell.named_parameters()) == ["W_hh", "W_ih", "b_hh", "b_ih"]


@pytest.mark.parametrize("cell_type", [RNNCell, LSTMCell], ids=lambda c: c.__name__)
def test_gradient_grows_with_the_number_of_steps(cell_type, rng):
    """Every step contributes to the same weights, so the gradient accumulates.

    A reverse pass that assigned instead of adding would give the same gradient for
    one step and for ten — and would still descend, just on the last step only.
    """
    X = Tensor(rng.standard_normal((2, 3)))

    def unrolled(steps: int) -> np.ndarray:
        set_seed(0)
        cell = cell_type(3, 4)
        state = None
        for _ in range(steps):
            state = cell(X) if state is None else cell(X, state)
        hidden = state[0] if isinstance(state, tuple) else state
        cell.zero_grad()
        pmath.sum(hidden).backward()
        return cell.parameters["W_hh"].grad.copy()

    one, many = unrolled(1), unrolled(6)

    # One step never uses W_hh against a non-zero hidden state, so its gradient is zero
    # there; six steps must be substantially non-zero.
    assert np.allclose(one, 0.0)
    assert np.abs(many).max() > 1e-6


@pytest.mark.parametrize("cell_type", [RNNCell, LSTMCell], ids=lambda c: c.__name__)
def test_a_deep_unroll_does_not_exhaust_the_interpreter(cell_type):
    """The reason `backward` uses an explicit stack rather than recursion.

    300 steps of an LSTM is a graph of tens of thousands of nodes; a recursive
    topological sort raises RecursionError well before that.
    """
    cell = cell_type(2, 3)
    X = Tensor(np.full((1, 2), 0.05))

    state = None
    for _ in range(300):
        state = cell(X) if state is None else cell(X, state)
    hidden = state[0] if isinstance(state, tuple) else state

    pmath.sum(hidden).backward()

    assert np.all(np.isfinite(cell.parameters["W_hh"].grad))


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cell_type", [RNNCell, LSTMCell], ids=lambda c: c.__name__)
def test_a_sequence_model_learns_to_predict_the_next_token(cell_type):
    """Embedding, a cell unrolled over six steps, and a head — trained end to end."""
    vocab, dim, hidden, steps, batch = 12, 8, 16, 6, 16
    rng = np.random.default_rng(0)
    starts = rng.integers(0, vocab, batch)
    sequences = np.array([[(s + t) % vocab for t in range(steps + 1)] for s in starts])
    inputs, targets = sequences[:, :-1], sequences[:, -1]

    set_seed(0)
    table = Embedding(vocab, dim)
    cell = cell_type(dim, hidden)
    head = Linear(hidden, vocab)
    modules = [table, cell, head]
    loss_fn = SparseCategoricalCrossentropy()

    def logits() -> Tensor:
        state = None
        for t in range(steps):
            x = table(inputs[:, t])
            state = cell(x) if state is None else cell(x, state)
        hidden_state = state[0] if isinstance(state, tuple) else state
        return head(hidden_state)

    optimizer = Adam([module.parameters for module in modules], learning_rate=0.05)
    first = float(loss_fn(targets, logits()).item())
    for _ in range(120):
        loss = loss_fn(targets, logits())
        for module in modules:
            module.zero_grad()
        loss.backward()
        optimizer.step()

    assert float(loss_fn(targets, logits()).item()) < first / 10
    assert (logits().data.argmax(axis=1) == targets).mean() == 1.0

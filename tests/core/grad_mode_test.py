"""Tape recording: `no_grad`, `detach`, and `requires_grad`.

Inference used to build a full graph and throw it away. That is wasted work per batch,
and an outright leak when the outputs are kept — each reverse closure holds the forward
pass's intermediate arrays, so collecting predictions over a validation set retains
every batch's graph. The weakref tests below are the direct statement of that.
"""

import gc
import weakref

import numpy as np
import pytest

import pynn.core.math as pmath
import pynn.functional as F
from pynn.core import Tensor, enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError


@pytest.fixture(autouse=True)
def restore_grad_mode():
    """Recording is global state; a failing test must not leave it off."""
    yield
    assert is_grad_enabled(), "a test left gradient recording disabled"


# --------------------------------------------------------------------------- #
# The mode itself
# --------------------------------------------------------------------------- #


def test_recording_is_on_by_default():
    assert is_grad_enabled()


def test_no_grad_turns_recording_off_for_the_block():
    with no_grad():
        assert not is_grad_enabled()
    assert is_grad_enabled()


def test_enable_grad_re_enables_inside_a_no_grad_block():
    with no_grad():
        with enable_grad():
            assert is_grad_enabled()
        assert not is_grad_enabled()


def test_the_previous_mode_is_restored_not_assumed():
    with no_grad(), set_grad_enabled(False):
        pass
    assert is_grad_enabled()


def test_the_mode_is_restored_after_an_exception():
    with pytest.raises(RuntimeError), no_grad():
        raise RuntimeError("something failed mid-inference")

    assert is_grad_enabled()


def test_no_grad_works_as_a_decorator():
    @no_grad()
    def predict(x: Tensor) -> Tensor:
        assert not is_grad_enabled()
        return F.tanh(x)

    assert not predict(Tensor(np.zeros((2, 2)))).requires_grad
    assert is_grad_enabled()


# --------------------------------------------------------------------------- #
# What no_grad does to the graph
# --------------------------------------------------------------------------- #


def test_operations_under_no_grad_record_no_children():
    x = Tensor(np.ones((2, 3)))

    with no_grad():
        output = pmath.sum(F.tanh(x) * 2.0)

    assert output.children == ()
    assert not output.requires_grad


def test_values_are_unaffected_by_the_mode(rng):
    x = Tensor(rng.standard_normal((3, 4)))
    recorded = F.sigmoid(x).data

    with no_grad():
        assert np.array_equal(F.sigmoid(x).data, recorded)


def test_backward_under_no_grad_is_an_error():
    x = Tensor(np.ones((2, 2)))

    with no_grad():
        loss = pmath.sum(x * 2.0)

    with pytest.raises(RuntimeError, match="does not require gradients"):
        loss.backward()


def test_no_grad_leaves_existing_gradients_alone(rng):
    x = Tensor(rng.standard_normal((3, 4)))
    pmath.sum(x * 2.0).backward()
    recorded = x.grad.copy()

    with no_grad():
        pmath.sum(x * 5.0)

    assert np.array_equal(x.grad, recorded)


def test_recording_resumes_after_the_block(rng):
    x = Tensor(rng.standard_normal((3, 4)))

    with no_grad():
        pmath.sum(x * 2.0)

    pmath.sum(x * 3.0).backward()
    assert np.allclose(x.grad, 3.0)


def test_no_grad_does_not_retain_the_intermediates():
    """The claim `no_grad` exists to make: the graph is not held alive."""
    x = Tensor(np.ones((2, 2)))

    with no_grad():
        hidden = F.tanh(x)
        output = pmath.sum(hidden)

    reference = weakref.ref(hidden)
    del hidden
    gc.collect()

    assert reference() is None
    assert output is not None  # the result itself is still usable


def test_recording_does_retain_the_intermediates():
    """The contrast, so the test above is not vacuously true."""
    x = Tensor(np.ones((2, 2)))
    hidden = F.tanh(x)
    output = pmath.sum(hidden)

    reference = weakref.ref(hidden)
    del hidden
    gc.collect()

    assert reference() is not None
    output.backward()


# --------------------------------------------------------------------------- #
# detach
# --------------------------------------------------------------------------- #


def test_detach_stops_the_gradient(rng):
    x = Tensor(rng.standard_normal((3, 4)))
    detached = x.detach()

    pmath.sum(detached * 2.0).backward()

    # The detached copy is an ordinary leaf and accumulates as one; what it does not
    # do is pass anything further back.
    assert np.allclose(detached.grad, 2.0)
    assert np.all(x.grad == 0.0)


def test_a_wholly_detached_graph_cannot_be_differentiated(rng):
    x = Tensor(rng.standard_normal((3, 4)))

    with pytest.raises(RuntimeError, match="does not require gradients"):
        pmath.sum(x.detach()).backward()


def test_detach_keeps_the_values_and_dtype(rng):
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    detached = x.detach()

    assert np.array_equal(detached.data, x.data)
    assert detached.dtype == np.float32
    assert not detached.requires_grad


def test_detach_is_a_copy_not_a_view():
    x = Tensor(np.ones((2, 2)))
    detached = x.detach()
    detached.data += 1.0

    assert np.all(x.data == 1.0)


def test_a_detached_branch_does_not_block_the_others(rng):
    """One detached operand must not stop the other from receiving a gradient."""
    a = Tensor(rng.standard_normal((3, 4)))
    b = Tensor(rng.standard_normal((3, 4)))

    pmath.sum(a * b.detach()).backward()

    assert np.allclose(a.grad, b.data)
    assert np.all(b.grad == 0.0)


# --------------------------------------------------------------------------- #
# requires_grad
# --------------------------------------------------------------------------- #


def test_a_new_tensor_requires_grad():
    assert Tensor(np.zeros(3)).requires_grad


def test_an_output_requires_grad_when_any_input_does(rng):
    a = Tensor(rng.standard_normal((2, 2)))

    assert (a + a.detach()).requires_grad
    assert not (a.detach() * a.detach()).requires_grad


def test_requires_grad_is_separate_from_trainable():
    """Freezing excludes a parameter from updates; it still receives gradients."""
    layer = Linear(4, 3)
    layer(Tensor(np.zeros((2, 4))))
    layer.freeze()

    weight = layer.parameters["W"]
    assert not weight.trainable
    assert weight.requires_grad

    pmath.sum(layer(Tensor(np.ones((2, 4))))).backward()
    assert np.any(weight.grad != 0.0)


# --------------------------------------------------------------------------- #
# In a training loop
# --------------------------------------------------------------------------- #


def test_an_evaluation_pass_under_no_grad_matches_a_recorded_one(rng):
    model = Sequential([Linear(4, 8, activation="tanh"), Linear(8, 2)])
    X = Tensor(rng.standard_normal((6, 4)))
    recorded = model(X).data

    model.eval()
    with no_grad():
        assert np.array_equal(model(X).data, recorded)


def test_evaluating_under_no_grad_does_not_disturb_training(rng):
    """The realistic loop: fit a batch, evaluate, and keep fitting."""
    X = Tensor(rng.standard_normal((8, 4)))
    y = Tensor(rng.standard_normal((8, 2)))
    loss_fn = MeanSquaredError()

    def train(evaluate: bool) -> float:
        from pynn.functional.initializers import set_seed

        set_seed(0)
        model = Sequential([Linear(4, 5, activation="tanh"), Linear(5, 2)])
        from pynn.optim import SGD

        optimizer = SGD(model, learning_rate=0.1)
        for _ in range(10):
            loss = loss_fn(y, model(X))
            model.zero_grad()
            loss.backward()
            optimizer.update()
            if evaluate:
                model.eval()
                with no_grad():
                    model(X)
                model.train()
        return float(loss_fn(y, model(X)).item())

    assert train(evaluate=True) == pytest.approx(train(evaluate=False))

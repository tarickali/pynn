import contextlib
import gc
import weakref

import numpy as np
import pytest

import pynn.core.math as pmath
import pynn.functional as F
from pynn.core import Tensor
from pynn.core.types import Array


def test_init(rng):
    # test : init with number
    data = 0
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    data = 1.0
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    # test : init with lists
    data = [0]
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    data = [[0.0, 1.0, 2.0], [1, 2, 3]]
    tensor = Tensor(data)
    assert tensor.data.tolist() == data
    assert tensor.dtype == np.float64
    assert isinstance(tensor.data, Array)

    # test : init with nddata
    data = rng.standard_normal((2, 3))
    tensor = Tensor(data)
    assert np.allclose(tensor.data, data)
    assert tensor.dtype == data.dtype
    assert isinstance(tensor.data, Array)

    data = np.zeros((2, 3))
    tensor = Tensor(data, np.float32)
    assert np.all(tensor.data == data)
    assert tensor.dtype == np.float32
    assert isinstance(tensor.data, Array)


def test_binary_operations(rng):
    #####----- test : add -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a + b
    x = Tensor(a)
    y = Tensor(b)
    z = x + y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a + b
    x = Tensor(a)
    y = Tensor(b)
    z = x + y
    assert isinstance(z, Tensor)
    assert z.shape == c.shape
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : sub -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : mul -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : matmul -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 2))
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    #####----- test : truediv -----#####
    # no broadcasting
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)

    # broadcasting
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal(4)
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.data, c)
    assert isinstance(z.data, Array)


def test_unary_operations(rng):
    #####----- test : pow -----#####
    a = rng.standard_normal((2, 3))
    b = a**2
    x = Tensor(a)
    y = x**2
    assert isinstance(y, Tensor)
    assert np.allclose(y.data, b)
    assert isinstance(y.data, Array)

    #####----- test : neg -----#####
    a = rng.standard_normal((2, 3))
    b = -a
    x = Tensor(a)
    y = -x
    assert isinstance(y, Tensor)
    assert np.allclose(y.data, b)
    assert isinstance(y.data, Array)


def test_comparison_operations(rng):
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    x = Tensor(a)
    y = Tensor(b)

    for op in [
        lambda u, v: u == v,
        lambda u, v: u != v,
        lambda u, v: u <= v,
        lambda u, v: u < v,
        lambda u, v: u >= v,
        lambda u, v: u > v,
    ]:
        expected = op(a, b)
        result = op(x, y)
        assert isinstance(result, Tensor)
        assert result.dtype == bool
        assert np.array_equal(result.data, expected)


def test_bool_rejects_a_multi_element_tensor():
    """`if a == b:` must not silently return True for every non-empty Tensor."""
    equal = Tensor([1.0, 2.0]) == Tensor([1.0, 2.0])
    unequal = Tensor([1.0, 2.0]) == Tensor([1.0, 3.0])

    assert equal.dtype == bool
    assert equal.all()
    assert not unequal.all()
    assert unequal.any()

    with pytest.raises(ValueError, match="ambiguous"):
        bool(equal)
    with pytest.raises(ValueError, match="ambiguous"):
        bool(unequal)
    with pytest.raises(ValueError, match="ambiguous"):
        if equal:
            pass


def test_bool_of_a_scalar_tensor():
    assert bool(Tensor(1.0)) is True
    assert bool(Tensor(0.0)) is False
    assert bool(Tensor([True], dtype=bool)) is True
    assert bool(Tensor([False], dtype=bool)) is False


def test_cast():
    x = Tensor([0, 1, 2])
    x.cast(int)
    assert x.dtype == int
    x.cast(np.float32)
    assert x.dtype == np.float32


# --------------------------------------------------------------------------- #
# dtype
#
# A float32 array used to be upcast to float64 on the way in, and every gradient was
# float64 regardless. Both are invisible until a model is twice the size and half the
# speed it was meant to be.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_a_floating_input_keeps_its_precision(dtype, rng):
    tensor = Tensor(rng.standard_normal((2, 3)).astype(dtype))

    assert tensor.dtype == dtype
    assert tensor.grad.dtype == dtype


@pytest.mark.parametrize(
    "data", [[1, 2, 3], 4, 4.0, np.array([1, 2, 3]), np.array([True, False])]
)
def test_a_non_floating_input_is_promoted_to_float64(data):
    tensor = Tensor(data)

    assert tensor.dtype == np.float64
    assert tensor.grad.dtype == np.float64


def test_an_explicit_dtype_still_wins():
    assert Tensor(np.zeros((2, 3), dtype=np.float64), np.float32).dtype == np.float32


def test_a_boolean_tensor_carries_a_float_gradient():
    """Comparisons produce bool Tensors; a gradient is real-valued regardless."""
    result = Tensor([1.0, 2.0]) == Tensor([1.0, 3.0])

    assert result.dtype == bool
    assert result.grad.dtype == np.float64


def test_float32_survives_a_forward_and_backward_pass(rng):
    x = Tensor(rng.standard_normal((4, 3)).astype(np.float32))
    w = Tensor(rng.standard_normal((3, 2)).astype(np.float32))

    output = x @ w
    assert output.dtype == np.float32

    pmath.sum(output).backward()
    assert x.grad.dtype == np.float32
    assert w.grad.dtype == np.float32


def test_an_explicit_seed_gradient_does_not_upcast(rng):
    x = Tensor(rng.standard_normal((2, 3)).astype(np.float32))

    (x * 2.0).backward(gradient=np.ones((2, 3), dtype=np.float64))

    assert x.grad.dtype == np.float32


def test_zero_grad_preserves_the_gradient_dtype():
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    x.zero_grad()

    assert x.grad.dtype == np.float32


def test_cast_moves_the_gradient_dtype_with_the_data():
    x = Tensor(np.ones((2, 3), dtype=np.float64))
    x.cast(np.float32)

    assert x.grad.dtype == np.float32


# --------------------------------------------------------------------------- #
# Reflected operators
#
# `2.0 - x` reaches `__rsub__`, which is a different code path from `x - 2.0` and
# gets the operand order wrong if it just delegates to `__sub__`.
# --------------------------------------------------------------------------- #


def test_reflected_operators_use_the_right_operand_order(rng):
    a = rng.standard_normal((2, 3))
    x = Tensor(a)

    assert np.allclose((2.0 + x).data, 2.0 + a)
    assert np.allclose((2.0 - x).data, 2.0 - a)
    assert np.allclose((2.0 * x).data, 2.0 * a)
    assert np.allclose((2.0 / x).data, 2.0 / a)


def test_reflected_matmul_uses_the_right_operand_order(rng):
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 4))

    assert np.allclose((a @ Tensor(b)).data, a @ b)


def test_an_ndarray_on_the_left_still_produces_a_tensor(rng):
    """NumPy used to win the dispatch and return an object-dtype array of Tensors.

    That is silently wrong: the result looks like an array, has no gradient, and
    every subsequent operation on it is Python-level object arithmetic.
    """
    a = rng.standard_normal((2, 3))
    x = Tensor(rng.standard_normal((2, 3)))

    for result in [a + x, a - x, a * x, a / x]:
        assert isinstance(result, Tensor)
        assert result.dtype == np.float64


def test_an_ndarray_on_the_left_stays_on_the_tape(rng):
    import pynn.core.math as pmath

    x = Tensor(rng.standard_normal((2, 3)))
    pmath.sum(np.full((2, 3), 3.0) * x).backward()

    assert np.allclose(x.grad, 3.0)


# --------------------------------------------------------------------------- #
# Re-running one graph
#
# Pinned, not endorsed. `backward` never zeroes what it finds, which is what makes
# gradient accumulation over micro-batches work — but that contract is about separate
# forward passes over the same leaves, and it does not extend to walking one finished
# graph twice. TASKS.md item 4 has the options for closing this.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("depth", "one_pass", "compounded"),
    [(1, 3.0, 12.0), (2, 9.0, 45.0), (4, 81.0, 567.0)],
    ids=["depth-1", "depth-2", "depth-4"],
)
def test_a_second_backward_over_one_graph_compounds(depth, one_pass, compounded):
    """Two passes over one graph do not double its gradient — they compound it.

    Every Tensor keeps a `grad`, intermediates included, and each reverse closure reads
    its output's *stored* gradient. So the second pass finds the first pass's values
    still sitting on every intermediate and propagates them again, and the overshoot
    grows with depth. PyTorch stores gradients on leaves only, which is why
    `backward(retain_graph=True)` there doubles exactly; its default refuses the second
    pass outright, as `free_graph` does here.

    This test exists so the numbers cannot drift unnoticed. It asserts what the library
    does, not what it should do.
    """
    x = Tensor(np.ones((2, 2)))
    node = x
    for _ in range(depth):
        node = node * 3.0
    loss = pmath.sum(node)

    loss.backward()
    assert np.allclose(x.grad, one_pass)

    loss.backward()
    assert np.allclose(x.grad, compounded)
    assert not np.allclose(x.grad, 2 * one_pass), "doubling would be the right answer"


def test_two_passes_over_separate_graphs_do_double():
    """The contrast, and the form the accumulation contract is actually about."""
    x = Tensor(np.ones((2, 2)))

    for _ in range(2):
        pmath.sum(x * 3.0).backward()

    assert np.allclose(x.grad, 6.0)


# --------------------------------------------------------------------------- #
# free_graph
#
# A finished graph is a reference cycle — every reverse closure references the Tensor
# it belongs to — so nothing reclaims it until the cyclic collector runs, which CPython
# schedules from object counts rather than from the megabytes of arrays hanging off
# them. `free_graph` breaks the cycles explicitly. The two things it must not do are
# change a gradient and fail quietly.
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def no_cyclic_collector():
    """Run with CPython's cyclic collector off.

    The claim under test is that reference counting alone reclaims a freed graph. With
    the collector running, a collection triggered by unrelated allocation would satisfy
    the assertion for the wrong reason.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def diamond(seed):
    """A graph whose input feeds three consumers, and its leaves.

    The shape that separates freeing correctly from freeing by luck: a single-consumer
    chain is reclaimed by any traversal that reaches the end of it, and gives the same
    gradients however the graph is torn down afterwards.
    """
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal((4, 3)))
    w = Tensor(rng.standard_normal((3, 3)))
    projected = x @ w
    hidden = F.tanh(projected)
    return x, w, pmath.sum(hidden * hidden + projected + x)


def test_free_graph_leaves_the_gradients_a_reused_input_earned():
    """Freeing must not disturb what backward computed, and the reuse is the point."""
    x_kept, w_kept, kept = diamond(0)
    kept.backward()

    x_freed, w_freed, freed = diamond(0)
    freed.backward()
    freed.free_graph()

    assert np.array_equal(x_freed.grad, x_kept.grad)
    assert np.array_equal(w_freed.grad, w_kept.grad)
    assert freed.item() == kept.item()


def test_free_graph_empties_the_output_and_keeps_the_leaves():
    x, w, loss = diamond(1)
    loss.backward()
    loss.free_graph()

    assert loss.children == ()
    # The leaves are the caller's: an optimizer is about to read those gradients.
    assert x.children == () and np.any(x.grad != 0.0)
    assert np.any(w.grad != 0.0)


def test_free_graph_reclaims_the_tape_without_the_collector():
    """The claim `free_graph` exists to make."""
    with no_cyclic_collector():
        x = Tensor(np.ones((2, 2)))
        projected = x * 2.0
        hidden = F.tanh(projected)
        loss = pmath.sum(hidden * projected)
        loss.backward()

        alive = [weakref.ref(node) for node in (projected, hidden)]
        del projected, hidden
        assert all(reference() is not None for reference in alive)

        loss.free_graph()
        assert all(reference() is None for reference in alive)


def test_a_finished_graph_is_not_reclaimed_on_its_own():
    """The contrast, so the test above is not vacuously true."""
    with no_cyclic_collector():
        x = Tensor(np.ones((2, 2)))
        hidden = F.tanh(x)
        loss = pmath.sum(hidden)
        loss.backward()

        reference = weakref.ref(hidden)
        del hidden

        assert reference() is not None
        assert loss is not None


def test_backward_after_free_graph_raises():
    """A freed graph that quietly returned zeros would look like a converged model."""
    x, _, loss = diamond(2)
    loss.backward()
    recorded = x.grad.copy()
    loss.free_graph()

    with pytest.raises(RuntimeError, match="released by free_graph"):
        loss.backward()

    # Refused before anything was accumulated, seed included.
    assert np.array_equal(x.grad, recorded)


def test_backward_through_a_freed_node_raises():
    """Freeing one graph poisons the tensors it shared, and must say so."""
    x = Tensor(np.ones((2, 2)))
    hidden = F.tanh(x)

    first = pmath.sum(hidden)
    first.backward()
    first.free_graph()

    with pytest.raises(RuntimeError, match="released by free_graph"):
        pmath.sum(hidden * 3.0).backward()


def test_free_graph_is_idempotent():
    _, _, loss = diamond(3)
    loss.backward()
    loss.free_graph()
    loss.free_graph()

    assert loss.children == ()


def test_free_graph_on_a_leaf_does_nothing():
    x = Tensor(np.ones((2, 2)))
    x.free_graph()

    # Not marked as freed: a leaf holds no piece of the tape, and marking it would
    # refuse every later graph built over the same parameter.
    pmath.sum(x * 2.0).backward()
    assert np.allclose(x.grad, 2.0)


def test_freeing_each_step_still_accumulates_across_steps():
    """The training-loop shape: new graph per step, same leaves, gradients add up."""
    x = Tensor(np.ones((2, 2)))

    for _ in range(3):
        loss = pmath.sum(x * 2.0)
        loss.backward()
        loss.free_graph()

    assert np.allclose(x.grad, 6.0)


# --------------------------------------------------------------------------- #
# Accessors and rejection paths
# --------------------------------------------------------------------------- #


def test_numpy_returns_the_underlying_array(rng):
    a = rng.standard_normal((2, 3))
    tensor = Tensor(a)

    assert isinstance(tensor.numpy(), Array)
    assert np.allclose(tensor.numpy(), a)


def test_item_returns_a_python_scalar():
    assert Tensor([[2.5]]).item() == 2.5


def test_shape_properties():
    tensor = Tensor(np.zeros((2, 3, 4)))

    assert tensor.shape == (2, 3, 4)
    assert tensor.ndim == 3
    assert tensor.size == 24


def test_repr_names_the_dtype_and_shape():
    text = repr(Tensor(np.zeros((2, 3))))

    assert text.startswith("Tensor(")
    assert "float64" in text
    assert "(2, 3)" in text


def test_pow_rejects_a_tensor_exponent():
    with pytest.raises(TypeError, match="Cannot perform operation"):
        Tensor([1.0, 2.0]) ** Tensor([2.0, 2.0])


def test_operations_reject_an_unconvertible_operand():
    with pytest.raises(TypeError, match="Cannot perform operation"):
        Tensor([1.0]) + "two"

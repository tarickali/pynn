"""The activation and initializer factories.

Layers take activations and initializers as strings, so the factories are the only path
by which most of these classes are ever constructed. `pynn.verify.check_api` already
walks every advertised name; what these add are the argument *forms* — None, a dict with
params, an already-constructed instance — and the rejection paths.
"""

import numpy as np
import pytest

from pynn.core import Tensor
from pynn.nn.activations import ELU, Identity, ReLU
from pynn.nn.factories import activation_factory, initializer_factory
from pynn.nn.initializers import Constant, RandomNormal, Zeros

# --------------------------------------------------------------------------- #
# activation_factory
# --------------------------------------------------------------------------- #


def test_activation_defaults_to_identity():
    assert isinstance(activation_factory(None), Identity)
    assert isinstance(activation_factory(), Identity)


def test_activation_dict_form_passes_params_through():
    activation = activation_factory({"name": "relu", "params": {"alpha": 0.2}})

    assert isinstance(activation, ReLU)
    assert activation.alpha == 0.2


def test_activation_dict_form_without_params_uses_the_defaults():
    assert activation_factory({"name": "elu"}).alpha == 1.0


def test_activation_softmax_axis_is_forwarded():
    assert activation_factory({"name": "softmax", "params": {"axis": 0}}).axis == 0
    assert activation_factory("softmax").axis == -1


def test_an_activation_instance_is_returned_unchanged():
    activation = ELU(alpha=0.5)
    assert activation_factory(activation) is activation


def test_activation_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="not available"):
        activation_factory("swish")


def test_activation_rejects_an_uninterpretable_argument():
    with pytest.raises(ValueError, match="Cannot interpret"):
        activation_factory(3.14)  # type: ignore[arg-type]


def test_a_factory_built_activation_computes(rng):
    x = Tensor(rng.standard_normal((3, 4)))
    assert np.allclose(activation_factory("relu")(x).data, np.maximum(0.0, x.data))


# --------------------------------------------------------------------------- #
# initializer_factory
# --------------------------------------------------------------------------- #


def test_initializer_defaults_to_random_normal():
    assert isinstance(initializer_factory(None), RandomNormal)
    assert isinstance(initializer_factory(), RandomNormal)


def test_initializer_dict_form_passes_params_through():
    initializer = initializer_factory({"name": "constant", "params": {"value": 0.5}})

    assert isinstance(initializer, Constant)
    assert np.allclose(initializer((2, 3)).data, 0.5)


def test_initializer_dict_form_without_params_uses_the_defaults():
    values = initializer_factory({"name": "random_uniform"})((200,)).data
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_an_initializer_instance_is_returned_unchanged():
    initializer = Zeros()
    assert initializer_factory(initializer) is initializer


def test_initializer_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="not available"):
        initializer_factory("orthogonal")


def test_initializer_rejects_an_uninterpretable_argument():
    with pytest.raises(ValueError, match="Cannot interpret"):
        initializer_factory(7)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(4, 3), (5, 5), (2, 8)])
def test_every_initializer_produces_the_requested_shape(shape):
    for name in [
        "he_normal",
        "he_uniform",
        "lecun_normal",
        "lecun_uniform",
        "ones",
        "random_normal",
        "random_uniform",
        "xavier_normal",
        "xavier_uniform",
        "zeros",
    ]:
        assert initializer_factory(name)(shape).shape == shape

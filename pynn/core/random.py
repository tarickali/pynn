"""The random source shared by weight initialization and the stochastic layers.

Drawn from a module-level `numpy.random.Generator` rather than the legacy global
`numpy.random` functions, which cannot be seeded independently of anything else in the
process. Layers construct their initializers through the string/factory interface and
`Dropout` draws its mask during the forward pass, so neither can be handed a generator
directly; `set_seed` is the hook that makes a whole run reproducible.
"""

from __future__ import annotations

import numpy as np

__all__ = ["default_rng", "set_seed"]

_rng: np.random.Generator = np.random.default_rng()


def set_seed(seed: int | None = None) -> None:
    """Reseed the generator that initializers and stochastic layers draw from.

    Parameters
    ----------
    seed : int | None
        Seed value. `None` reseeds from the OS entropy source.

    Examples
    --------
    >>> from pynn.functional.initializers import he_normal
    >>> set_seed(0)
    >>> first = he_normal((4, 3))
    >>> set_seed(0)
    >>> bool((first.data == he_normal((4, 3)).data).all())
    True
    """

    global _rng
    _rng = np.random.default_rng(seed)


def default_rng(rng: int | np.random.Generator | None = None) -> np.random.Generator:
    """Resolve a caller-supplied generator, falling back to the shared one.

    Parameters
    ----------
    rng : int | np.random.Generator | None
        A generator to use as-is, a seed for a fresh one, or `None` for the shared
        generator that `set_seed` controls.
    """
    return _rng if rng is None else np.random.default_rng(rng)

from collections.abc import Iterator

import numpy as np

__all__ = ["get_batches", "one_hot"]

#: Anything `numpy.random.default_rng` accepts, so a caller can pass a seed for a
#: reproducible epoch or hand over a generator they already own.
SeedLike = int | np.random.Generator | None


def one_hot(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Create a one-hot array from integer labels.

    Parameters
    ----------
    x : np.ndarray
        Integer labels, shape (n,) or (n, 1).
    k : int, default 10
        Number of classes.

    Returns
    -------
    np.ndarray
        One-hot encoded array, shape (n, k).
    """
    x = np.asarray(x).ravel()
    n = x.shape[0]
    o = np.zeros((n, k), dtype=np.float64)
    o[np.arange(n), x] = 1
    return o


def get_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int | None = None,
    m: int | None = None,
    *,
    shuffle: bool = True,
    rng: SeedLike = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Iterate over (X, y) in batches, shuffled by default.

    Shuffling is what makes minibatch gradient descent *stochastic*. Iterating a fixed
    order replays the identical sequence of gradients every epoch, so consecutive steps
    stay correlated and the gradient noise that lets SGD escape shallow minima is
    absent.

    Batches are yielded rather than materialized, so an epoch never holds a second copy
    of the dataset. Wrap the call in `list(...)` if you need them all at once.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets, aligned with `X` along the first axis.
    batch_size : int, default 32
        Examples per batch. The last batch is smaller when the number of examples is
        not a multiple of it.
    m : int, optional
        Deprecated alias for `batch_size`.
    shuffle : bool, default True
        Permute the examples before batching.
    rng : int | np.random.Generator | None
        Seed or generator for the permutation. The default draws fresh entropy, so
        pass a seed when a run needs to be reproducible.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        One `(X_batch, y_batch)` pair per batch.

    Raises
    ------
    ValueError
        If `batch_size` is not positive, or `X` and `y` disagree on the number of
        examples. Both are checked eagerly, before the first batch is drawn.
    """

    bs = batch_size if batch_size is not None else (m if m is not None else 32)
    if bs <= 0:
        raise ValueError(f"batch_size must be positive, got {bs}")

    n = X.shape[0]
    if y.shape[0] != n:
        raise ValueError(
            f"X and y must have the same number of examples, got {n} and {y.shape[0]}"
        )

    order = np.random.default_rng(rng).permutation(n) if shuffle else None

    def batches() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, n, bs):
            stop = start + bs
            index = slice(start, stop) if order is None else order[start:stop]
            yield X[index], y[index]

    return batches()

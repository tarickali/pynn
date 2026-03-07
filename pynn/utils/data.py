import numpy as np

__all__ = ["one_hot", "get_batches"]


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
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split (X, y) into batches of size batch_size.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets.
    batch_size : int, default 32
        Batch size.
    m : int, optional
        Alias for batch_size (mininet/pynet compatibility).

    Returns
    -------
    list of (X_batch, y_batch) tuples
    """
    bs = batch_size if batch_size is not None else (m if m is not None else 32)
    n = X.shape[0]
    batches = []
    b = 0
    for i in range(n // bs):
        a, b = i * bs, (i + 1) * bs
        batches.append((X[a:b], y[a:b]))
    if b != n:
        batches.append((X[b:], y[b:]))
    return batches

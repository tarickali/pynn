import numpy as np

__all__ = ["make_pair", "pad_for_conv"]


def make_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Return (x, x) if x is int, else x. Used for kernel_size, stride, etc."""
    if isinstance(x, int):
        return (x, x)
    return x


def pad_for_conv(x: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Pad spatial dims (last two) of array (batch, ch, h, w)."""
    if pad_h == 0 and pad_w == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )

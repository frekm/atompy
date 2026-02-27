import numpy as np
from numpy.typing import NDArray, ArrayLike

from atompy.hist_utils import _shared

__all__ = [
    "rebin",
    "get_binsizes",
    "integrate",
]


def rebin(
    hist: NDArray[np.number], edges: NDArray[np.number], factor: int
) -> tuple[NDArray[np.number], NDArray[np.number]]:
    return _shared.rebin(hist, edges, factor)


def get_binsizes(edges: NDArray[np.number]) -> NDArray[np.number]:
    return np.diff(edges)


def integrate(hist: NDArray[np.number], edges: NDArray[np.number]) -> float:
    """
    Calculate the integral of a histogram

    Parameters
    ----------
    hist : ndarray
        The histogram values.

    edges : ndarray
        The histogram edges. Length must be one larger than *hist*.

    Returns
    -------
    float
        The integral, that is, ``hist * binsizes``.

    Examples
    --------

        >>> hist = np.array((1.0, 2.0, 1.0))
        >>> edges = np.array((0.0, 1.0, 1.5, 2.5))
        >>> np.sum(hist)
        4.0
        >>> ap.hist_utils.hist1d.integrate(hist, edges)
        3.0

    """
    return float(np.sum(hist * get_binsizes(edges)))

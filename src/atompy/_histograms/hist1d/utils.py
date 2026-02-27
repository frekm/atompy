import numpy as np
from numpy.typing import NDArray, ArrayLike

from atompy._histograms import _shared

__all__ = [
    "rebin",
]


def rebin(
    hist: NDArray[np.number], edges: NDArray[np.number], factor: int
) -> tuple[NDArray[np.number], NDArray[np.number]]:
    return _shared.rebin(hist, edges, factor)

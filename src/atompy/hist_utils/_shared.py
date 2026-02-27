import numpy as np
from numpy.typing import NDArray, ArrayLike

import atompy.utils as utils


def nbins(edges: NDArray[np.number]) -> int:
    return len(edges) - 1


def rebin(
    hist: NDArray[np.number], edges: NDArray[np.number], factor: int
) -> tuple[NDArray[np.number], NDArray[np.number]]:
    old_n = nbins(edges)
    if old_n % factor != 0:
        msg = f"Invalid {factor=}. Possible factors for this histogram are {utils.get_all_dividers(old_n)}."
        raise ValueError(msg)
    new_hist = np.empty(hist.size // factor)
    for i in range(new_hist.size):
        new_hist[i] = np.sum(hist[i * factor : i * factor + factor])
    new_edges = np.full(new_hist.size + 1, edges[-1])
    for i in range(new_edges.size - 1):
        new_edges[i] = edges[i * factor]
    return new_hist, new_edges

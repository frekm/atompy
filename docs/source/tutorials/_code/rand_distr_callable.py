import numpy as np
from typing import Literal
from collections.abc import Callable


def sample_distribution_func(
    f: Callable[[np.ndarray], np.ndarray],
    xlim: tuple[float, float],
    size: int,
    ylim: Literal["auto"] | tuple[float, float] = "auto",
) -> np.ndarray:
    """
    Sample a distribution described by `f`.

    Parameters
    ----------
    f : Callable
        f must take one argument of type ``ndarray`` and return type
        ``ndarray``.

    xlim : tuple[float, float]
    """
    output = np.empty(size)
    n_samples = 0

    if ylim == "auto":
        x = np.linspace(*xlim, 100)
        y = f(x)
        ylim = [np.amin(y), np.amax(y)]

    rng = np.random.default_rng()

    while n_samples < size:
        buffer = size - n_samples
        sample = rng.uniform(*xlim, buffer)
        test = rng.uniform(*ylim, buffer)

        sample = np.ma.compressed(np.ma.masked_array(sample, test > f(sample)))

        output[n_samples : n_samples + sample.size] = sample
        n_samples += sample.size

    return output

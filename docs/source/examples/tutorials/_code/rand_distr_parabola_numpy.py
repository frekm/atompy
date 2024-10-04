import numpy as np


def sample_parabola(
    xlim: tuple[float, float],
    size: int
) -> np.ndarray:
    output = np.empty(size)
    ylim = 0, xlim[1]**2
    n_samples = 0

    rng = np.random.default_rng()

    while n_samples < size:
        buffer = size - n_samples

        # create multiple samples at once
        sample = rng.uniform(*xlim, buffer)  # random x-value
        test = rng.uniform(*ylim, buffer)  # random y-value

        # test all samples at once
        sample = np.ma.compressed(np.ma.masked_array(sample, test > sample**2))

        # save "good" samples
        output[n_samples:n_samples + sample.size] = sample

        # repeat next turn with less samples
        n_samples += sample.size

    return output

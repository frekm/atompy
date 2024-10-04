import numpy as np


def sample_distribution(
    distribution_edges: np.ndarray,
    distribution_values: np.ndarray,
    size: int
) -> np.ndarray:
    output = np.empty(size)
    n_samples = 0

    rng = np.random.default_rng()

    while n_samples < size:
        buffer = size - n_samples
        sample = rng.uniform(distribution_edges[0], distribution_edges[-1],
                             buffer)
        test = rng.uniform(0.0, np.max(distribution_values), buffer)

        edges_index = np.digitize(sample, distribution_edges[1:-1])

        sample = np.ma.compressed(np.ma.masked_array(
            sample, test > distribution_values[edges_index]))

        output[n_samples:n_samples + sample.size] = sample
        n_samples += sample.size

    return output
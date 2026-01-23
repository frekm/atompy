import numpy as np


def sample_distribution_discrete(
    values: np.ndarray, probabilities: np.ndarray, size: int
) -> np.ndarray:
    probabilities_ = probabilities / np.sum(probabilities)
    rng = np.random.default_rng()
    return rng.choice(values, size, p=probabilities_)

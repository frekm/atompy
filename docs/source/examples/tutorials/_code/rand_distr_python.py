import random


def sample_parabola(
    xlim: tuple[float, float],
    size: int
) -> list[float]:
    output = []
    ylim = 0, xlim[1]**2

    while len(output) < size:
        sample = random.uniform(*xlim)  # random x-value
        test = random.uniform(*ylim)  # random y-value
        if test < sample**2:
            output.append(sample)

    return output

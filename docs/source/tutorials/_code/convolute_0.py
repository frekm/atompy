import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np
import scipy.ndimage as ndimage

import atompy as ap


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if (size + 1) % 2 != 0:
        raise ValueError("size must be odd")

    x = np.arange(-size // 2 + 1, size // 2 + 1)
    g = ap.gauss(x, "sum", 0.0, sigma)
    return g


rng = np.random.default_rng(42)
n = 100

edges = np.linspace(0, 10, n + 1)
original_data = np.zeros(n)
original_data[n // 2] = 100

original = ap.Hist1d(original_data, edges, title="original")

sigma = 3.0
kernel_x = np.arange(-6 * sigma - 1, 6 * sigma + 1)
kernel = ap.gauss(kernel_x, "sum", 0.0, sigma)
norm = ndimage.convolve1d(np.ones_like(original.values), kernel, mode="reflect")
convolved_data = ndimage.convolve1d(original.values, kernel, mode="reflect")
convolved = ap.Hist1d(convolved_data, original.edges, title="convolved")

fig, axs = plt.subplots(1, 3, layout=mplu.FixedLayoutEngine())

original.plot_step(ax=axs[0])
convolved.plot_step(ax=axs[2])

for ax in axs.flat:
    mplu.set_axes_size(3, 3, ax=ax)


plt.show()

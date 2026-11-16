import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

plt.style.use("atom")

# create some histogram
bin_centers = np.linspace(-2, 2, 45)
hist = ap.Hist1d.from_centers(ap.gauss(bin_centers), bin_centers)

# plot it
_, ax = plt.subplots(1, 1, layout=mplu.FixedLayoutEngine())
hist.plot_step(ax, lw=3.0, label="histogram")
hist.interpolated().plot(ax, lw=2.0, label="interpolated")
ax.legend()

mplu.set_axes_size(3, aspect=3 / 4)

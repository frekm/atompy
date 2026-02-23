import matplotlib.pyplot as plt
import atompy as ap
import numpy as np

plt.style.use("atom")

# create some histogram
bin_centers = np.linspace(-2, 2, 45)
hist = ap.Hist1d.from_centers(ap.gauss(bin_centers), bin_centers)

# plot it
ax = plt.subplot()
hist.plot_step(ax, lw=3.0, label="original")
hist.keep(-1, 1).plot_step(ax, lw=1.5, label="original")
ax.legend()

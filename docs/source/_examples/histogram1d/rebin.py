import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

plt.style.use("atom")

# create some histogram
bin_centers = np.linspace(-2, 2, 45)
hist = ap.Hist1d.from_centers(ap.gauss(bin_centers), bin_centers)

_, ax = hist.rebin(3).plot_step(label="rebinned")
hist.plot_step(ax=ax, label="original")

plt.legend()

import matplotlib.pyplot as plt
import atompy as ap
import numpy as np

plt.style.use("atom")

# create some histogram
bin_centers = np.linspace(-2, 2, 45)
hist = ap.Hist1d.from_centers(ap.gauss(bin_centers), bin_centers)

_, ax = plt.subplots()

hist.plot_step(ax=ax, lw=2.0, label="original")
hist.remove(-1, 1).plot_step(ax=ax, lw=1.0, label="with gate")

plt.legend()

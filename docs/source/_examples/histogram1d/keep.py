import matplotlib.pyplot as plt
import atompy as ap
import numpy as np

# create some histogram
bin_centers = np.linspace(-2, 2, 45)
hist = ap.Hist1d.from_centers(ap.gauss(bin_centers), bin_centers)

plt.step(*hist.for_step(extent_to=0), linewidth=2.0, label="original")
plt.step(*hist.keep(-1, 1).for_step(extent_to=0), linewidth=1.0, label="with gate")
plt.legend()

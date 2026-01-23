import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

# create some histogram
bin_centers = np.linspace(-2, 2, 45)
hist = ap.Hist1d.from_centers(ap.gauss(bin_centers), bin_centers)

plt.step(*hist.for_step(), label="original")
plt.step(*hist.rebin(3).for_step(), label="rebinned")
plt.legend()

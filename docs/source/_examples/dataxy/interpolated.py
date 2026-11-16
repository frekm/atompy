import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

plt.style.use("atom")

# create some histogram
x = np.linspace(-2, 2, 45)
data = ap.DataXY.from_function(ap.gauss, x)

# plot it
_, ax = plt.subplots(1, 1, layout=mplu.FixedLayoutEngine())
data.interpolated().plot(ax, lw=2.0, label="interpolated")
data.plot(ax, label="data", plot_fmt="o")
ax.legend()

mplu.set_axes_size(3, aspect=3 / 4)

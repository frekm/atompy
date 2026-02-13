import matplotlib.pyplot as plt
import atompy as ap
import numpy as np
import mplutils as mplu

plt.style.use("atom")

# create some histogram
bin_centers = np.linspace(-2, 2, 45)
hist = ap.Hist1d.from_centers(ap.gauss(bin_centers), bin_centers).pad_with(0)

_, ax = plt.subplots(layout=mplu.FixedLayoutEngine())
mplu.set_axes_size(3.0, 3.0 / 4.0)

hist.plot(ax=ax, drawstyle="steps-mid", label="original")
hist.remove(-1, 1).plot(ax=ax, drawstyle="steps-mid", label="with gate")

plt.legend()

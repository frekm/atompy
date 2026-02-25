import numpy as np
import matplotlib.pyplot as plt
import atompy as ap

plt.style.use("atom")

rng = np.random.default_rng(42)
lim = (-2, 2)
size = 1_000

hist = ap.Hist2d(
    *np.histogram2d(*rng.normal(size=(2, size)), range=(lim, lim)),
    title="A 2D Histogram",
    xlabel="X Label",
    ylabel="Y Label",
    zlabel="Intensity",
)

fig, ax, cb = hist.plot()

plt.show()

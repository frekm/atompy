import matplotlib.pyplot as plt
import numpy as np
import atompy as ap
import mplutils as mplu

rng = np.random.default_rng(42)
lim = (-2, 2)
size = 1_000

hist = ap.Hist2d(*np.histogram2d(*rng.normal(size=(2, size)), range=(lim, lim)))

fig, axs = plt.subplots(ncols=2)

colorbars = []
cmaps = "viridis", "cividis"

for ax, cmap in zip(axs, cmaps):
    mplu.set_axes_size_inches(2.0, ax=ax)
    hist.plot(
        ax=ax,
        xlabel="X Label",
        ylabel="Y Label",
        zlabel="Intensity",
        title=cmap,
        pcolormesh_kwargs=dict(cmap=cmap),
    )

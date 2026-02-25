import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

plt.style.use("atom")

rng = np.random.default_rng(42)
lim = (-2, 2)
size = 1_000

hist = ap.Hist2d(
    *np.histogram2d(*rng.normal(size=(2, size)), range=(lim, lim)),
    xlabel="X Label",
    ylabel="Y Label",
    zlabel="Intensity",
)

fig, axs = plt.subplots(ncols=2, layout="compressed", figsize=(6.0, 3.0))

cmaps = "viridis", "cividis"

for ax, cmap in zip(axs, cmaps):
    ax.set_box_aspect(1)
    hist.plot(ax=ax, title=cmap, cmap=cmap)

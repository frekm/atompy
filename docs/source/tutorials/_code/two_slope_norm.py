import atompy as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplutils as mplu

# some data to plot
fr = 0.5
to = 1.5
size = 1000
data = np.linspace(-fr, to, size).reshape(size, 1)
hist = ap.Hist2d(data, np.arange(size + 1), np.arange(2))

plt.style.use("atom")
plt.rcParams["axes.grid"] = False
plt.rcParams["image.cmap"] = "RdBu"
_, axs = plt.subplots(1, 3, layout=mplu.FixedLayoutEngine())

for ax in axs:
    mplu.set_axes_size(2.5, ax=ax)
    zero_crossing = fr * size / (fr + to)
    ax.axvline(zero_crossing)
    ax.text(zero_crossing + 10, 0.95, "zero crossing", va="top")

norm1 = mcolors.TwoSlopeNorm(vmin=-fr, vcenter=0, vmax=to)
norm2 = mcolors.TwoSlopeNorm(vmin=-max(fr, to), vcenter=0, vmax=max(fr, to))

titles = (
    "Misaligned zero",
    "Different dynamic range ",
    "Equal dynamic range",
)

for i, norm in enumerate((None, norm1, norm2)):
    im = axs[i].pcolormesh(*hist.for_pcolormesh(), norm=norm, rasterized=True)

    axs[i].set_title(titles[i], pad=40)

    cb = mplu.add_colorbar(im, axs[i], location="top")
    cb.ax.set_xlim(-fr, to)
    cb.ax.axvline(0)

plt.show()

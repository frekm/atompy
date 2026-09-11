import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

# some example data
x = np.array((0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0))
y = np.array((0.0, 0.1, 0.3, 0.6, 1.0))
z = np.arange(0, x.size * y.size, 1).reshape(x.size, y.size)
d = ap.DataXYZ(x, y, z)

plt.style.use("atom")
plt.rcParams["axes.grid"] = False

shadings = ("nearest", "flat", "gouraud")

fig, axs = plt.subplots(1, 3, layout=mplu.FixedLayoutEngine())

for ax, shading in zip(axs, shadings):
    mplu.set_axes_size(2.5, 2.5, ax=ax)
    im = ax.pcolormesh(*d.for_pcolormesh(shading), shading=shading, rasterized=True)
    mplu.add_colorbar(im, ax=ax, label="z")
    ax.set_title(f"{shading=}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

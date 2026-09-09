import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np
from matplotlib import patheffects

import atompy as ap

plt.style.use("atom")
plt.rcParams["axes.grid"] = False

x = np.array((1, 1, 1, 2, 2, 2))
y = np.array((1, 2, 3, 1, 2, 3))
z = np.array((11, 12, 13, 21, 22, 23))

x_, y_, z_ = ap.columns_to_meshgrid(x, y, z)

_, ax = plt.subplots(layout=mplu.FixedLayoutEngine())

im = ax.pcolormesh(x_, y_, z_.T, shading="auto")
cb = mplu.add_colorbar(im, ax, label="z")

ax.set_xlabel("x")
ax.set_ylabel("y")

# show values of pixels in plot
for i, xi in enumerate(x_):
    for j, yi in enumerate(y_):
        text = ax.text(xi, yi, f"{z_[i, j]}", va="center", ha="center", c="w")
        text.set_path_effects(
            [
                patheffects.withStroke(linewidth=1.5, foreground="k"),
                patheffects.Normal(),
            ]
        )

mplu.set_axes_size(3)

import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

x = y = np.linspace(-5, 5, 50)
z = x[:, None] ** 2 + y[None, :] ** 3
d = ap.DataXYZ(x, y, z, xlabel="x", ylabel="y")

plt.style.use("atom")
plt.rcParams["axes.grid"] = False
_, axs = plt.subplots(1, 3, layout=mplu.FixedLayoutEngine())

d.plot(ax=axs[0], title="original")
d.get_closest_x(0).plot(ax=axs[1], plot_fmt="o", c="r", title="get_closest_x(0)")
axs[0].axvline(0, c="r")
d.get_closest_y(0).plot(ax=axs[2], plot_fmt="o", c="b", title="get_closest_y(0)")
axs[0].axhline(0, c="b")

for ax in axs.flat:
    mplu.set_axes_size(2.5, 2.5, ax=ax)

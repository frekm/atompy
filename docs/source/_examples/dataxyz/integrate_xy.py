import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

xn, yn = 10, 5
x = np.linspace(-1, 1, xn)
y = np.linspace(-1, 1, yn)
z = x[:, None] - y[None, :]
data2d = ap.DataXYZ(x, y, z, xlabel="x", ylabel="y")

plt.style.use("atom")
plt.rcParams["image.cmap"] = "bwr"
_, axs = plt.subplots(1, 3, layout=mplu.FixedLayoutEngine())

axs[0].grid(False)
data2d.plot(ax=axs[0])

data2d.integrate_y().plot(ax=axs[1], plot_fmt="o", title="integrate_y")
data2d.integrate_x().plot(ax=axs[2], plot_fmt="o", title="integrate_x")

for ax in axs.flat:
    mplu.set_axes_size(2.5, 2.5, ax=ax)

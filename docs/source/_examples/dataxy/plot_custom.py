import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np
from matplotlib.axes import Axes

import atompy as ap


def my_plot_method(data: ap.DataXY, ax: Axes, fac: float, **kwargs):
    ax.set_title(f"{data.title}, scaled by {fac}")
    ax.plot(data.x, data.y * fac, **kwargs)


data = ap.DataXY.from_function(
    ap.gauss, np.linspace(-5, 5, 100), title="Normal distribution", xlabel="x"
)

plt.style.use("atom")
fig, axs = plt.subplots(1, 2, layout=mplu.FixedLayoutEngine())

data.plot(ax=axs[0])
data.set_plot_method(my_plot_method)
data.plot(ax=axs[1], fac=10, lw=5)

for ax in axs.flat:
    mplu.set_axes_size(3, ax=ax)

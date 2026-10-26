import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

d = ap.DataXY((1, 2, 3, 4), (10, 20, 30, 40))

plt.style.use("atom")
fig, axs = plt.subplots(1, 2, layout=mplu.FixedLayoutEngine())

d.plot_errorbar(axs[0], title=r"$1\sigma$")
d.plot_errorbar(axs[1], errorfunc=lambda x: 3 * np.sqrt(x), title=r"$3\sigma$")

for ax in axs.flat:
    mplu.set_axes_size(3, ax=ax)
    ax.set_ylim(axs[1].get_ylim())

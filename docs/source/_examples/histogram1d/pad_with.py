import atompy as ap
import numpy as np
import matplotlib.pyplot as plt
import mplutils as mplu

plt.style.use("atom")

hist = ap.Hist1d(np.arange(4) + 1, np.arange(5))

_, axs = plt.subplots(1, 2, layout=mplu.FixedLayoutEngine())

axs[0].plot(*hist.for_plot(), drawstyle="steps-mid")
axs[1].plot(*hist.pad_with(0).for_plot(), drawstyle="steps-mid")

axs[0].set_ylim(axs[1].get_ylim())
axs[0].set_xlim(axs[1].get_xlim())

for ax in axs:
    mplu.set_axes_size(3, ax=ax)

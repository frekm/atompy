import matplotlib.pyplot as plt
import numpy as np
import atompy as ap
import mplutils as mplu

rng = np.random.default_rng(42)
lim = (-2, 2)
size = 1_000

hist = ap.Hist2d(*np.histogram2d(*rng.normal(size=(2, size)), range=(lim, lim)))

_, axs = plt.subplots(1, 2)
for ax in axs:
    mplu.set_axes_size_inches(2.0, ax=ax)

hist.plot(ax=axs[0], title="Original histogram")
hist.norm_row_to_sum().plot(ax=axs[1], title="Row-normalized histogram")

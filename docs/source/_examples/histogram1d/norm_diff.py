import numpy as np
import atompy as ap
import mplutils as mplu
import matplotlib.pyplot as plt

plt.style.use("atom")

rng = np.random.default_rng(42)
data0 = rng.normal(-1, size=(2000))
data1 = rng.normal(1, size=(3000))

hist0 = ap.Hist1d(*np.histogram(data0, range=(-2, 2)))
hist1 = ap.Hist1d(*np.histogram(data1, range=(-2, 2)))

_, axs = plt.subplots(1, 3)
for ax in axs:
    mplu.set_axes_size_inches(2, ax=ax)

kwargs = dict(drawstyle="steps-mid")
hist0.pad_with(0).plot(axs[0], title="Histogram 1", **kwargs)
hist1.pad_with(0).plot(axs[1], title="Histogram 2", **kwargs)
hist0.norm_diff(hist1).pad_with(0).plot(axs[2], title="Normalized Difference", **kwargs)


plt.show()

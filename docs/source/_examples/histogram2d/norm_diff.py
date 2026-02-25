import numpy as np
import atompy as ap
import matplotlib.pyplot as plt

plt.style.use("atom")

rng = np.random.default_rng(42)
data0 = rng.normal((0, -1), size=(2000, 2))
data1 = rng.normal((0, 1), size=(3000, 2))

hist0 = ap.Hist2d(*np.histogram2d(*data0.T, range=((-2, 2), (-2, 2))))
hist1 = ap.Hist2d(*np.histogram2d(*data1.T, range=((-2, 2), (-2, 2))))

_, axs = plt.subplots(1, 3, layout="compressed", figsize=(8.0, 2.5))
for ax in axs:
    ax.set_box_aspect(1)

hist0.plot(axs[0], title="Histogram 1")
hist1.plot(axs[1], title="Histogram 2")
hist0.norm_diff(hist1).plot(axs[2], cmap="coolwarm", title="Normalized Difference")

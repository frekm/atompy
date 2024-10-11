import atompy as ap
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["image.cmap"] = "atom"

rng = np.random.default_rng(42)
x, y = rng.normal(size=(2, 10_000))

hist = ap.Hist2d(*np.histogram2d(x, y, 20, range=[[-2, 2],[-2, 2]]))

_, (ax0, ax1) = plt.subplots(1, 2)

ax0.imshow(**hist.for_imshow())
ax1.imshow(**hist.without_yrange((-1, 1)).for_imshow())

ap.make_me_nice()
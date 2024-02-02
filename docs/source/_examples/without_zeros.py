import numpy as np
import atompy as ap
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "lmf2root"
plt.rcParams["image.aspect"] = "auto"

gen = np.random.default_rng(42)
hist = ap.Hist2d(*np.histogram2d(*gen.normal(size=(2, 100))))

fig, ax = ap.subplots(ncols=2, ratio=1)

ax[0][0].imshow(**hist.for_imshow())                # left panel
ax[0][1].imshow(**hist.without_zeros.for_imshow())  # right panel

ap.make_margins_tight(ax, pad=5)

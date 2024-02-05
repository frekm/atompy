import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

# config and create a figure
plt.rcParams["image.cmap"] = "lmf2root"
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["image.aspect"] = "auto"
fig, ax = ap.subplots(nrows=2, ratio=1, ypad=3)

# example data
gen = np.random.default_rng(42)
hist2d = ap.Hist2d(*np.histogram2d(*gen.normal(size=(2, 1_000))))
hist1d = hist2d.projected_onto_x

# plotting
ax[0][0].step(*hist1d.for_step)
ax[1][0].imshow(**hist2d.for_imshow())

# formatting
ap.change_ratio(2, ax[0][0], anchor="lower")
ax[0][0].set_xticklabels([])
ax[0][0].set_xlim(ax[1][0].get_xlim())
ap.make_margins_tight(ax, pad=5)

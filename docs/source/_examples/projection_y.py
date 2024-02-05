import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

# config and create a figure
plt.rcParams["image.cmap"] = "lmf2root"
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["image.aspect"] = "auto"
fig, ax = ap.subplots(ncols=2, ratio=1, xpad=3)

# example data
gen = np.random.default_rng(42)
hist2d = ap.Hist2d(*np.histogram2d(*gen.normal(size=(2, 1_000))))
hist1d = hist2d.projected_onto_y

# plotting
ax[0][0].imshow(**hist2d.for_imshow())
ax[0][1].step(hist1d.for_step[1][::-1], hist1d.for_step[0][::-1])

# formatting
ap.change_ratio(0.5, ax[0][1], anchor="left", adjust="width")
ax[0][1].set_yticklabels([])
ax[0][1].set_ylim(ax[0][0].get_ylim())
ap.make_margins_tight(ax, pad=5)

import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

# config and create a figure
plt.rcParams["image.cmap"] = "atom"
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["image.aspect"] = "auto"

_, axs = plt.subplots(2, 1, sharex=True)

# formatting of axes sizes
w, h = 3.0, 3.0
ap.set_axes_size(w, h, axs[1])
ap.set_axes_size(w, h*0.3, axs[0])

# example data
gen = np.random.default_rng(42)
hist2d = ap.Hist2d(*np.histogram2d(*gen.normal(size=(2, 1_000))))
hist1d = hist2d.projected_onto_x

# plotting
axs[0].step(*hist1d.for_step)
im = axs[1].imshow(**hist2d.for_imshow())

ap.add_colorbar(im, axs[1])

ap.make_me_nice(row_pad_pts=1, fix_figwidth=False)

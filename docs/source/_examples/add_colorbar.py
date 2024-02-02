import atompy as ap
import numpy as np
import matplotlib.pyplot as plt

# configure standard image parameters
plt.rcParams["image.cmap"] = "lmf2root"  # works only if atompy is imported
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["image.aspect"] = "auto"

# create a figure and axes
fig, ax = ap.subplots(ratio=1)

# create some sample data
data = np.arange(9).reshape((3, 3))
extent_of_data = (-5, 5, -1, 1)

# plot
image = ax[0][0].imshow(data, extent=extent_of_data)

# add colorbar and save a Colorbar instance
colorbar = ap.add_colorbar(ax[0][0], image, where="top", label="Intensity")

# modify colorbar after its creation
tick_max = colorbar.ax.get_xlim()[1]
colorbar.ax.set_xticks(
    ticks=[0, 0.5*tick_max, 1.0*tick_max],
    labels=["0%", "50%", "100%"])

# remove unnecessary margins, the colorbar has to be passed for it to work
ap.make_margins_tight(ax, colorbars=colorbar, pad=5)

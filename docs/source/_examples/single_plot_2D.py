"""
Plot a single 2D-Histogram
"""
import atompy as ap
import matplotlib.pyplot as plt

# load 2D histogram from root file to plot it with imshow
image, extents = ap.import_root_for_imshow(
    "example.root", "He_Compton/electrons/momenta/px_vs_py")

# format figure
plt.rcParams["image.cmap"] = "lmf2root"
plt.rcParams["image.aspect"] = "auto"
plt.rcParams["image.interpolation"] = "none"

# create a Figure with a single Axes
fig, ax = ap.subplots(ratio=1)

# plot 
cmap_image = ax[0][0].imshow(image, extent=extents)

# add a colorbar
colorbar = ap.add_colorbar(ax, cmap_image, label="Yield (counts)")

# format plot
ax[0][0].set_xlabel(r"$p_x$ (a.u.)")
ax[0][0].set_ylabel(r"$p_y$ (a.u.)")

# adjust margins such that everything fits
ap.make_margins_tight(ax, colorbars=colorbar, pad=5.0)
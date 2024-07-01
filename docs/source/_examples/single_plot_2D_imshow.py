"""
Plot a single 2D-Histogram from a ROOT file.
"""
import atompy as ap
import matplotlib.pyplot as plt

# configure using rcParams
plt.rcParams["figure.figsize"] = 3., 3. # the height will be adjusted by make_me_nice()

# load 2D histogram from root file to plot it with imshow
image, extents = ap.load_2d_from_root(
    "example.root", "He_Compton/electrons/momenta/px_vs_py",
    output_format="imshow")

# format figure
plt.rcParams["image.cmap"] = "lmf2root"
plt.rcParams["image.aspect"] = "auto"
plt.rcParams["image.interpolation"] = "none"

# create a Figure with a single Axes
ax = plt.subplot()
ax.set_box_aspect(1.0)

# plot 
cmap_image = ax.imshow(image, extent=extents)

# add a colorbar
cb = ap.add_colorbar(cmap_image, ax)
cb.set_label("Yield (counts)", rotation=270, va="baseline")

# format plot
ax.set_xlabel(r"$p_x$ (a.u.)")
ax.set_ylabel(r"$p_y$ (a.u.)")

ap.make_me_nice()

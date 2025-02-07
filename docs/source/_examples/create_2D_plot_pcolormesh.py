"""
Plot a single 2D-Histogram from a ROOT file.
"""
import atompy as ap
import matplotlib.pyplot as plt

# load 2D histogram from root file to plot it with imshow
x, y, z = ap.load_2d_from_root(
    "example.root", "He_Compton/electrons/momenta/px_vs_py",
    output_format="pcolormesh")

# format figure
plt.rcParams["image.cmap"] = "atom"
plt.rcParams["image.aspect"] = "auto"
plt.rcParams["image.interpolation"] = "none"

fig, ax, cb = ap.create_2d_plot(x, y, z,
                                xlabel=r"$p_x$ (a.u.)",
                                ylabel=r"$p_y$ (a.u.)",
                                colorbar_label="Yield (counts)")

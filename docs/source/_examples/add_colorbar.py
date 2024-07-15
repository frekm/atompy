import atompy as ap
import numpy as np
import matplotlib.pyplot as plt

# configure standard image parameters
plt.rcParams["image.cmap"] = "atom"  # works only if atompy is imported
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["image.aspect"] = "auto"

_, ax = plt.subplots(figsize=(3.2, 3.2))
ax.set_box_aspect(1.0)

image = ax.imshow(np.arange(9).reshape((3, 3)))

colorbar = ap.add_colorbar(image, ax, location="top")

# modify colorbar after its creation
tick_max = colorbar.ax.get_xlim()[1]
colorbar.set_label("Intensity")
locs = [0, 0.5, 1.0]
colorbar.ax.set_xticks(
    ticks=[l*tick_max for l in locs],
    labels=[f"{l*100:.0f}%" for l in locs])

ap.make_me_nice()

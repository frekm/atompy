import matplotlib.pyplot as plt
import atompy as ap

plt.rcParams["figure.facecolor"] = "grey"

fig, axs = plt.subplots(2, 1, figsize=(3.2, 3.2))

axs[0].set_box_aspect(1.0)
axs[1].set_box_aspect(1./1.618)

axs[0].set_ylim(0.0, 0.1)

ap.align_axes_horizontally(axs[0], axs[1], alignment="right")

ap.make_me_nice(margin_pad_pts=(8.0, 5.0, 5.0, 5.0))  # left, right, top, bottom

ap.align_axes_horizontally(axs[0], axs[1], alignment="right")
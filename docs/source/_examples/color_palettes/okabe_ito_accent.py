import atompy as ap
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

palette = ap.PALETTE_OKABE_ITO_ACCENT

ax = plt.subplot()
ax.axis("off")
ax.set_aspect("equal")

for i, color in enumerate(palette):
    ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=color))
    ax.text(i+0.5, 0.5, f"{to_hex(color)}", ha="center", va="center", c="w")

ax.set_xlim(0, len(palette))

ap.set_axes_size(len(palette), 1)
ap.make_me_nice(fix_figwidth=False)

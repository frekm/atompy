"""
Plot a single 1D-Histogram from a ROOT file.
"""

import atompy as ap
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_box_aspect(3/4)

x, y = ap.load_1d_from_root(
    "example.root", "He_Compton/electrons/momenta/p_mag")
ax.step(x, y, where="mid")

ax.set_xlabel(r"$p_{mag}$ (a.u.)")
ax.set_ylabel("Yield (counts)")

ap.make_me_nice()

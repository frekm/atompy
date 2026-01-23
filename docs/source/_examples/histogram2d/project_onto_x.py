import matplotlib.pyplot as plt
import numpy as np
import atompy as ap
import mplutils as mplu

rng = np.random.default_rng(42)
lim = (-2, 2)
size = 1_000

hist = ap.Hist2d(
    *np.histogram2d(
        rng.normal(-0.5, size=size),
        rng.normal(0.5, size=size),
        range=(lim, lim),
    )
)

_, axs = plt.subplots(2, 2, squeeze=False)
axs = axs.flat

im = axs[0].pcolormesh(*hist.for_pcolormesh())
axs[0].set_title("2D Histogram")
mplu.add_colorbar(im, axs[0]).set_label("counts")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[1].bar(*hist.project_onto_x().for_bar())
axs[1].set_title("project all")

axs[2].bar(*hist.keep(xlower=-1, xupper=1).project_onto_x().for_bar())
axs[2].set_title("project within x gate")

axs[3].bar(*hist.remove(xlower=-1, xupper=1).project_onto_x().for_bar())
axs[3].set_title("project outside x gate")

for ax in axs[1:]:
    ax.set_xlabel("x")
    ax.set_ylabel("counts")
    ax.set_ylim(axs[1].get_ylim())

for ax in axs:
    ax.set_box_aspect(1.0)
    ax.set_xlim(*lim)

mplu.make_me_nice(min_runs=3)

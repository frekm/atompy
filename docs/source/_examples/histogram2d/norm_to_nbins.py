import matplotlib.pyplot as plt
import atompy as ap
import numpy as np
import mplutils as mplu
import matplotlib.patheffects as patheffects

xedges = np.array([0, 1, 2, 3])
yedges = np.array([0, 1, 3, 6, 10])
values = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
)
hist = ap.Hist2d(values, xedges, yedges)
hists = (hist, hist.norm_to_xbins(), hist.norm_to_ybins())

_, axs = plt.subplots(1, len(hists), layout=mplu.FixedLayoutEngine())

titles = (
    "original",
    f"normed to xbins (={hist.xbins})",
    f"normed to ybins (={hist.ybins})",
)

for i, ax in enumerate(axs):
    ax.set_title(titles[i], pad=42)
    im = ax.pcolormesh(*hists[i].for_pcolormesh(), rasterized=True, cmap="atom")
    mplu.add_colorbar(im, ax, location="top", thickness_pts=8.0)
    mplu.set_axes_size(2.5, ax=ax)

    # show values of bin in plot
    for j, x in enumerate(hists[i].xcenters):
        for k, y in enumerate(hists[i].ycenters):
            text = ax.text(
                x, y, f"{hists[i].values[j, k]:.2f}", va="center", ha="center", c="w"
            )
            text.set_path_effects(
                [
                    patheffects.withStroke(linewidth=1.5, foreground="k"),
                    patheffects.Normal(),
                ]
            )

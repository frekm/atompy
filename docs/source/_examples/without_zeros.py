import numpy as np
import atompy as ap
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "lmf2root"
plt.rcParams["image.aspect"] = "auto"

gen = np.random.default_rng(42)
hist = ap.Hist2d(*np.histogram2d(*gen.normal(size=(2, 100))))

fig, ax_ = ap.subplots(ncols=3, ratio=1)
ax = ap.abcify_axes(ax_)

im = [] # empty list to store colorbar images in
im.append(ax["a"].imshow(**hist.for_imshow()))
im.append(ax["b"].imshow(**hist.without_zeros.for_imshow(), vmin=0))
im.append(ax["c"].imshow(**hist.without_zeros.for_imshow()))

colorbars = ap.add_colorbar(ax_, im, where="top")
ap.make_margins_tight(ax_, pad=5, colorbars=colorbars)
ap.add_abc(ax_, prefix="(", suffix=")")



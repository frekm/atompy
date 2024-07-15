import numpy as np
import atompy as ap
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "atom"
plt.rcParams["image.aspect"] = "auto"

gen = np.random.default_rng(42)
hist = ap.Hist2d(*np.histogram2d(*gen.normal(size=(2, 100))))

_, axs = plt.subplot_mosaic([["a", "b", "c"]])

imga = axs["a"].imshow(**hist.for_imshow())
imgb = axs["b"].imshow(**hist.without_zeros.for_imshow(), vmin=0)
imgc = axs["c"].imshow(**hist.without_zeros.for_imshow())

for img, ax in zip([imga, imgb, imgc], axs.values()):
    ax.set_box_aspect(1.0)
    ap.add_colorbar(img, ax, location="top")

ap.add_abc()
ap.make_me_nice()

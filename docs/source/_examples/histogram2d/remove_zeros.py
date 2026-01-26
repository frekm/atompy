import numpy as np
import atompy as ap
import matplotlib.pyplot as plt
import mplutils as mplu

plt.style.use("atom")

gen = np.random.default_rng(42)
hist = ap.Hist2d(*np.histogram2d(*gen.normal(size=(2, 100))))

_, axs = plt.subplot_mosaic([["a", "b", "c"]])

imga = axs["a"].pcolormesh(*hist.for_pcolormesh())
imgb = axs["b"].pcolormesh(*hist.remove_zeros().for_pcolormesh(), vmin=0)
imgc = axs["c"].pcolormesh(*hist.remove_zeros().for_pcolormesh())

for img, ax in zip([imga, imgb, imgc], axs.values()):
    mplu.set_axes_size_inches(2, ax=ax)
    mplu.add_colorbar(img, ax, location="top")

mplu.make_me_nice()

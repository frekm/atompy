import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

rng = np.random.default_rng(42)
x = rng.normal(size=10_000)

hist = ap.Hist1d(*np.histogram(x, bins=20, range=(-2, 2)))

_, (ax0, ax1) = plt.subplots(1, 2)

for ax in (ax0, ax1):
    ax.set_box_aspect(1./1.618)

ax0.step(*hist.for_step)
ax1.step(*hist.within_range((-1, 1), keepdims=True).for_step)

ap.make_me_nice()

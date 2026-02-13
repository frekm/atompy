import numpy as np
import atompy as ap
import matplotlib.pyplot as plt
import mplutils as mplu

plt.style.use("atom")

# get some random data
gen = np.random.default_rng(42)
x_sample = gen.normal(size=10_000)
y_sample = gen.uniform(-1, 1, size=10_000)

# create a histogram
hist = ap.Hist2d(
    *np.histogram2d(x_sample, y_sample, bins=(100, 10), range=((-5, 5), (-1, 1)))
)
mean, mean_errors = hist.profile_along_y()
_, standard_deviation = hist.profile_along_y("s")

_, axs = plt.subplots(1, 2, layout=mplu.FixedLayoutEngine())
for ax in axs.flat:
    ax.pcolormesh(*hist.for_pcolormesh())
    mplu.set_axes_size(3, ax=ax)

kwargs = dict(fmt="o-", color="k")

axs[0].errorbar(mean, hist.ycenters, xerr=mean_errors, **kwargs)
axs[0].set_title("Errorbars = Mean errors")

axs[1].errorbar(mean, hist.ycenters, xerr=standard_deviation, **kwargs)
axs[1].set_title("Errorbars = Standard deviation")

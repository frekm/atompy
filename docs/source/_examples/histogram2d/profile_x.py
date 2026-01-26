import numpy as np
import atompy as ap
import matplotlib.pyplot as plt
import mplutils as mplu

plt.style.use("atom")

# get some random data
gen = np.random.default_rng(42)
x_sample = gen.uniform(-1, 1, size=10_000)
y_sample = gen.normal(size=10_000)

# create a histogram
hist = ap.Hist2d(
    *np.histogram2d(x_sample, y_sample, bins=(10, 100), range=((-1, 1), (-5, 5)))
)
mean, mean_errors = hist.profile_along_x()
_, standard_deviation = hist.profile_along_x("s")

_, axs = plt.subplots(1, 2)

for ax in axs.flat:
    ax.pcolormesh(*hist.for_pcolormesh())
    ax.set_box_aspect(1.0)

kwargs = dict(fmt="o-", color="k")

axs[0].errorbar(hist.xcenters, mean, yerr=mean_errors, **kwargs)
axs[0].set_title("Errorbars = Mean errors")

axs[1].errorbar(hist.xcenters, mean, yerr=standard_deviation, **kwargs)
axs[1].set_title("Errorbars = Standard deviation")

mplu.make_me_nice()

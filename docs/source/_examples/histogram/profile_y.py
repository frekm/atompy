import numpy as np
import atompy as ap
import matplotlib.pyplot as plt

# get some random data
gen = np.random.default_rng(42)
x_sample = gen.normal(size=100_000)
y_sample = gen.uniform(-1, 1, size=100_000)

# create a histogram
hist = ap.Hist2d(*np.histogram2d(x_sample, y_sample,
                 bins=(100, 10), range=((-5, 5), (-1, 1))))
mean, mean_errors = hist.get_profile_along_y()
_, standard_deviation = hist.get_profile_along_y("s")

# configure plots
plt.rcParams["image.cmap"] = "lmf2root"  # works only if atompy is imported
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["image.aspect"] = "auto"

_, axs = plt.subplots(1, 2)
for ax in axs.flat:
    ax.imshow(**hist.for_imshow())
    ax.set_box_aspect(1.0)

axs[0].errorbar(mean, hist.ycenters, xerr=mean_errors)
axs[0].set_title("Errorbars = Mean errors")

axs[1].errorbar(mean, hist.ycenters, xerr=standard_deviation)
axs[1].set_title("Errorbars = Standard deviation")

ap.make_me_nice()

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

# plot
fig, ax = ap.subplots(ncols=2, ratio=1)
for a in ap.flatten(ax):
    a.imshow(**hist.for_imshow())
ax[0][0].errorbar(mean, hist.ycenters, xerr=mean_errors)         # left panel
ax[0][1].errorbar(mean, hist.ycenters, xerr=standard_deviation)  # right panel
ap.make_margins_tight(ax, pad=5)

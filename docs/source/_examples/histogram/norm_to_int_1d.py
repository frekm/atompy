import numpy as np
import atompy as ap
import matplotlib.pyplot as plt

# generate a histogram
gen = np.random.default_rng(42)
hist = ap.Hist1d(*np.histogram(gen.normal(size=1000)))

# plot
ax_raw = plt.subplot(121)
ax_raw.set_box_aspect(1/1.618)

ax_norm = plt.subplot(122)
ax_norm.set_box_aspect(1/1.618)

ax_raw.step(*hist.for_step)
ax_raw.set_title("raw histogram")

ax_norm.step(*hist.normalized_to_integral.for_step)
ax_norm.set_title("histogram normalized to total counts")

ap.make_me_nice()

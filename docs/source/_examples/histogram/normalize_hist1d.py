import numpy as np
import atompy as ap
import matplotlib.pyplot as plt

# generate a histogram
gen = np.random.default_rng(42)
hist = ap.Hist1d(*np.histogram(gen.normal(size=1000)))

_, axs = plt.subplots(1, 4)

axs[0].step(*hist.for_step)
axs[0].set_title("Raw")
axs[1].step(*hist.normalized_to_max.for_step)
axs[1].set_title("Norm. to maximum")
axs[2].step(*hist.normalized_to_sum.for_step)
axs[2].set_title("Norm. to sum")
axs[3].step(*hist.normalized_to_integral.for_step)
axs[3].set_title("Norm. to integral")

for ax in axs:
    ap.set_axes_size(2.0, 2.0, ax)

ap.make_me_nice(fix_figwidth=False)

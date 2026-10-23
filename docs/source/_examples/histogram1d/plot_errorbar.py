import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

h = ap.Hist1d.from_centers((10, 20, 30, 40), (0, 1, 2, 3))

plt.style.use("atom")
fig, axs = plt.subplots(1, 2, layout=mplu.FixedLayoutEngine())

kwargs = dict(fmt="o", capsize=3.0, ms=5.0, capthick=2.0, elinewidth=2.0)

h.plot_errorbar(axs[0], title="$1\sigma$", **kwargs)
h.plot_errorbar(
    axs[1], errorfunc=lambda x: 3 * np.sqrt(x), title=r"$3\sigma$", **kwargs
)

for ax in axs.flat:
    mplu.set_axes_size(3, ax=ax)

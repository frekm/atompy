import matplotlib.pyplot as plt
import mplutils as mplu
import numpy as np

import atompy as ap

x = y = np.linspace(0, 5, 6)
z = np.arange(0, x.size * y.size, 1).reshape(x.size, y.size)
d = ap.DataXYZ(x, y, z)

plt.style.use("atom")
plt.rcParams["axes.grid"] = False
_, axs = plt.subplots(1, 3, layout=mplu.FixedLayoutEngine())

d.plot(ax=axs[0], title="original")
d.remove_x(1, 4).plot(ax=axs[1], title="setval=0.0")
d.remove_x(1, 4, setval=np.nan).plot(ax=axs[2], title="setval=NaN")

for ax in axs.flat:
    mplu.set_axes_size(2.5, 2.5, ax=ax)

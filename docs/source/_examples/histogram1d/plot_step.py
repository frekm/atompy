import atompy as ap
import matplotlib.pyplot as plt
import mplutils as mplu

plt.style.use("atom")

fig, axs = plt.subplots(1, 2)
for ax in axs:
    mplu.set_axes_size_inches(3.0, 3.0 / 4.0, ax=ax)

hist = ap.Hist1d((3, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5))
hist.plot_step(ax=axs[0])
hist.plot_step(ax=axs[1], start_at="auto")

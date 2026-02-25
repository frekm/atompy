import atompy as ap
import matplotlib.pyplot as plt

plt.style.use("atom")

fig, axs = plt.subplots(1, 2, layout="compressed", figsize=(6.4, 3.0))

hist = ap.Hist1d((3, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5))
axs[0].set_title("start_at = 0")
hist.plot_step(ax=axs[0])
axs[1].set_title('start_at = "auto"')
hist.plot_step(ax=axs[1], start_at="auto")

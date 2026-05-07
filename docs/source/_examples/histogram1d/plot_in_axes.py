import matplotlib.pyplot as plt

import atompy as ap

plt.style.use("atom")

hist = ap.Hist1d((1, 2, 3, 4), (0, 1, 2, 3, 4))

_, axs = plt.subplots(1, 2, layout="compressed")

hist.plot(axs[0], plot_fmt="o")
hist.pad_with(0).plot(axs[1], drawstyle="steps-mid")
plt.show()

import matplotlib.pyplot as plt

import atompy as ap

plt.style.use("atom")

hist = ap.Hist1d((1, 2, 3, 4), (0, 1, 2, 3, 4))
hist.plot(drawstyle="steps-mid")

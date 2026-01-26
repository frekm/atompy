import atompy as ap
import matplotlib.pyplot as plt

plt.style.use("atom")

hist = ap.Hist1d((1, 2, 3, 4), (0, 1, 2, 3, 4))

_, ax = hist.pad_with(0).plot(drawstyle="steps-mid")

import atompy as ap
import matplotlib.pyplot as plt

plt.style.use("atom")

edges = (0, 1, 2, 3)
hist0 = ap.Hist1d((1, 2, 3), edges, title="H1")
hist1 = ap.Hist1d((10, 20, 30), edges, title="H2")

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["figure.figsize"] = 9.0, 5.0

histos = (
    hist0,
    hist1,
    hist0 - hist1,
    hist0.norm_diff(hist1),
    hist0.norm_to_sum() - hist1.norm_to_sum(),
)
locs = (231, 232, 234, 235, 236)

for hist, loc in zip(histos, locs):
    hist.plot_step(plt.subplot(loc), start_at="auto")

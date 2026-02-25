import matplotlib.pyplot as plt
import atompy as ap

image = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
)
extents = ((0, 1), (10, 11))

hist0 = ap.Hist2d.from_image(image, extents, title="image[0] is top row")
hist1 = ap.Hist2d.from_image(
    image, extents, origin="lower", title="image[0] is bottom row"
)

plt.style.use("atom")
plt.rcParams["axes.grid"] = False
_, axs = plt.subplots(1, 2, layout="compressed", figsize=(6.0, 3.0))
for ax, hist in zip(axs, (hist0, hist1)):
    hist.plot(ax)
    ax.set_box_aspect(1)

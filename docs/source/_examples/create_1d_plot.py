import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

# create a few datasets
x = np.linspace(0, 2, 100)
dataset1 = x, x
dataset2 = x, x**2
dataset3 = x, x**3

# configure the figure using rcParams
plt.rcParams["figure.dpi"] = 300
plt.rcParams["legend.fontsize"] = "small"

# create a figure with one axes
fig, ax = ap.create_1d_plot(
    # separate each dataset by a comma
    dataset1,
    dataset2,
    dataset3,
    labels = (
        "$f(x) = x$",
        "$f(x) = x^2$",
        "$f(x) = x^3$",
    ),
    # configure the plot, if desired
    plot_kwargs_all={"lw":2.0},
    plot_kwargs_per=({"c":"b", "lw":1.0}, {"c":"r"}, {"c":"m"}),
    legend_kwargs={"loc":"upper left"},
    xlabel="$x$",
    ylabel="$f(x)$",
    xmin=x.min(),
    xmax=x.max(),
    ymin=0.0,
)

# if necessary, you can add more things to the axes later
ax.plot(1, 1, "x", markeredgecolor="k")
ax.annotate("Intersection ", (1, 1), horizontalalignment="right")

# or save the figure using fig.savefig("output.pdf")

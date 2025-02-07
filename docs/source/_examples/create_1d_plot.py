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
    # configure the plot, if desired
    plot_kwargs_all=dict(linewidth=2.0),
    plot_kwargs_per=(
        dict(label=r"$f(x) = x$",   color="b", linewidth=1.0),
        dict(label=r"$f(x) = x^2$", color="r"),
        dict(label=r"$f(x) = x^3$", color="m")
    ),
    legend_kwargs=dict(loc="upper left"),
    xlabel="$x$",
    ylabel="$f(x)$",
    xmin=x[0],
    xmax=x[-1],
    ymin=0.0,
)

# if necessary, you can add more things to the axes later
ax.plot(1, 1, "x", markeredgecolor="k")
ax.annotate("Intersection ", (1, 1), horizontalalignment="right")

# or save the figure using fig.savefig("output.pdf")

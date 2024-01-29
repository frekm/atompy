import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

# create a figure with 6 subplots, where the top-right one is a polar plot
projections = [None, None, "polar", None, None, None]
fig, axes = ap.subplots(2, 3, ratio=1, projections=projections)

# plot something top left
axes[0][0].plot([0, 1], "b-")

# plot something in the second column
for a in ap.get_column(1, axes):
    a.plot([0, 1], "r:")

# create a dictionary of axes, so that we can refer to them in a more
# readable way
axes_abc = ap.abcify_axes(axes)

# plot a 2d plot in panel (f)
image = axes_abc["f"].imshow(np.arange(9).reshape((3, 3)))

# add a colorbar to the 2d plot
colorbar = ap.add_colorbar(axes_abc["f"], image)

# change ratio of one panel to 1.618
ap.change_ratio(1.618, axes_abc["d"])

# trim the margins of the figure to be just big enough for all the elements
ap.make_margins_tight(axes, colorbars=colorbar)

# add labels to the axes
ap.add_abc(axes, xoffset=5, yoffset=-5, va="top", prefix="(", suffix=")")

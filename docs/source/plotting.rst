Plotting
========

atompy's :code:`subplots()`
---------------------------

atompy's :code:`subplots()` gives you manual control to set the dimensions of
the figure manually.

When only using, e.g., `plt.subplots <https://matplotlib.org/
stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_, the figure size is
determined using `rcParams["figure.figsize"] <https://matplotlib.org/
stable/users/explain/customizing.html#the-default-matplotlibrc-file>`_.
However, this may result in a figure-height which doesn't match the
ratio of the axes within the figure, resulting in unfitting margins.

This can be alliviated using a `constrained layout <https://matplotlib.org/
stable/gallery/subplots_axes_and_figures/demo_constrained_layout.html>`_,
which, however, removes significant amount of control over the figure
layout (for instance, the figure-width may change).

atompy's :code:`subplots()` returns a grid of axes with margins, paddings,
ratios, the width of the figure (or the width of the axes), etc are fixed by
the user. The height of the figure is generally calculated according to the
needs.

Since the margins (left, right, top, bottom) may not be known at the creation
point of the figure (as they depend on the labels, etc), one can create a
figure/axes-grid first, then readjust the margins such that everything should
fit using :func:`atompy.make_margins_tight`.

What if you want to have axes with different ratios? This is also possible.
After creating a figure with axes using :code:`atompy.subplots()`, you can
change the ratio using :func:`atompy.change_ratio`. Unfortunately, then, 
:func:`atompy.make_margins_tight` may not fully work if one wants to keep
the figure-width fixed (see documentation of
:func:`atompy.make_margins_tight`). It will, however, work perfectly fine
for fixed axes-width (which I think is preferable often enough, anyway).


.. autofunction:: atompy.subplots

Constants
---------

.. data:: atompy.PTS_PER_INCH
    :annotation: = 72.0

.. data:: atompy.MM_PER_INCH
    :annotation: = 25.4

The colors listed below are also wrapped in a class :code:`colors`. One can
access these, e.g., like :code:`colors.RED`.

.. data:: atompy.RED
    :annotation: = "#AE1117"

.. data:: atompy.TEAL
    :annotation: = "#008081"

.. data:: atompy.BLUE
    :annotation: = "#2768F5"

.. data:: atompy.GREEN
    :annotation: = "#007F00"

.. data:: atompy.GREY
    :annotation: = "#404040"

.. data:: atompy.ORANGE
    :annotation: = "#FD8D3C"

.. data:: atompy.PINK
    :annotation: = "#D4B9DA"

.. data:: atompy.YELLOW
    :annotation: = "#FCE205"

.. data:: atompy.LEMON
    :annotation: = "#EFFD5F"

.. data:: atompy.CORN
    :annotation: = "#E4CD05"

.. data:: atompy.PURPLE
    :annotation: = "#CA8DFD"

.. data:: atompy.DARK_PURPLE
    :annotation: = "#9300FF"

.. data:: atompy.FOREST_GREEN
    :annotation: = "#0B6623"

.. data:: atompy.BRIGHT_GREEN
    :annotation: = "#3BB143"



Colorbars
---------

.. autofunction:: atompy.add_colorbar

.. autofunction:: atompy.add_colorbar_large

.. autoclass:: atompy.Colorbar

.. autoclass:: atompy.ColorbarLarge

Colormaps
---------

.. autofunction:: atompy.create_colormap

.. autofunction:: atompy.create_colormap_from_hex


Formatting
----------

.. autofunction:: atompy.square_polar_frame

.. autofunction:: atompy.change_ratio

.. autofunction:: atompy.add_abc

.. autofunction:: atompy.abcify_axes

.. autofunction:: atompy.get_equal_tick_distance

.. autofunction:: atompy.equalize_xtick_distance

.. autofunction:: atompy.equalize_ytick_distance

.. autofunction:: atompy.make_margins_tight

.. autofunction:: atompy.textwithbox



Linestyles and markerstyles
---------------------------

.. autofunction:: atompy.dotted

.. autofunction:: atompy.dash_dotted

.. autofunction:: atompy.dashed

.. autofunction:: atompy.emarker


Miscellaneous
-------------

.. autofunction:: atompy.get_figure_layout

.. autofunction:: atompy.convert_figure_layout_to_relative

.. autoclass:: atompy.FigureLayout
.. autoclass:: atompy.FigureMargins
.. autoclass:: atompy.FigureMarginsFloat






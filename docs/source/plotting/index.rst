Plotting
========

atompy's :func:`.subplots` gives you manual control to set the dimensions of
the figure manually. See the documentation of :func:`.subplots` for basic
usage examples.

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

atompy's :func:`.subplots` returns a grid of axes with margins, paddings,
ratios, the width of the figure (or the width of the axes), etc that are fixed 
by the user. The height of the figure is generally calculated according to the
needs.

.. note::

    Since :func:`.subplots` returns :code:`matplotlib` axes and figures,
    :code:`matplotlib`'s API for plotting is unchanged.

Manipulating margins and ratios after figure creation
-----------------------------------------------------

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

When to use :code:`atompy` for plotting?
----------------------------------------
- :code:`atompy` provides lots of functionality to fine-tune the layout
  of a plot, making it a good choice for finalizing nice plots.
- If you want to add colorbars in a more convent way.

When not to use :code:`atompy` for plotting?
--------------------------------------------
- If you just want to plot some data for ongoing analysis, :code:`atompy`
  is somewhat overkill.
- If you write code that you want to share with other people. Since 
  :code:`atompy` is currently not on PyPI (and won't be anytime soon), it is
  harder to share code with people that don't have :code:`atompy` installed.

Navigation
----------

A selection of constants provided by :code:`atompy` can be found
:doc:`here <constants>`.

.. toctree::
  :maxdepth: 2
  :hidden:

  constants

.. currentmodule:: atompy

Creating and manipulating Figures and Axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  subplots
  abcify_axes
  make_margins_tight
  change_ratio
  square_polar_frame
  get_equal_tick_distance
  equalize_xtick_distance
  equalize_ytick_distance
  add_abc


Colorbars
^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  add_colorbar
  add_colorbar_large
  Colorbar
  ColorbarLarge
  create_colormap
  create_colormap_from_hex


Line and markerstyles
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  dotted
  dash_dotted
  dashed
  emarker

Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  textwithbox
  get_figure_layout
  convert_figure_layout_to_relative
  FigureLayout
  FigureMargins
  FigureMarginsFloat

Examples
^^^^^^^^

.. toctree::

  examples/index
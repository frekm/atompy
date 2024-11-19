Plotting
========

Constants
^^^^^^^^^

Some constants (colors, conversion factors, ...) are provided
at :doc:`constants`.

.. currentmodule:: atompy

Manipulate Figures and Axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  make_me_nice

  add_abc

  set_axes_size

  align_axes_vertically
  align_axes_horizontally

  get_column_pads_inches
  get_row_pads_inches
  set_min_column_pads
  set_min_row_pads

  get_sorted_axes_grid

  get_figure_margins_inches

  get_axes_position_inch
  get_axes_tightbbox_inch
  get_axes_margins_inches

  get_renderer

  square_polar_axes

  add_polar_guideline


Colorbars
^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  add_colorbar
  update_colorbars
  clear_colorbars
  cmap_from_x_to_y


Plotting
^^^^^^^^

.. autosummary::
  :toctree: _autogen

  dotted
  dash_dotted
  dashed
  textwithbox

  ImshowData
  PcolormeshData

Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  Edges
  AliasError
  FigureWidthTooLargeError

.. toctree::
  :hidden:

  constants
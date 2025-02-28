

=========
Constants
=========

.. _constants conversions:

Conversions
-----------

.. data:: atompy.PTS_PER_INCH
    :annotation: = 72.0 pts/inch

.. data:: atompy.MM_PER_INCH
    :annotation: = 25.4 mm/inch



Figure sizes of various journals
--------------------------------

.. data:: atompy.FIGURE_WIDTH_NATURE_1COL
    :annotation: = 3.54 inch
    
.. data:: atompy.FIGURE_WIDTH_NATURE_2COL
    :annotation: = 7.09 inch

.. data:: atompy.FIGURE_WIDTH_PRL_1COL
    :annotation: = 3.375 inch

.. data:: atompy.FIGURE_WIDTH_PRL_2COL
    :annotation: = 6.75 inch

.. data:: atompy.FIGURE_WIDTH_SCIENCE_1COL
    :annotation: = 2.25 inch

.. data:: atompy.FIGURE_WIDTH_SCIENCE_2COL
    :annotation: = 4.75 inch

.. data:: atompy.FIGURE_WIDTH_SCIENCE_3COL
    :annotation: = 7.25 inch


.. _constants colors:

Colors
------

The colors listed below are also wrapped in a tuple :code:`colors`. One can
access these, e.g., like :code:`colors.RED`.

.. data:: atompy.red
    :annotation: = "#AE1117"

.. data:: atompy.teal
    :annotation: = "#008081"

.. data:: atompy.blue
    :annotation: = "#2768F5"

.. data:: atompy.green
    :annotation: = "#007F00"

.. data:: atompy.grey
    :annotation: = "#404040"

.. data:: atompy.orange
    :annotation: = "#FD8D3C"

.. data:: atompy.pink
    :annotation: = "#D4B9DA"

.. data:: atompy.yellow
    :annotation: = "#FCE205"

.. data:: atompy.lemon
    :annotation: = "#EFFD5F"

.. data:: atompy.corn
    :annotation: = "#E4CD05"

.. data:: atompy.purple
    :annotation: = "#CA8DFD"

.. data:: atompy.dark_purple
    :annotation: = "#9300FF"

.. data:: atompy.forest_green
    :annotation: = "#0B6623"

.. data:: atompy.bright_green
    :annotation: = "#3BB143"

Color Palettes
**************

.. _constants palettes:

See the excellent book
`Fundamentals of Data Visualization <https://f0nzie.github.io/dataviz-rsuite/color-basics.html>`__
by Claus O. Wilke for a motivation of these color palettes.

The below color palettes can be used automatically by
:class:`matplotlib.axes.Axes` by updating its cycler (see the respective documentation
at `matplotlib.org <https://matplotlib.org/stable/users/explain/artists/color_cycle.html>`__.)

Alternativley, ``atompy`` provides the :func:`.set_color_cycle` method to
achieve this more conveniently.

.. data:: atompy.PALETTE_OKABE_ITO

    .. plot:: _examples/color_palettes/okabe_ito.py

.. data:: atompy.PALETTE_OKABE_ITO_MUTE

    .. plot:: _examples/color_palettes/okabe_ito_mute.py

.. data:: atompy.PALETTE_OKABE_ITO_ACCENT

    .. plot:: _examples/color_palettes/okabe_ito_accent.py
    
.. data:: atompy.PALETTE_COLORBREWER_DARK2

    .. plot:: _examples/color_palettes/colorbrewer_dark2.py
    
.. data:: atompy.PALETTE_COLORBREWER_MUTE

    .. plot:: _examples/color_palettes/colorbrewer_mute.py

.. data:: atompy.PALETTE_COLORBREWER_ACCENT

    .. plot:: _examples/color_palettes/colorbrewer_accent.py
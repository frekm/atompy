=============================================
:meth:`.Hist1d.plot` and :meth:`.Hist2d.plot`
=============================================

Default plotting methods
========================

:class:`.Hist1d` and :class:`.Hist2d` provide methods for basic plotting, as
showcased by the following examples:

.. plot:: _examples/histogram1d/plot.py
    :include-source:

.. plot:: _examples/histogram1d/plot_in_axes.py
    :include-source:


.. plot:: _examples/histogram2d/plot.py
    :include-source:

.. plot:: _examples/histogram2d/plot_in_axes.py
    :include-source:


Custom plotting methods
=======================

It is possible to create a derived class inheriting from :class:`.Hist1d` or
:class:`.Hist2d`, and implementing a custom plotting methods for these.

The following example showcases the perks of this approach. Note that the plotting
method uses a custom layout engine (similar to :func:`~matplotlib.pyplot.tight_layout`)
provided by `mplutils <https://mpl-utils.readthedocs.io/en/latest/>`__.

.. plot:: _examples/custom_plot_method.py
    :include-source:
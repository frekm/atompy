#############
2D histograms
#############

The examples below assume the following imports:

.. code:: python

    import atompy as ap
    import matplotlib.pyplot as plt
    import numpy as np

First, create a :class:`.Hist2d` object.

You can load a histogram from a text file (:func:`.load_2d_from_txt`),
a `ROOT <https://root.cern.ch/>`_ file (:func:`.load_2d_from_root`), or you 
instantiate it with the output from :func:`numpy.histogram2d`.

.. code:: python

    # load from text file
    hist = ap.load_2d_from_txt("path/to/file.txt", output_format="Hist2d")

    # load from ROOT file
    hist = ap.load_2d_from_root("path/to/file.root", output_format="Hist2d")

    # instantiate from np.histogram output
    hist = ap.Hist2d(*np.histogram(some_xdata, some_ydata))

After you loaded a histogram, you can do a number of operations on it
(for a full list, see :class:`here <.Hist2d>`).

Plot histograms using ``pcolormesh`` or ``imshow``
==================================================

To plot a histogram, you can use :obj:`matplotlib.pyplot.pcolormesh` or
:obj:`matplotlib.pyplot.imshow`.

.. tip::

    When using :obj:`matplotlib.pyplot.pcolormesh`, set the keyword
    ``rasterized = True``. This can *drastically* reduce the resulting file
    size.

For convenience, :class:`.Hist2d` has two properties
:attr:`.Hist2d.for_pcolormesh` and :attr:`.Hist2d.for_imshow`, the result
in appropriate output that can be used for plotting.

Basic usage
-----------

.. code:: python

    x, y, z = hist.for_pcolormesh
    plt.pcolormesh(x, y, z, rasterized=True)

    image, extent = hist.for_imshow
    plt.imshow(image, extent=extent)

:attr:`.Hist2d.for_pcolormesh` and :attr:`.Hist2d.for_imshow` don't simply
return three (or two) arrays, but wrap those in a special class, namely
:class:`.PcolormeshData` and :class:`.ImshowData`, respectively.

This way, the above lines can be compressed to one-liners each

.. code:: python

    plt.pcolormesh(**hist.for_pcolormesh())

    plt.imshow(**hist.for_imshow())

For details on the possible syntax, see the documentation pages of
:class:`.PcolormeshData` and :class:`.ImshowData`.

Change colormap
===============

There are multiple ways to change the colormap that is used for 2D plots in 
``matplotlib``. My favorite one is to configure ``rcParams`` to use the desired
colormap.

.. code:: python

    plt.rcParams["image.cmap"] = "atom"

With the above method, one can use any _registered_ colormap. When loading
``atompy``, an “atomic physics” colormap is registered, which then is used
by the above method.

Instead, one could also use a keyword argument in the plotting call:

.. code:: python

    plt.imshow(**hist.for_imshow(), cmap="atom")

In these examples, zeros are included in the colormap. If this is unwanted,
one can conveniently remove them with the :attr:`.Hist2d.without_zeros`
method (see also :doc:`here <examples_hist2d/without_zeros>`):

.. code:: python

    plt.imshow(**hist.without_zeros.for_imshow())


Common histogram operations
===========================

For an exhaustive overview of the provided methods, see the documentation pages
of :class:`.Hist2d`.

.. _example projections:

Projections
-----------
See also :doc:`here <examples_hist2d/prox>` (x) and
:doc:`here <examples_hist2d/proy>` (y).

Methods :attr:`.Hist2d.projected_onto_x` and :attr:`.Hist2d.projected_onto_y` 
project the 2D histogram onto the _x_ and _y_ axis, respectively, returning
a :class:`.Hist1d` object.

For example

.. code:: python

    hist1d = hist2d.projected_onto_x
    plt.step(*hist1d.for_step)

If one wants to only project the histogram within a specific region, one can
first apply a gate, then project. For example

.. code:: python

    hist1d = hist2d.within_yrange(ymin, ymax).projected_onto_y
    plt.step(*hist1d.for_step)


Applying gates
--------------

As seen in the :ref:`above example <example projections>`, one can apply gates
using :attr:`.Hist2d.within_xrange` or :attr:`.Hist2d.within_yrange`.

For example

.. code:: python

    gated_hist = hist2d.within_xrange(xmin, xmax).within_yrange(ymin.ymax)


All examples
============

.. toctree::
    :glob:

    examples_hist2d/*

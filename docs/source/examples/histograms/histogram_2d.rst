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
    hist = ap.load_2d_from_txt("path/to/file.txt", output="Hist1d")

    # load from ROOT file
    hist = ap.load_2d_from_root("path/to/file.root", output="Hist1d")

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
:meth:`.Hist2d.for_pcolormesh` and :meth:`.Hist2d.for_imshow`, the result
in appropriate output that can be used for plotting.

Basic usage:

.. code:: python

    x, y, z = hist.for_pcolormesh
    plt.pcolormesh(x, y, z, rasterized=True)

    image, extent = hist.for_imshow
    plt.imshow(image, extent=extent)

:meth:`.Hist2d.for_pcolormesh` and :meth:`.Hist2d.for_imshow` don't simply
return three (or two) arrays, but wrap those in a special class, namely
:class:`.PcolormeshData` and :class:`.ImshowData`, respectively.

This way, the above lines can be compressed to one-liners each

.. code:: python

    plt.pcolormesh(**hist.for_pcolormesh())

    plt.imshow(**hist.for_imshow())

For details on the possible syntax, see the documentation pages of
:class:`.PcolormeshData` and :class:`.ImshowData`.

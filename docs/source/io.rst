Input/Output
============

Save histograms
---------------

.. autofunction:: atompy.save_ascii_hist1d

.. autofunction:: atompy.save_ascii_hist2d

Load data as histograms
-----------------------

When loading data as histograms, the output is analogous to numpy's
histogram functions (`histogram <https://numpy.org/doc/stable/reference/
generated/numpy.histogram.html>`_ and `histogram2d <https://numpy.org/doc/
stable/reference/generated/numpy.histogram2d.html>`_).

.. autofunction:: atompy.load_ascii_hist1d
.. autofunction:: atompy.load_ascii_hist2d

.. autofunction:: atompy.load_root_hist1d
.. autofunction:: atompy.load_root_hist2d

Load 1D-data
------------

Loading ASCII data is simply a wrapper function for numpy's `loadtxt <https://
numpy.org/doc/stable/reference/generated/numpy.loadtxt.html>`_ (for
completionist's sake). Loading ROOT-1D-histogram returns the center of the
bin and the corresponding y-value.

.. autofunction:: atompy.load_ascii_data1d
.. autofunction:: atompy.load_root_data1d

Load 2D-data as images
----------------------

2D-data can be displayed using either `imshow <https://matplotlib.org/stable/
api/_as_gen/matplotlib.pyplot.imshow.html>`_ or `pcolormesh <https://
matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html>`_ (or
using some other method that I don't know of). **atompy** provides a method
to import 2D-data such that it can be plotted using `imshow`.

If you want to use `pcolormesh` instead, load the data as a 2D-histogram
(loading either :func:`ASCII <atompy.load_ascii_hist2d>` or
:func:`ROOT <atompy.load_root_hist2d>` data), then plot the data using the 
histogram, xedges, and yedges.

.. autofunction:: atompy.load_ascii_for_imshow
.. autofunction:: atompy.load_root_for_imshow

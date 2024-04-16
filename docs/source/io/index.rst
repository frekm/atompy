============
Input/Output
============

Navigation
----------

.. currentmodule:: atompy

Save histograms
^^^^^^^^^^^^^^^

Save histograms created by `np.histogram <https://numpy.org/doc/stable/
reference/generated/numpy.histogram.html>`_ or `np.histogram2d <https://
numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`_ as 
ASCII-files.

.. autosummary::
    :toctree: _autogen

    save_ascii_hist1d
    save_ascii_hist2d

Load data as histograms
^^^^^^^^^^^^^^^^^^^^^^^

Loading data as histograms returns a :class:`.Hist1d` or :class:`.Hist2d`
instance, which provides some histogram operations (e.g., rebinning,
projections, etc).

.. autosummary::
    :toctree: _autogen

    load_ascii_hist1d
    load_ascii_hist2d
    load_root_hist1d
    load_root_hist2d

Load 1d data
^^^^^^^^^^^^

Loading ASCII data is simply a wrapper function for numpy's `loadtxt <https://
numpy.org/doc/stable/reference/generated/numpy.loadtxt.html>`_ (for
completionist's sake). Loading ROOT-1D-histogram returns the center of the
bin and the corresponding y-value.

.. autosummary::
    :toctree: _autogen

    load_ascii_data1d
    load_root_data1d

Import 2d data for plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

2D-data can be displayed using either `imshow <https://matplotlib.org/stable/
api/_as_gen/matplotlib.pyplot.imshow.html>`_ or `pcolormesh <https://
matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html>`_ (or
using some other method that I don't know of). **atompy** provides
methods to import ASCII data such that it can be plotted with either.

.. autosummary::
    :toctree: _autogen

    import_ascii_for_imshow
    import_root_for_imshow
    import_ascii_for_pcolormesh
    import_root_for_pcolormesh
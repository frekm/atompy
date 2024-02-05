============
Load 1D-data
============

Loading ASCII data is simply a wrapper function for numpy's `loadtxt <https://
numpy.org/doc/stable/reference/generated/numpy.loadtxt.html>`_ (for
completionist's sake). Loading ROOT-1D-histogram returns the center of the
bin and the corresponding y-value.

.. autofunction:: atompy.load_ascii_data1d
.. autofunction:: atompy.load_root_data1d
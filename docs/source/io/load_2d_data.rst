===========================
Import 2D-data for plotting
===========================

2D-data can be displayed using either `imshow <https://matplotlib.org/stable/
api/_as_gen/matplotlib.pyplot.imshow.html>`_ or `pcolormesh <https://
matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html>`_ (or
using some other method that I don't know of). **atompy** provides
methods to import ASCII data such that it can be plotted with either.

.. autofunction:: atompy.import_ascii_for_imshow
.. autofunction:: atompy.import_root_for_imshow

.. autofunction:: atompy.import_ascii_for_pcolormesh
.. autofunction:: atompy.import_root_for_pcolormesh
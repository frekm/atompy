============
Input/Output
============

Navigation
----------

.. currentmodule:: atompy

Save histograms
^^^^^^^^^^^^^^^

Save histograms created by :func:`numpy.histogram` or :func:`numpy.histogram2d` 
as text files.

.. autosummary::
    :toctree: _autogen

    save_1d_as_txt
    save_2d_as_txt


Load data
^^^^^^^^^

You can either load data from a text or a `ROOT <https://root.cern.ch/>`_ file.
You can either load it as a histogram, i.e., as an instance of :class:`.Hist1D` 
or :class:`.Hist2d`, or as NumPy :class:`numpy.ndarray`.

When loading 2D data as ``ndarray``, they are wrapped in a
:class:`.PcolormeshData` (or :class:`.ImshowData`). These are basically
wrapper functions providing some convenience. See documentation of 
:func:`.load_2d_from_txt` and :func:`.load_2d_from_root` for more information.

.. autosummary::
    :toctree: _autogen

    load_1d_from_txt
    load_1d_from_root
    load_2d_from_txt
    load_2d_from_root

Convert 2D data
^^^^^^^^^^^^^^^

If you have a set of ``x``, ``y``, and ``z`` data of the following layout


.. code-block::

    x0 y0 z00
    x1 y0 z10
    x2 y0 z20
    ...
    xM yN zM0
    x0 y1 z01
    x1 y1 z11
    ...
    xM yN zMN

you can convert it into a format suitable for
:func:`matplotlib.pyplot.pcolormesh` or :func:`matplotlib.pyplot.imshow`:

.. autosummary::
    :toctree: _autogen

    for_pcolormesh
    for_imshow
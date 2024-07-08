#############
1D histograms
#############

The examples below assume the following imports:

.. code:: python

    import atompy as ap
    import matplotlib.pyplot as plt
    import numpy as np

First, create a :class:`.Hist1d` object.

You can load a histogram from a text file (:func:`.load_1d_from_txt`),
a `ROOT <https://root.cern.ch/>`_ file (:func:`.load_1d_from_root`), or you 
instantiate it with the output from :func:`numpy.histogram`.

.. code:: python

    # load from text file
    hist = ap.load_1d_from_txt("path/to/file.txt", output="Hist1d")

    # load from ROOT file
    hist = ap.load_1d_from_root("path/to/file.root", output="Hist1d")

    # instantiate from np.histogram output
    hist = ap.Hist1d(*np.histogram(some_data))

After you loaded a histogram, you can do a number of operations on it
(for a full list, see :class:`here <.Hist1d>`).

Reformat for ``plot`` and ``step``
==================================

A histogram is stored as bin-edges and histogram-values, where the *edges* array
is one larger than the *values* array. When plotting a histogram, this has to
be considered (otherwise, the plotted data may be shifted)

:meth:`.Hist1d.for_plot` and :meth:`.Hist1d.for_step` are two convenience
properties of :class:`.Hist1d` that return two arrays which can be used to
plot the histogram using :obj:`matplotlib.pyplot.plot` and
:obj:`matplotlib.pyplot.step`, respectively.

.. attention::

    :obj:`matplotlib.pyplot.step` shifts the plot corresponding to its
    ``where`` keyword. :meth:`.Hist1d.for_step` assumes the default behavior
    of ``where = pre``. Any other keyword will result in incorrectly shifted
    bins.


The following code snippet showcases the usage:

.. code:: python

    x, y = hist.for_plot
    plt.plot(x, y)

    x, y = hist.for_step
    plt.step(x, y)

With this functionality, one can do the above in one line of code by directly
unpacking the returns using the ``*`` operator:

.. code:: python

    # unpack directly in the function call
    plt.plot(*hist.for_plot)
    plt.step(*hist.for_step)


Chain multiple method calls
===========================

Some methods of :class:`.Hist1d` return a new instance of :class:`.Hist1d`.
Then, one can chain multiple operations in one line of code.

E.g., first, one can rebin the histogram, then normalize it to it's maximum,
discard anything that is not within a specific range, then plot it;
all in one line:

.. code:: python

    plt.plot(hist.rebinned(2).normalized_to_integral.within_range(0, 10).for_plot)


``Hist1d`` Examples
===================

.. toctree::
    :glob:

    examples_hist1d/*

    

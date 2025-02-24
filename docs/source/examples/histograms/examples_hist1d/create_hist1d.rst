Create histograms
-----------------

In the following code, three methods are used to create a histogram of
a discrete dataset.

The first method (a,b) creates a histogram using :func:`numpy.histogram` with a 
fixed number of bins, the second (c,d) with a fixed bin width, and the third
(e,f) with :func:`numpy.unique`.

:func:`numpy.unique` counts the occurance of each value in the dataset.
Choosing the appropriate bin width (in this case, 1),
:func:`numpy.histogram` reproduces the output of :func:`numpy.unique`.
However, care is to be taken when plotting the two histograms. As one can see,
plotting using :func:`matplotlib.pyplot.step` and
:func:`matplotlib.pyplot.bar` produce different outputs for both histograming
methods, even though they should be the same.

By default, :func:`numpy.histogram` performs the histogramming with a fixed
number of bins. As one can see in Panels (a) and (b), the choice of how many
bins are used influences the representation of the data. This can be desirable
or not.

`This chapter <https://f0nzie.github.io/dataviz-rsuite/histograms-density-plots.html>`__
from the excellent book
*Fundamentals of Data Visualization* by Claus O. Wilke talks a little bit on
the importance of properly binning histograms. Also check out the documentation
of :func:`numpy.histogram_bin_edges`.


.. plot:: _examples/histogram/create_hist1ds.py
    :include-source:

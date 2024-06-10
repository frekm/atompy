:hide-toc:

======================
atompy's documentation
======================

*atompy* is a collection of functions and classes to do minor data 
analysis and to create plots. Plotting is done using `matplotlib
<https://matplotlib.org/>`_, data analysis is mostly done by `numpy
<https://numpy.org/>`_.

The source code can be found on `github <https://github.com/crono-kircher/
atompy/>`_.

Some commonly used histogram operations (like projections, etc) are implemented
with the :class:`.Hist1d` and :class:`.Hist2d` classes.

Some commonly used vector operations are implemented with the :class:`.Vector`
class.

Some theoretical modelling for atomic physics are provided by the submodules
:code:`atompy.physics` (genereal stuff) and
:code:`atompy.physics.compton_scattering`.

.. toctree::
  :maxdepth: 1

  plotting/index
  analysis/index
  io/index
  physics/index

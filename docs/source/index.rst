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

Overview of provided functionality
----------------------------------

- How to use :code:`atompy` for plotting is documented under
  :doc:`plotting/index`. Some examples are shown 
  :doc:`here<plotting/examples>`.

- An overview of the provided :class:`.Vector` class is given under.
  :doc:`vector`

- An overview of the provided classes :class:`.Hist1d` and :class:`.Hist2d`
  is given under :doc:`histograms/index`.

- How to use :code:`atompy` data stored in ASCII or ROOT files is documented
  under :doc:`io/index`.

- The physics submodule is documented under :doc:`physics/index`.

- The documentation of miscellaneous functions is under :doc:`misc/index`.


.. toctree::
   :hidden:

   plotting/index
   vector
   histograms/index
   io/index
   physics/index
   misc/index

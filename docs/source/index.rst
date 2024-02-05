.. atompy documentation master file, created by
   sphinx-quickstart on Fri Oct 20 13:58:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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

---

Information on how to use :code:`atompy` for plotting.

.. toctree::
   :maxdepth: 2

   plotting/index

.. toctree::
   :maxdepth: 2

   vectors/index

.. toctree::
   :maxdepth: 2

   histograms/index

.. toctree::
   :maxdepth: 2

   io/index

.. toctree::
   :maxdepth: 2

   physics/index

.. toctree::
   :maxdepth: 2

   misc/index

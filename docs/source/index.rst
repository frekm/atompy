======
atompy
======

``atompy`` is a collection of functions and classes to do minor data 
analysis and to create plots. Plotting is done using `matplotlib
<https://matplotlib.org/>`_, data analysis is mostly done by `numpy
<https://numpy.org/>`_.

The source code can be found on `GitHub <https://github.com/crono-kircher/
atompy/>`_.

Overview
========

+----------------------------------------+---------------------------------------------------+
| :doc:`Plotting <plotting/index>`       | Documentation of functionality related to         |
|                                        | manipulating ``matplotlib`` plots.                |
+----------------------------------------+---------------------------------------------------+
| :doc:`Data Analysis <analysis/index>`  | Documentation of provided functionality related   |
|                                        | to data analysis, most notably                    |
|                                        | :doc:`Vector <analysis/vector/index>`,            |
|                                        | :doc:`Hist1d <analysis/histograms/hist1d/index>`, |
|                                        | and                                               |
|                                        | :doc:`Hist2d <analysis/histograms/hist1d/index>`  |
+----------------------------------------+---------------------------------------------------+
| :doc:`Input/Output <io/index>`         | Documentation of convenience functions that load  |
|                                        | (from text or                                     |
|                                        | `ROOT <https://root.cern.ch/>`_                   |
|                                        | files) or save data (to text files).              |
+----------------------------------------+---------------------------------------------------+
| :doc:`Physics <physics/index>`         | Documentation of the physics submodule.           |
+----------------------------------------+---------------------------------------------------+
| :doc:`Examples <examples/index>`       | Examples on how to use ``atompy`` functionality.  |
+----------------------------------------+---------------------------------------------------+

.. _installation:

Installation
============
There are multiple ways of installing ``atompy``.

After installation using any of these methods, you can import ``atompy`` the
usual ways, e.g.,

.. code-block:: python

  import atompy as ap

If you properly install ``atompy`` (that is, any method but the simple drop-in),
I recommend installing it in a `virtual environment <https://docs.python.org/3/library/venv.html>`_.

To create a virutal environment, open a terminal in your working directory,
then run

.. code-block:: shell

  python -m venv .venv

to create a virtual environment called ``.venv``.

Activate it using a script provided in ``.venv/Scripts/``.

Simple drop-in
--------------
The easiest way is to download ``atompy.zip`` from GitHub and drop it into
your working directory.

1. Navigate to the GitHub `Release Page <`https://github.com/frekm/atompy/releases>`__.
2. Download ``atompy.zip`` from the assets list.
3. Extract its contents to your working directory.

This method does not install the dependencies of ``atompy``. You'll have to
install these manually (e.g., with ``pip``).


Using ``git``
-------------

.. code-block:: shell

  pip install git+https//github.com/crono-kircher/atompy

From source
-----------
1. Download the source code from the GitHub `Release Page <`https://github.com/crono-kircher/atompy/releases>`_.
2. Unpack it and run

.. code-block:: shell

  pip install <path>/atompy-<version>/src




.. toctree::
  :hidden:

  plotting/index
  analysis/index
  io/index
  physics/index
  examples/index

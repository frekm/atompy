======
atompy
======

``atompy`` is a collection of functions and classes to do minor data 
analysis and to create plots. Plotting is done using `matplotlib
<https://matplotlib.org/>`_, data analysis is mostly done by `numpy
<https://numpy.org/>`_.

The source code can be found on `GitHub <https://github.com/frekm/
atompy/>`_.

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

1. Navigate to the GitHub `Release Page <https://github.com/frekm/atompy/releases>`__.
2. Download ``atompy.zip`` from the assets list.
3. Extract its contents to your working directory.

This method does not install the dependencies of ``atompy``. You'll have to
install these manually (e.g., with ``pip``).


Using the online repository
---------------------------

With `git`
**********

If you have `git` installed on your system, you can use it to download and install
`atompy` using `pip`.

To install the latest commit, use

.. code-block:: shell

    pip install git+https://github.com/frekm/atompy.git


To install a particular release, e.g., v5.0.0, use

.. code-block:: shell

    pip install git+https://github.com/frekm/atompy.git@v5.0.0

You can add a line to your `requirements.txt`

.. code-block::
    :caption: requirements.txt

    atompy @ git+https://github.com/frekm/atompy.git        # latest version
    atompy @ git+https://github.com/frekm/atompy.git@v5.0.0 # particular version


Without `git`
*************

If you don't have `git` installed on your system, you need to modify the above links.

To install the latest commit, use

.. code-block:: shell

    pip install https://github.com/frekm/atompy/archive/refs/heads/main.zip

To install a particular release, e.g., v5.0.0, use

.. code-block:: shell

    pip install https://github.com/frekm/atompy/archive/refs/heads/main.zip

You can add a line to your `requirements.txt`

.. code-block::
    :caption: requirements.txt

    atompy @ https://github.com/frekm/atompy/archive/refs/heads/main.zip  # latest version
    atompy @ https://github.com/frekm/atompy/archive/refs/tags/v5.0.0.zip # particular version


From source
-----------
1. Download the source code from the GitHub `Release Page <https://github.com/frekm/atompy/releases>`__.
2. Unpack it and run

.. code-block:: shell

    pip install <path>/atompy-<version>/src

Alternatively, if you have `git` installed

.. code-block:: shell

    git clone https://github.com/frekm/atompy.git
    cd atompy
    pip install .

.. .. toctree::




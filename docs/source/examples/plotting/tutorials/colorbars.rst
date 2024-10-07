Colormaps and Colorbars
=======================

Add colorbar to plots
---------------------

Attaching a colorbar to a plot in ``matplotlib`` is quite flexible and,
hence, quite cumbersome. While adding a basic colorbar is quite easy (see
`here <https://matplotlib.org/stable/users/explain/axes/colorbar_placement.html>`_
for a ``matplotlib`` tutorial),
scaling the colorbar correctly can become challenging. The following example
hopefully showcases this:

.. plot:: _examples/colorbars/cbar_matplotlib.py
    :include-source:

As one can see, the colorbar is scaled incorrectly (I'd say). Of course, there
are ways around this... one of them, if you're using ``atompy``, is 
:func:`.add_colorbar`:

.. plot:: _examples/colorbars/cbar_atompy.py
    :include-source:

Additionaly, you can now use :func:`.make_me_nice` to maximize whitespace of
the figure. Adding ``ap.make_me_nice()`` to the above code yields:

.. plot:: _examples/colorbars/cbar_atompy_2.py

For a better overview of the provided functionality, see the documentation page
of :func:`add_colorbar`.

.. note::

    If you plan to use :func:`.make_me_nice`, you **must** add colorbars
    using :func:`.add_colorbar`.



Creating custom colormaps
-------------------------

Sometimes it's useful to create custom colormaps that aren't already available
(you can go through the build in colormaps from ``matplotlib``
`here <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_).

Yo create custom colormaps using 
:obj:`matplotlib.colors.LinearSegmentedColormap`.
In particular, if your colormap does not have any discontinuous jumps in it,
you can use `from_list <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html#matplotlib.colors.LinearSegmentedColormap.from_list>`_.

For example, a colormap that goes red-green-blue is implemented 
like this:

.. code:: python

    from matplotlib.colors import LinearSegmentedColormap

    cm_rgb = LinearSegmentedColormap.from_list(
        "red_green_blue", # name of the colormap
        (0.0, (1.0, 0.0, 0.0)),  # red   to green from 0%  to 50%
        (0.5, (0.0, 1.0, 0.0)),  # green to blue  from 50% to 100%
        (1.0, (0.0, 0.0, 1.0)) 
    )

Now, one can use the colormap in, e.g., :func:`matplotlib.pyplot.imshow`:

.. code:: python

    import matplotlib.pyplot as plt

    plt.imshow(image, cmap=cm_rgb)

Register a colormap
*******************

``matplotlib`` has a register of colormaps, so one can refer to them using
strings, e.g.:

.. code:: python

    plt.imshow(image, cmap="viridis")

Or, one can set a default colormap, removing the need of the keyword
argument, e.g.:

.. code:: python

    plt.rcParams["image.cmap"] = "cividis"
    plt.imshow(image)

You can register your own custom colormap, adding this functionality:

.. code:: python

    import matplotlib.colormaps

    matplotlib.colormaps.register(cm_rgb)

    plt.rcParams["image.cmap"] = "red_green_blue"
    plt.imshow(image)






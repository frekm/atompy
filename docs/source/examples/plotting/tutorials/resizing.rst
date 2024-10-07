=============
Resizing Axes
=============

``matplotlib`` provides multiple ways to maximize the size of the axes such 
that everything still fits. This includes:

- :func:`matplotlib.pyplot.tight_layout`
- `Constrained layout <https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html>`_
- Setting ``bbox_inches = "tight"`` in :func:`matplotlib.pyplot.savefig`.

These work all *within* the figure, meaning that the size of the figure is
overall not changed. For a fixed aspect ratio of the axes, however, the height
of the figure depends on the width. None of the methods listed above adjust
the height of the figure to match a desired aspect ratio of the axes *and*
maximize the used area of the figure.

This may have some shortcomings, as the following example showcases:

.. plot:: _examples/constrained_layout_shortcomings.py
    :include-source:

As one can see, since the height of the figure is fixed, a lot of space is 
wasted. One could manually adjust the height to a more appropriate value,
but that is tedious.

Compare this to the ``atompy`` way:

.. plot:: _examples/constrained_layout_atompy_solution.py
    :include-source:

Of course, :func:`.make_me_nice` has its own shortcomings. It can only
handle n :math:`\times` m grids (i.e., axes cannot span multiple rows or
columns). But for most use-cases, it should suffice.

Additionaly, it handles **colorbars** much nicer. See :doc:`colorbars` for more
information.

See the documentation page of :func:`.make_me_nice` for the full 
functionality.


Unexpected bahaviour and workarounds
====================================

Fixed position and fixed aspect ratios
--------------------------------------

Let's say you have the following setup: The top-row axes is narrower than the
bottom row axes and you want to align it to the edges. You can do this using
the :obj:`matplotlib.axes.Axes.set_position` and :obj:`matplotlib.axes.Axes.get_position`.

.. plot:: _examples/multi_plots_multi_aspects_1.py
    :include-source:

If you now add ``ap.make_me_nice()``, you get the following:

.. plot:: _examples/multi_plots_multi_aspects_2.py

As one can see, the top axes is not aligned
propperly any more. One unfortunately has to re-align the top row for it to
work. As this is a necessity, ``atompy`` provides some methods to streamline
the process, namely :func:`.align_axes_horizontally` and 
:func:`.align_axes_vertically`.

Let's adjust the code:

.. plot:: _examples/multi_plots_multi_aspects_3.py
    :include-source:

Proper alignment!

... unless, of course, a situation like this arises:

.. plot:: _examples/multi_plots_multi_aspects_4.py

Note that the required x-padding is larger for the upper axes. However, since
the axes is re-aligned after the call to :func:`.make_me_nice`, the required
x-padding for the lower axes is used. Of course, that is wrong.

The solution: Throw :func:`.align_axes_horizontally` around more liberally:

.. plot:: _examples/multi_plots_multi_aspects_5.py
    :include-source:

And yet, since we live in a volatile universe where the depths of infitity 
reach out into the pits of that-what-shall-not and nothing is certain but the
uncertainty of life, this still does not result in propper alignment.

If all fails, you can always do minor corrections using the ``margin_pad_pts``
keyword:

.. plot:: _examples/multi_plots_multi_aspects_6.py
    :include-source:

Alternatively, one can use a fixed axes size (instead of a fixed figure size).
Then, these problems should not arise.

.. plot:: _examples/multi_plots_multi_aspects_7.py
    :include-source:

Using the optional keyword ``max_figwidth=True``, one ensures that the new 
figure width is not larger than the desired figure width, which, in turn, 
ensures that the resulting graphic will not reach into the margins of a text 
(in the example, the text width would be specified by ``fh``).







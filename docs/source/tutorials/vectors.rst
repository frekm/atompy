====================
Working with vectors
====================

AplePy provides the :class:`.Vector` and :class:`.VectorArray` classes.

On this page you'll learn how to use them.

Initialization
==============

Initialize a single vector and get/set its components

.. code-block:: python

    >>> vec = ap.Vector(0, 2, 3)
    >>> vec
    Vector(0.0, 2.0, 3.0)
    >>> vec.x
    0.0
    >>> vec.y
    2.0
    >>> vec.z
    3.0
    >>> vec.x = 1
    >>> vec
    Vector(1.0, 2.0, 3.0)
    >>> for c in vec:
    ...     c
    ...
    1.0
    2.0
    3.0


Initialize a collection of vectors:

.. code-block:: python

    >>> vecs = ap.VectorArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> vecs
    VectorArray([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])

Accessing a component returns all respective values:

.. code-block:: python

    >>> vecs.x
    array([1., 4., 7.])
    >>> vecs.y
    array([2., 5., 8.])
    >>> vecs.z
    array([3., 6., 9.])

You can access one vector of the collection:

.. code-block:: python

    >>> vecs[0]
    Vector[1.0, 2.0, 3.0]
    >>> for vec in vecs:
    ...     vec
    ...
    Vector[1.0, 2.0, 3.0]
    Vector[4.0, 5.0, 6.0]
    Vector[7.0, 8.0, 9.0]

You can re-assign components of a vector collection as well

.. code-block:: python

    >>> vecs.x = 10, 11, 12
    >>> vecs.y = 0
    >>> vecs.z = vecs.x
    vecs
    VectorArray([[10.  0. 10.]
                 [11.  0. 11.]
                 [12.  0. 12.]])

You can also reassign single vectors in the collection

.. code-block:: python

    >>> vecs[0] = vec
    >>> vecs[1] = np.array([4, 5, 6])
    >>> vecs[2] = 7, 8, 9
    vecs
    VectorArray([[1. 2. 3.]
                 [4. 5. 6.]
                 [7. 8. 9.]])

You can also access the underlying NumPy array. This can be helpful if you need
to work more efficiently on the data.

.. code-block:: python

    >>> vecs.asarray()
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    >>> vecs.asarray()[1:, :2]
    array([[4., 5.],
           [7., 8.]])
    >>> vecs.asarray()[1:,:2] = 0, 1
    vecs
    VectorArray([[1. 2. 3.]
                 [0. 1. 6.]
                 [0. 1. 9.]])


Common vector operations
========================

Addition and Subtraction
------------------------

You can add and subtract two vectors:

.. code-block:: python

    >>> vec_a = ap.Vector(1, 2, 3)
    >>> vec_a
    Vector(1.0, 2.0, 3.0)
    >>> vec_b = ap.Vector(3, 2, 1)
    >>> vec_b
    Vector(3.0, 2.0, 1.0)
    >>> vec_a + vec_b
    Vector(4.0, 4.0, 4.0)
    >>> vec_a - vec_b
    Vector(-2.0, 0.0, 2.0)

You can perform operations in place:

.. code-block:: python

    >>> vec_b += vec_a
    >>> vec_b
    Vector(4.0, 4.0, 4.0)

The same, of course, is possible for vector arrays. You can also mix single vectors
and vector arrays:

.. code-block:: python

    >>> vec = ap.Vector(1, 2, 3)
    >>> vecs = ap.VectorArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> vec + vecs
    VectorArray([[ 2.  4.  6.]
                 [ 5.  7.  9.]
                 [ 8. 10. 12.]])

You can also perform calculations in place.
Note that adding a :class:`.VectorArray` to a :class:`.Vector` in place converts
it to :class:`.VectorArray`:

.. code-block:: python

    >>> vecs += vec
    vecs
    VectorArray([[ 2.  4.  6.]
                 [ 5.  7.  9.]
                 [ 8. 10. 12.]])
    vec
    Vector(1.0, 2.0, 3.0)
    >>> vec += vecs
    vec
    VectorArray([[ 3.  6.  9.]
                 [ 6.  9. 12.]
                 [ 9. 12. 15.]])
    vecs
    VectorArray([[ 2.  4.  6.]
                 [ 5.  7.  9.]
                 [ 8. 10. 12.]])

The other operators are not implemented.

If you want to multiply a vector by a factor,
use :meth:`.Vector.scale` (:meth:`.VectorArray.scale`).

If you want to perform the dot product with another vector,
use :meth:`.Vector.dot` (:meth:`.VectorArray.dot`).

If you want to perform the outer (cross) product with another vector,
use :meth:`.Vector.cross` (:meth:`.VectorArray.cross`).


.. _tutorial vector scaling:

Scaling a vector
----------------
You can scale a vector using the :meth:`.Vector.scale` and
:meth:`.VectorArray.scale` methods:

.. code-block:: python

    >>> vec = ap.Vector(1, 2, 3)
    >>> vec.scale(10)
    Vector(10.0, 20.0, 30.0)

.. code-block:: python

    >>> vecs = ap.VectorArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> vecs.scale(10)
    VectorArray([[10. 20. 30.]
                 [40. 50. 60.]
                 [70. 80. 90.]])
    >>> vecs.scale((1, 10, 100))
    VectorArray([[  1.  20. 300.]
                 [  4.  50. 600.]
                 [  7.  80. 900.]])

By default, scaling returns a new copy of the vector. If you want to perform scaling
in place, you can set the ``copy`` keyword to ``False``:

.. code-block:: python

    >>> vecs
    VectorArray([[1. 2. 3.]
                 [4. 5. 6.]
                 [7. 8. 9.]])
    >>> vecs.scale(10)
    VectorArray([[10. 20. 30.]
                 [40. 50. 60.]
                 [70. 80. 90.]])
    >>> vecs
    VectorArray([[1. 2. 3.]
                 [4. 5. 6.]
                 [7. 8. 9.]])
    >>> vecs.scale(10, copy=False)
    VectorArray([[10. 20. 30.]
                 [40. 50. 60.]
                 [70. 80. 90.]])
    >>> vecs
    VectorArray([[10. 20. 30.]
                 [40. 50. 60.]
                 [70. 80. 90.]])


.. _tutorial dot product:

Dot product
-----------
You can calculate the dot product of two vectors using the :meth:`.Vector.dot` and
:meth:`.VectorArray.dot` methods:

.. code-block:: python

    >>> vec_a = ap.Vector(1, 2, 3)
    >>> vec_b = ap.Vector(3, 2, 1)
    >>> vec_a.dot(vec_b)
    10.0

.. code-block:: python

    >>> vecs_a = ap.VectorArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> vecs_b = ap.VectorArray([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
    >>> vecs_a.dot(vecs_b)
    array([50., 77., 50.])

You can also mix :class:`.Vector` and :class:`.VectorArray`:

.. code-block:: python

    >>> vecs_a.dot(vec_a)
    array([14., 32., 50.])
    >>> vec_b.dot(vecs_b)
    array([46., 28., 10.])


Cross product
-------------
You can calculate the dot product of two vectors using the :meth:`.Vector.cross` and
:meth:`.VectorArray.cross` methods:


.. code-block:: python

    >>> vec_a = ap.Vector(1, 0, 0)
    >>> vec_b = ap.Vector(0, 1, 0)
    >>> vec_a.cross(vec_b)
    Vector(0.0, 0.0, 1.0)

.. code-block:: python

    >>> vecs_a = ap.VectorArray([[1, 0, 0], [0, 1, 0]])
    >>> vecs_b = ap.VectorArray([[0, 1, 0], [0, 0, 1]])
    >>> vecs_a.cross(vecs_b)
    VectorArray([[0. 0. 1.]
                 [1. 0. 0.]])

Analogously to the :ref:`dot product<tutorial dot product>`, you can mix
:class:`.Vector` and :class:`.VectorArray`:

.. code-block:: python

    >>> vec_a.cross(vecs_b)
    VectorArray([[ 0.  0.  1.]
                 [ 0. -1.  0.]])


.. _tutorial vector magnitude:

Magnitude
---------
You can calculate the magnitude(s) using the :meth:`.Vector.mag` and
:meth:`.VectorArray.mag` methods:

.. code-block:: python

    >>> vec = ap.Vector(1, 2, 3)
    >>> vec.mag()
    3.7416573867739413

.. code-block:: python

    >>> vecs = ap.VectorArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> vecs.mag()
    array([ 3.74165739,  8.77496439, 13.92838828])


Normalizing a vector
--------------------
You can obtain a normalized vector using the :meth:`.Vector.norm` and
:meth:`.VectorArray.norm` methods.

The normalization is performed by calculating the
:ref:`magiutude<tutorial vector magnitude>`, then
:ref:`scaling<tutorial vector scaling>` the vector appropriately.


Angles
------
The polar and azimuth angle are in reference to a
`spherical coordinate system <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`__.

.. math::

    x &= r\sin\theta\cos\phi\\
    y &= r\sin\theta\sin\phi\\
    z &= r\cos\theta

:class:`.Vector` and :class:`.VectorArray` provide methods to calculate :math:`\theta`,
:math:`\cos\theta` and :math:`\phi`:

- :meth:`.Vector.theta`
- :meth:`.VectorArray.theta`
- :meth:`.Vector.cos_theta`
- :meth:`.VectorArray.cos_theta`
- :meth:`.Vector.phi`
- :meth:`.VectorArray.phi`

Furthermore, one can calculate the angle to another vector using
:meth:`.Vector.angle_to` or :meth:`.VectorArray.angle_to`.
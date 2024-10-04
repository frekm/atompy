Sample a random distribution
============================

Parabola example
----------------

Writing Monte-Carlo simulations solely in Python (only using standard modules)
is possibly the slowest thing you can create in the universe, but it 
demonstrates the principle.

Let's sample a parabola using only the Python standard library:

.. literalinclude:: _code/rand_distr_python.py
    :language: python
    :caption: Example 1

Using ``timeit``, one can test the runtime:

.. code-block:: python

    >>> import timeit
    >>> timeit.timeit("sample_parabola((-2, 2), 10_000_000)", number=1)
    9.827801200095564

Luckily, this can be sped up significantly using ``numpy`` methods:

.. literalinclude:: _code/rand_distr_parabola_numpy.py
    :language: python
    :caption: Example 2

This results in a runtime of

.. code-block:: python

    >>> import timeit
    >>> timeit.timeit("sample_parabola((-2, 2), 10_000_000)", number=1)
    0.331170630000997

A speedup of about 30!

This is all well and good if we know our distribution (here, a parabola).
What if we want to do this for an arbitrary distribution?

The next section will go into that.


Sample arbitrary distribution
-----------------------------

.. literalinclude:: _code/rand_distr_arb_distr.py
    :language: python
    :caption: Example 3

The runtime of this is about 5 times slower than the numpy-example above.
Sampling an arbitrary distribution comes at the cost of runtime!


Sample arbitrary analytic function
----------------------------------

What if we combine the first examples (sampling a parabola, i.e., an analytic
function) and the previous example (sampling a completely arbitrary 
distribution)?

.. literalinclude:: _code/rand_distr_callable.py
    :language: python
    :caption: Example 4

The runtime of this is about the same as in Example 2, but we have more
flexibility.


Sample discrete arbitrary distribution
--------------------------------------

Sometimes it is not necessary to get a continuous distribution of values.
In this case, one can use :func:`numpy.random.choice`.

.. literalinclude:: _code/rand_distr_discrete.py
    :language: python
    :caption: Example 5

Even though one is able to sample an arbitrary distribution with this, it is
still about as fast as Example 2!

However, if one is not careful, this may result in
`Moiré patterns <https://de.wikipedia.org/wiki/Moir%C3%A9-Effekt>`_
when histogramming the data into the wrong amount of bins, as illustrated
by the following figure:

.. plot::
    :caption: Moiré pattern when sampling a distribution with 100 discrete
        x-values and histogramming it into 80 bins.

    import matplotlib.pyplot as plt
    import numpy as np

    def sample_discrete_distribution(
        values: np.ndarray,
        probabilities: np.ndarray,
        size: int
    ) -> np.ndarray:
        probabilities_ = probabilities / np.sum(probabilities)
        rng = np.random.default_rng()
        return rng.choice(values, size, p=probabilities_)

    n_bins_distribution = 100
    n_bins_histogram = 80 # which is not 100!

    x = np.linspace(-2, 2, n_bins_distribution)
    y = x**2
    samples = sample_discrete_distribution(x, y, 1_000_000)

    plt.hist(samples, 80)

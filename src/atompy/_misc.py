from os import PathLike
from typing import Any, Literal, Callable, cast
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray, ArrayLike
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib
import mplutils as mplu

import time

from ._errors import UnmatchingEdgesError

cm_atom = mcolors.LinearSegmentedColormap.from_list(
    "atom",
    [
        (0.0, (0.5, 1.0, 1.0)),
        (0.3, (0.0, 0.0, 1.0)),
        (0.7, (1.0, 0.0, 0.0)),
        (1.0, (1.0, 1.0, 0.0)),
    ],
)
matplotlib.colormaps.register(cm_atom, force=True)
cm_atom_from_white = mcolors.LinearSegmentedColormap.from_list(
    "atom_from_white",
    [
        (0.0, (1.0, 1.0, 1.0)),
        (0.065, (0.5, 1.0, 1.0)),
        (0.3, (0.0, 0.0, 1.0)),
        (0.7, (1.0, 0.0, 0.0)),
        (1.0, (1.0, 1.0, 0.0)),
    ],
)
matplotlib.colormaps.register(cm_atom_from_white, force=True)


def _get_topmost_figure(ax: Axes) -> Figure:
    """
    Get the parent figure of `ax`.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`

    Returns
    -------
    figure : :class:`matplotlib.figure.Figure`
    """
    fig = ax.get_figure()
    if fig is None:
        raise ValueError("'ax' is not assigned to any figure")
    while isinstance(fig, SubFigure):
        fig = fig.get_figure()
    return cast(Figure, fig)


def _raise_unmatching_edges(
    a: NDArray[Any], b: NDArray[Any], xy: Literal["x", "y", "xy", ""] = ""
) -> None:
    if not np.allclose(a, b):
        raise UnmatchingEdgesError(xy)


def get_all_dividers(n: int) -> tuple[int, ...]:
    """
    Find all dividers of `n`.

    Parameters
    ----------
    n : int
        Integer to find all dividers from. Must be positive.

    Returns
    -------
    dividers : (int, ...)

    Examples
    --------
    ::

        >>> ap.get_all_dividers(1)
        (1,)
        >>> ap.get_all_dividers(2)
        (1, 2)
        >>> ap.get_all_dividers(6)
        (1, 2, 3, 6)
    """
    if n < 0:
        raise ValueError("n must be positive")
    all_dividers = []
    for divider in range(1, n // 2 + 1):
        if n % divider == 0:
            all_dividers.append(divider)
    all_dividers.append(n)
    return tuple(all_dividers)


def for_pcolormesh_from_root(
    fname: str | PathLike,
    hname: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load data from a `ROOT <https://root.cern.ch/>`__ file to plot it with
    :func:`matplotlib.pyplot.pcolormesh`.

    Parameters
    ----------
    fname : str
        The ROOT filename.

    hname : str
        Within the file, the path/to/the/histogram.

    Returns
    -------
    X, Y, C : ndarray
        Output formatted to work with :func:`matplotlib.pyplot.pcolormesh`.

    See also
    --------
    for_pcolormesh
    for_pcolormesh_from_txt

    Examples
    --------

    .. plot:: _examples/io/for_pcolormesh_from_root.py
        :include-source:


    """
    with uproot.open(fname) as file:  # type: ignore
        values, xedges, yedges = file[hname].to_numpy()  # type: ignore
    return xedges, yedges, values.T


def for_pcolormesh_from_txt(
    fname: str | PathLike,
    *,
    iteration_order: Literal["x_first", "y_first"] = "x_first",
    data_layout: Literal["rows", "columns"] = "columns",
    xyz_indices: tuple[int, int, int] = (0, 1, 2),
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    **loadtxt_kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load 2D data from a text file such that it can be plotted
    with :func:`matplotlib.pyplot.pcolormesh`.

    The file should contain three columns (or rows, depending on `data_layout`)
    that represent the x, y, and z values of the 2D data.

    Parameters
    ----------
    fname : str
        Filename.

    iteration_order : "x_first" or "y_first", default "x_first"
        Specify if the outer iteration is along x or y.

    data_layout : "rows" or "columns", default "columns"
        Specify if the data in the text file is given in rows or columns.

    xyz_indices : (int, int, int), default (0, 1, 2)
        Specify which column (or row) corresponds to what.

    xmin, xmax, ymin, ymax : float, optional
        If x (y) bins do not have constant size, at least one corresponding
        limit has to be provided.

        .. note::

            This does not refer to the limits of the bin centers, but the limits of
            the bin edges!

    **loadtxt_kwargs
        Extra keyword arguments for :func:`numpy.loadtxt`.

        This can be used, e.g., to skip a certain number of lines, or to
        change the column separator.

    See also
    --------
    for_pcolormesh
    for_pcolormesh_from_root
    """
    data = np.loadtxt(fname, **loadtxt_kwargs)
    if data_layout == "columns":
        data = data.T
    elif data_layout != "rows":
        raise ValueError(f"{data_layout=}, but it must be 'rows' or 'columns'")
    return mplu.for_pcolormesh(
        data[xyz_indices[0]],
        data[xyz_indices[1]],
        data[xyz_indices[2]],
        iteration_order=iteration_order,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


def convert_cosine_to_angles(
    cos_angles: ArrayLike, y_data: ArrayLike, full_range: bool = False
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """
    Convert data given as a cosine to radians.

    Parameters
    ----------
    cos_angles : array_like
        cosines of angles, within [-1, 1]

    y_data : array_like
        the corresponding y-data

    full_range : bool
        The range of the output data
        - `True`: 0 .. 2*pi (pi .. 2*pi will be mirrored)
        - `False`: 0 .. pi

    Returns
    -------
    angles : ndarray

    y_data : ndarray

    See also
    --------
    Hist1d.convert_cosine_to_angles

    Examples
    --------
    ::

        >>> import numpy as np
        >>> import atompy as ap
        >>> x, y = np.linspace(-1, 1, 5), np.linspace(0, 4, 5)
        >>> x, y
        (array([-1. , -0.5,  0. ,  0.5,  1. ]), array([0., 1., 2., 3., 4.]))
        >>> ap.convert_cosine_to_angles(x, y)
        (array([0., 1.04, 1.57, 2.09, 3.14]), array([4., 3., 2., 1., 0.]))
        >>> ap.convert_cosine_to_angles(x, y, full_range=True)
        (array([0., 1.04, 1.57, 2.09, 3.14, 3.14, 4.18, 4.71, 5.23, 6.28]),
         array([4., 3., 2., 1., 0., 0., 1., 2., 3., 4.]))
    """
    angles = np.flip(np.arccos(cos_angles))
    y_data = np.flip(y_data)
    if full_range:
        angles = np.append(angles, angles + np.pi)
        y_data = np.append(y_data, np.flip(y_data))
    if not isinstance(y_data, np.ndarray):
        y_data = np.array(y_data)
    return angles, y_data


def centers_to_edges(
    centers: ArrayLike,
    lower: None | float = None,
    upper: None | float = None,
) -> NDArray[np.float64]:
    """
    Work out bin edges from bin centers.

    If the bins don't have constant size, at least one limit has to be
    provided, from which the edges can be determined.

    .. attention::

        If `centers` are not the centers of *all* bins, or if `lower` or `upper`
        are not indeed the lower or upper edge, `centers_to_edges` will silently
        produce nonsense.

    Parameters
    ----------
    centers : array_like, shape(n)
        centers of the bins

    lower, uppper : float, optional
        Lower/upper limits of the range.

        At least one limit must be provided if bins don't have a constant
        size. If both lower and upper limits are provided, the lower one
        will be prioritized.

    Returns
    -------
    edges : ndarray, shape(n+1)
        Edges of the bins.

    See also
    --------
    Hist1d.from_centers : Create a :class:`.Hist1d` directly from centers.

    Examples
    --------
    Get edges from equally spaced centers::

        >>> ap.centers_to_edges([0.5, 1.5, 2.5, 3.5, 4.5])
        [0. 1. 2. 3. 4. 5.]

    Get edges from centers that are not equally spaced::

        >>> ap.centers_to_edges([0.5, 1.5, 3.0, 4.5, 5.5], lower=0.0)
        [0. 1. 2. 4. 5. 6.]
        >>> ap.centers_to_edges([0.5, 1.5, 3.0, 4.5, 5.5], upper=6.0)
        [0. 1. 2. 4. 5. 6.]
    """
    # if bins don't have a constant size, determine xbinedges differently
    centers = np.asarray(centers, copy=True).astype(np.float64)
    edges = np.empty(len(centers) + 1)
    binsizes = np.diff(centers)
    if not np.allclose(binsizes, binsizes[0], atol=0.0):
        if lower is not None:
            # take lower edge and work out binsize forward
            edges[0] = lower
            for i in range(len(centers)):
                edges[i + 1] = 2.0 * centers[i] - edges[i]

        elif upper is not None:
            # take upper edge and work out binsize backward
            edges[-1] = upper
            for i in reversed(range(len(centers))):
                edges[i] = 2.0 * centers[i] - edges[i + 1]
        else:
            # cannot determine binsize, throw exception
            raise ValueError(
                "cannot determine binsizes without 'upper' or 'lower' bounds"
            )
    else:  # bins have equal size
        edges[:-1] = centers - 0.5 * binsizes[0]
        edges[-1] = centers[-1] + 0.5 * binsizes[0]
    return edges


def edges_to_centers(edges: ArrayLike) -> NDArray[np.float64]:
    """
    Calculate centers from *edges*.

    Parameters
    ----------
    edges : array_like, shape (n, )

    Returns
    -------
    centers : ndarray, shape (n-1,)

    Examples
    --------
        >>> ap.edges_to_centers((0, 1, 2, 3, 4, 5))
        array([0.5, 1.5, 2.5, 3.5, 4.5])
    """
    edges = np.asarray(edges).astype(np.float64)
    return edges[:-1] + 0.5 * np.diff(edges)


def gauss(
    x: ArrayLike,
    scale: ArrayLike | Literal["pdf", "integral", "sum", "max"] = "pdf",
    mu: ArrayLike = 0.0,
    sigma: ArrayLike = 1.0,
) -> NDArray[np.float64]:
    """
    Gaussian function.

    Parameters
    ----------
    x : array_like
        x-values where the Gaussian will be evaluated.

    scale : array_like or str
        The scale of the Gaussian.

        If a string, must be one of the following:

        "pdf"
            Return probability density function of a
            `normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`__.

        "integral"
            Return Gaussian normalized to the integral within the `x`-range. `x` must
            be equally spaced.

        "sum"
            Return Gaussian normalized to the sum of all y-values.

        "max"
            Return Gaussian normalized to 1.

    mu : array_like, default 0.0
        Mean or expectation value of the Gaussian.

    sigma : array_like, default 1.0
        Variance of the Gaussian.

    Returns
    -------
    y : ndarray

    Examples
    --------

    .. plot:: _examples/misc/gauss.py

    """
    x = np.asarray(x).astype(np.float64)
    mu = np.asarray(mu).astype(np.float64)
    sigma = np.asarray(sigma).astype(np.float64)
    exp = np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
    if scale == "pdf":
        normfac = 1.0 / sigma / np.sqrt(2 * np.pi)
        return exp * normfac
    elif scale == "integral":
        dx = np.diff(x)
        if not np.all(np.isclose(dx, dx[0], atol=0.0)):
            # TODO calculate integral smarter
            raise ValueError("x must be equally spaced to calculate the integral")
        integral = np.sum(exp) * dx[0]
        return exp / integral
    elif scale == "sum":
        return exp / np.sum(exp)
    elif scale == "max":
        return exp / np.amax(exp)
    else:
        fac = np.asarray(scale).astype(np.float64)
        return fac * exp


def crop(
    x: ArrayLike, y: ArrayLike, lower: float = -np.inf, upper: float = np.inf
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return x,y data where lower <= x < upper.

    Parameters
    ----------
    x, y : array_like
        The x and y data

    lower : float, default -np.inf
        lower limit, inclusive

    upper : float, default +np.inf
        upper limit, exclusive

    Returns
    -------
    new_x : ndarray
        cropped x-data

    new_y : ndarray
        cropped y-data

    Examples
    --------
    ::

        >>> import atompy as ap
        >>> import numpy as np
        >>> x = y = np.arange(6)
        >>> x, y
        (array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 4, 5]))
        >>> ap.crop(x, y, 1, 4)
        (array([1, 2, 3]), array([1, 2, 3]))

    """
    x_ = np.asarray(x)
    y_ = np.asarray(y)
    xi = np.flatnonzero(np.logical_and(x_ >= lower, x_ < upper))
    xout = x_[xi[0] : xi[-1] + 1]
    yout = y_[xi[0] : xi[-1] + 1]
    return xout, yout


def sample_distribution(
    edges: NDArray[np.float64],
    values: NDArray[np.float64],
    size: int,
) -> NDArray:
    """
    Sample a distribution described by ``edges`` and ``values``.

    Parameters
    ----------
    edges : ndarray, shape(n+1,)
        The bin edges from the input distribution. Monotonically increasing.

    values : ndarray, shape(n,)
        The correpsonding values. Must be >=0 everywhere.

    size : int
        Size of the output sample distribution.

    Returns
    -------
    sample : ndarray, shape(size,)
        A sample ranging from ``edges[0]`` to ``edges[-1]`` with
        a distribution corresponding to ``values``.

    Notes
    -----
    See also :doc:`/tutorials/rand_distr`.

    See also
    --------
    sample_distribution_func
    sample_distribution_discrete
    """

    output = np.empty(size)
    output_size = 0

    rng = np.random.default_rng()

    line0 = f"Creating a distribution of {size} samples"
    t0 = time.time()

    while output_size < size:
        line = f"\r{line0}: {100 * output_size/size} percent done."
        print(line, end="")
        buffer = size - output_size
        sample = rng.uniform(edges[0], edges[-1], buffer)
        test = rng.uniform(0.0, np.max(values), buffer)

        edges_index = np.digitize(sample, edges[1:-2])

        sample = np.ma.compressed(
            np.ma.masked_array(sample, test > values[edges_index])
        )

        output[output_size : output_size + sample.size] = sample
        output_size += sample.size

    t1 = time.time()
    print(f"\r{line0}. Total runtime: {t1-t0:.2f}s                           ")

    return output


def sample_distribution_func(
    f: Callable,
    size: int,
    xlimits: tuple[float, float],
    ylimits: tuple[float, float] | Literal["auto"],
    rng: Generator,
    *args,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Sample a distribution described by ``f``.

    Parameters
    ----------
    f : Callable
        Function which shape the distribution should follow. Call signature
        is ``f(x, *args, **kwargs)``.

    size : int
        Size of the distribution.

    xlimits : tuple[int, int]
        Lower and upper limits in-between which to sample the distribution.

    ylimits : tuple[int, int] or ``"auto"``
        Maximum and minimum value of ``f(x)``. Return should be larger zero
        in-between ``xlimits``.

        If ``"auto"``, calculate 100 steps of ``f(x)``, where
        ``xlimits[0] <= x < xlimits[1]`` and set
        ``ylimtis = (0.0, max(f(x)))``.

    Other parameters
    ----------------
    *args
        Additional positional arguements of ``f``.

    **kwargs
        Additional keyword arguments of ``f``.

    Returns
    -------
    sample : ndarray, shape(size,)
        A sample ranging from ``xlimits[0]`` to ``xlimits[1]`` with
        a distribution corresponding to ``f``.

    Notes
    -----
    See also :doc:`/tutorials/rand_distr`.

    See also
    --------
    sample_distribution
    sample_distribution_discrete
    """
    if xlimits[0] > xlimits[1]:
        xlimits = xlimits[1], xlimits[0]

    if ylimits == "auto":
        xtmp = np.linspace(xlimits[0], xlimits[1], 100)
        ytmp = f(xtmp, *args, **kwargs)
        ylimits = (0.0, float(np.max(ytmp)))
    elif ylimits[0] > ylimits[1]:
        ylimits = ylimits[1], ylimits[0]

    output = np.empty(size)
    output_size = 0

    line0 = f"Creating a distribution of {size} samples"

    t0 = time.time()
    while output_size < size:
        line = f"\r{line0}: {100 * output_size/size} percent done."
        print(line, end="")
        buffer = size - output_size
        sample = rng.uniform(xlimits[0], xlimits[-1], buffer)
        test = rng.uniform(ylimits[0], ylimits[1], buffer)

        sample = np.ma.compressed(
            np.ma.masked_array(sample, test > f(sample, *args, **kwargs))
        )

        output[output_size : output_size + sample.size] = sample
        output_size += sample.size

    t1 = time.time()
    print(f"\r{line0}. Total runtime: {t1-t0:.2f}s                           ")

    return output


def sample_distribution_discrete(
    values: NDArray[np.float64], probabilities: NDArray[np.float64], size: int
) -> NDArray[np.float64]:
    """
    Sample a discrete distribution of ``values``, where the probability is
    given by  ``probabilities``.

    Parameters
    ----------
    values : ndarray
        Values that the samples can take.

    probabilities : ndarray
        Corresponding probabilities.

    size : int
        Size of the distribution

    Returns
    -------
    samples : ndarray

    See also
    --------
    sample_distribution
    sample_distribution_func

    Notes
    -----
    Be aware of `Moiré patterns <https://en.wikipedia.org/wiki/Moir%C3%A9_pattern>`_
    when resampling/rebinning the output.

    See also :doc:`/tutorials/rand_distr`.
    """
    probabilities_ = probabilities / np.sum(probabilities)
    rng = np.random.default_rng()
    return rng.choice(values, size, p=probabilities_)

import time
import warnings
from collections.abc import Callable
from os import PathLike
from typing import Any, Literal

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import uproot
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from ._core import NUMBER_T, deprecated_keyword_doing_nothing_msg

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


def detect_iteration_order(
    x: NDArray[np.number], y: NDArray[np.number]
) -> Literal["x_first", "y_first"]:
    """
    Detect if in two arrays x,y, x or y is iterated through first.

    Parameters
    ----------
    x, y : ndarray

    Returns
    -------
    "x_first" or "y_first"

    Raises
    ------
    ValueError
        If iteration order is not detectable from the first two entries
        in x,y, a ValueError is raised.

    Examples
    --------
    ::

        >>> x = np.array((0, 0, 0, 1, 1, 1))
        >>> y = np.array((0, 1, 2, 0, 1, 2))
        >>> ap.detect_iteration_order(x, y)
        "y_first"
        >>> ap.detect_iteration_order(y, x)
        "x_first"
    """
    if np.isclose(x[0], x[1]) and not np.isclose(y[0], y[1]):
        result = "y_first"
    elif not np.isclose(x[0], x[1]) and np.isclose(y[0], y[1]):
        result = "x_first"
    else:
        raise ValueError("could not detect any iteration order")
    return result


def columns_to_meshgrid(
    x: NDArray[NUMBER_T],
    y: NDArray[NUMBER_T],
    z: NDArray[NUMBER_T],
    *,
    ignore_duplicates: bool = False,
    ignore_missing: bool = False,
    fill_value_missing: float = np.nan,
) -> tuple[NDArray[NUMBER_T], NDArray[NUMBER_T], NDArray[NUMBER_T]]:
    """
    Reinterpret data given as a look-up table as a mesh grid.

    Parameters
    ----------
    x, y : ndarray
        Coordinates of the input data points.

    z : ndarray
        Corresponding z-values.

    ignore_duplicates : bool, default False
        If False, duplicate (x, y) pairs cause a ValueError.

        If True, the last z-value for each duplicate pair is used.

    ignore_missing : bool, default False
        If False, missing (x, y) combinations cause a ValueError.

        If True, missing combinations are filled with `fill_value_missing`.

    fill_value_missing : scalar, default np.nan
        Value used to fill missing (x, y) combinations.

    Returns
    -------
    unique_x : ndarray, shape (N,)
        Sorted unique x-values.

    unique_y : ndarray, shape (M,)
        Sorted unique y-values.

    z_meshgrid : ndarray, shape (N, M)
        z_meshgrid[n, m] corresponds to unique_x[n], unique_y[m].

    Examples
    --------

    >>> import atompy as ap
    >>> ap.columns_to_meshgrid(
    ...     (1, 1, 1, 2, 2, 2),
    ...     (1, 2, 3, 1, 2, 3),
    ...     (11, 12, 13, 21, 22, 23)
    ... )
    (array([1, 2]), array([1, 2, 3]), array([[11., 12., 13.], [21., 22., 23.]]))

    If some values are missing in the look-up table, by default, a
    ValueError will be raised.

    >>> ap.columns_to_meshgrid(
    ...     (1, 1, 1, 2, 2),
    ...     (1, 2, 3, 1, 2),
    ...     (11, 12, 13, 21, 22)
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Missing (x, y) combinations found. Set 'ignore_missing' to True if intended.
    >>> ap.columns_to_meshgrid(
    ...     (1, 1, 1, 2, 2),
    ...     (1, 2, 3, 1, 2),
    ...     (11, 12, 13, 21, 22),
    ...     ignore_missing=True
    ... )
    (array([1, 2]), array([1, 2, 3]), array([[11., 12., 13.], [21., 22., nan]]))

    If some values are duplicate, by default, a ValueError will be raised.

    >>> ap.columns_to_meshgrid(
    ...     (1, 1, 1, 2, 2, 2, 2),
    ...     (1, 2, 3, 1, 2, 3, 3),
    ...     (11, 12, 13, 21, 22, 23, 0)
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Duplicate (x, y) pairs found. Set 'ignore_duplicates' to True if intended.
    >>> ap.columns_to_meshgrid(
    ...     (1, 1, 1, 2, 2, 2, 2),
    ...     (1, 2, 3, 1, 2, 3, 3),
    ...     (11, 12, 13, 21, 22, 23, 0),
    ...     ignore_duplicates=True
    ... )
    (array([1, 2]), array([1, 2, 3]), array([[11, 12, 13], [21, 22,  0]]))

    .. plot:: _examples/columns_to_meshgrid.py
        :include-source:
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError("x, y, and z must be one-dimensional arrays")

    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, and z must have the same shape")

    unique_x = np.unique(x)
    unique_y = np.unique(y)

    x_indices = np.searchsorted(unique_x, x)
    y_indices = np.searchsorted(unique_y, y)

    shape = (len(unique_x), len(unique_y))
    flat_indices = np.ravel_multi_index((x_indices, y_indices), shape)

    _, counts = np.unique(flat_indices, return_counts=True)

    if not ignore_duplicates and np.any(counts > 1):
        raise ValueError(
            "Duplicate (x, y) pairs found. Set 'ignore_duplicates' to True if intended."
        )

    n_unique_pairs = len(counts)
    n_grid_points = len(unique_x) * len(unique_y)

    if not ignore_missing and n_unique_pairs != n_grid_points:
        raise ValueError(
            "Missing (x, y) combinations found. Set 'ignore_missing' to True if intended."
        )

    if ignore_missing:
        dtype = np.result_type(z.dtype, fill_value_missing)
        result_z = np.full(shape, fill_value_missing, dtype=dtype)
    else:
        result_z = np.empty(shape, dtype=z.dtype)
    result_z[x_indices, y_indices] = z

    return unique_x, unique_y, result_z


def columns_to_meshgrid_edges(
    xlower: ArrayLike,
    xupper: ArrayLike,
    ylower: ArrayLike,
    yupper: ArrayLike,
    z: ArrayLike,
    *,
    ignore_duplicates: bool = False,
    ignore_missing: bool = False,
    fill_value_missing: float = np.nan,
) -> tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number]]:
    """
    Reinterpret data given as a look-up table as a mesh grid.

    Parameters
    ----------
    xlower, xupper, ylower, yupper : ndarray
        lower/upper x/y edges defining the cells of the z-values.

    z : ndarray
        Corresponding z-values.

    ignore_duplicates : bool, default False
        If False, duplicate (x, y) pairs cause a ValueError.

        If True, the last z-value for each duplicate pair is used.

    ignore_missing : bool, default False
        If False, missing (x, y) combinations cause a ValueError.

        If True, missing combinations are filled with `fill_value_missing`.

    fill_value_missing : scalar, default np.nan
        Value used to fill missing (x, y) combinations.

    Returns
    -------
    x_edges : ndarray, shape (N+1,)
        Sorted unique x-edges.

    y_edges : ndarray, shape (M+1,)
        Sorted unique y-edges.

    z_meshgrid : ndarray, shape (N, M)
        z_meshgrid[n, m] corresponds to x_edges[n] <= x <= x_edges[n+1],
        unique_y[m] <= y <= unique_y[m+1].
    """
    xlower = np.asarray(xlower)
    xupper = np.asarray(xupper)
    ylower = np.asarray(ylower)
    yupper = np.asarray(yupper)
    z = np.asarray(z)

    if any(arr.ndim != 1 for arr in (xlower, xupper, ylower, yupper, z)):
        raise ValueError(
            "xlower, xupper, ylower, yupper, and z must be one-dimensional arrays"
        )

    if not (xlower.shape == xupper.shape == ylower.shape == yupper.shape == z.shape):
        raise ValueError(
            "xlower, xupper, ylower, yupper, and z must have the same shape"
        )

    if np.any(xlower >= xupper) or np.any(ylower >= yupper):
        raise ValueError("Each cell's lower bounds must be less than its upper bounds")

    x_out = np.unique(np.concatenate((xlower, xupper)))
    y_out = np.unique(np.concatenate((ylower, yupper)))

    xlower_indices = np.searchsorted(x_out, xlower)
    xupper_indices = np.searchsorted(x_out, xupper)
    ylower_indices = np.searchsorted(y_out, ylower)
    yupper_indices = np.searchsorted(y_out, yupper)

    # Each row must describe exactly one grid cell.
    if np.any(xupper_indices != xlower_indices + 1) or np.any(
        yupper_indices != ylower_indices + 1
    ):
        raise ValueError(
            "Input cells must correspond to adjacent edges in the output grid"
        )

    shape = (len(x_out) - 1, len(y_out) - 1)

    flat_indices = np.ravel_multi_index((xlower_indices, ylower_indices), shape)

    _, counts = np.unique(flat_indices, return_counts=True)

    if not ignore_duplicates and np.any(counts > 1):
        raise ValueError(
            "Duplicate cells found. Set 'ignore_duplicates' to True if intended."
        )

    n_unique_cells = len(np.unique(flat_indices))
    n_grid_cells = shape[0] * shape[1]

    if not ignore_missing and n_unique_cells != n_grid_cells:
        raise ValueError(
            "Missing cells found. Set 'ignore_missing' to True if intended."
        )

    if ignore_missing:
        dtype = np.result_type(z.dtype, fill_value_missing)
        z_out = np.full(shape, fill_value_missing, dtype=dtype)
    else:
        z_out = np.empty(shape, dtype=z.dtype)
    z_out[xlower_indices, ylower_indices] = z

    return x_out, y_out, z_out


def match_xyz(
    x: NDArray[NUMBER_T],
    y: NDArray[NUMBER_T],
    z: NDArray[NUMBER_T],
    iteration_order: Literal["x_first", "y_first"] = "x_first",
) -> NDArray[NUMBER_T]:
    """
    Match two-dimensional z data to corresponding x and y values.

    Assumes that z[i,j] corresponds to x[i], y[j].

    Parameters
    ----------
    x : ndarray, shape(n,)
        1-dimensional x values.

    y : ndarray, shape(m,)
        1-dimensional y values.

    z : ndarray, shape(n, m)
        2-dimensional z data.

    iteration_order : "x_first" or "y_first", default "x_first"
        Control which dimension iterates through first in the output.

        See examples.

    Returns
    -------
    lut : ndarray, shape(3, n*m)
        Look-up table that matches z to x and y.

        lut[0/1/2] are the x/y/z values, respectively.

    See also
    --------
    match_xy_edges_z

    Examples
    --------

    >>> import atompy as ap
    >>> import numpy as np
    >>> x = np.array((1, 2, 3))
    >>> y = np.array((1, 2))
    >>> z = np.array(((11, 12), (21, 22), (31, 32)))
    >>> ap.match_xyz(x, y, z)
    array([[ 1,  2,  3,  1,  2,  3],
           [ 1,  1,  1,  2,  2,  2],
           [11, 21, 31, 12, 22, 32]])
    >>> ap.match_xyz(x, y, z, iteration_order="y_first")
    array([[ 1,  1,  2,  2,  3,  3],
           [ 1,  2,  1,  2,  1,  2],
           [11, 12, 21, 22, 31, 32]])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")

    if z.shape != (len(x), len(y)):
        raise ValueError(f"{z.shape=} does not match ({len(x)=}, {len(y)=})")

    if iteration_order not in ("x_first", "y_first"):
        raise ValueError(f"{iteration_order=}, but it must be 'x_first' or 'y_first'")

    xx, yy = np.meshgrid(x, y, indexing="ij")

    o = "C" if iteration_order == "y_first" else "F"
    result = np.vstack([xx.ravel(order=o), yy.ravel(order=o), z.ravel(order=o)])

    return result


def match_xy_edges_z(
    xedges: NDArray[NUMBER_T],
    yedges: NDArray[NUMBER_T],
    z: NDArray[NUMBER_T],
    iteration_order: Literal["x_first", "y_first"] = "x_first",
) -> NDArray[NUMBER_T]:
    """
    Match two-dimensional z data to corresponding x and y edges.

    Assumes that z[i,j] belongs to x[i] <= x <= x[i+1], y[j] <= y <= y[j+1].

    Parameters
    ----------
    xedges : ndarray, shape(n,)
        1-dimensional x edges.

    yedges : ndarray, shape(m,)
        1-dimensional y edges.

    z : ndarray, shape(n, m)
        2-dimensional z data.

    iteration_order : "x_first" or "y_first", default "x_first"
        Control which dimension iterates through first in the output.

        See examples.

    Returns
    -------
    lut : ndarray, shape(5, n*m)
        Look-up table that matches z to x- and y-edges.

        - lut[0] are the lower bounds of the x edges.
        - lut[1] are the upper bounds of the x edges.
        - lut[2] are the lower bounds of the y edges.
        - lut[3] are the upper bounds of the y edges.
        - lut[4] are the corresponding z values.

    See also
    --------
    match_xyz

    Examples
    --------

    >>> import atompy as ap
    >>> import numpy as np
    >>> xedges = np.array((0, 1, 2, 3))
    >>> yedges = np.array((0, 1, 2))
    >>> z = np.array(((11, 12), (21, 22), (31, 32)))
    >>> ap.match_xy_edges_z(xedges, yedges, z)
    array([[ 0,  1,  2,  0,  1,  2],
           [ 1,  2,  3,  1,  2,  3],
           [ 0,  0,  0,  1,  1,  1],
           [ 1,  1,  1,  2,  2,  2],
           [11, 21, 31, 12, 22, 32]])
    >>> ap.match_xy_edges_z(xedges, yedges, z, iteration_order="y_first")
    array([[ 0,  0,  1,  1,  2,  2],
           [ 1,  1,  2,  2,  3,  3],
           [ 0,  1,  0,  1,  0,  1],
           [ 1,  2,  1,  2,  1,  2],
           [11, 12, 21, 22, 31, 32]])
    """
    xedges = np.asarray(xedges)
    yedges = np.asarray(yedges)
    z = np.asarray(z)

    if xedges.ndim != 1 or yedges.ndim != 1:
        raise ValueError("xedges and yedges must be one-dimensional")

    expected_shape = (len(xedges) - 1, len(yedges) - 1)
    if z.shape != expected_shape:
        raise ValueError(f"{z.shape=} does not match {expected_shape=}")

    if iteration_order not in ("x_first", "y_first"):
        raise ValueError(f"{iteration_order=} must be 'x_first' or 'y_first'")

    # Each z[i, j] belongs to:
    # xedges[i] <= x <= xedges[i+1]
    # yedges[j] <= y <= yedges[j+1]
    xlower, ylower = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xupper, yupper = np.meshgrid(xedges[1:], yedges[1:], indexing="ij")

    dtype = np.result_type(xedges, yedges, z)
    result = np.empty((5, z.size), dtype=dtype)

    flatten_order = "C" if iteration_order == "y_first" else "F"
    result[0] = xlower.ravel(order=flatten_order)
    result[1] = xupper.ravel(order=flatten_order)
    result[2] = ylower.ravel(order=flatten_order)
    result[3] = yupper.ravel(order=flatten_order)
    result[4] = z.ravel(order=flatten_order)

    return result


def for_pcolormesh(
    xcenters: ArrayLike,
    ycenters: ArrayLike,
    z: ArrayLike,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    ignore_duplicates: bool = False,
    ignore_missing: bool = False,
    fill_value_missing: float = np.nan,
    iteration_order: None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert `xcenters, ycenters, z` such that they can be plotted by
    :func:`matplotlib.pyplot.pcolormesh`.

    Parameters
    ----------
    xcenters : array_like

        A flat sequence of x-coordinates, representing the horizontal position of each
        element in the grid.

        If the bins are not all equal in size, `xmin` or `xmax` needs to be
        specified.

    ycenters : array_like

        A flat sequence of y-coordinates, representing the vertical position of each
        element in the grid.

        If the bins are not all equal in size, `ymin` or `ymax` needs to be
        specified.

    z : array_like
        A flat sequence of data values corresponding to each (xcenters, ycenters)
        coordinate

        The x, y, and z sequences must all be the same length, where each
        z[i] corresponds to the position (xcenters[i], ycenters[i]) in a 2D grid.

    xmin, xmax, ymin, ymax : float, optional
        If x (y) bins do not have constant size, at least one corresponding
        limit has to be provided.

        .. note::

            This does not refer to the limits of the bin centers, but the limits of
            the bin edges!

    ignore_duplicates : bool, default False
        If False, duplicate (x, y) pairs cause a ValueError.

        If True, the last z-value for each duplicate pair is used.

    ignore_missing : bool, default False
        If False, missing (x, y) combinations cause a ValueError.

        If True, missing combinations are filled with `fill_value_missing`.

    fill_value_missing : scalar, default np.nan
        Value used to fill missing (x, y) combinations.

    Other parameters
    ----------------
    iteration_order : None
        .. version-deprecated:: 5.10.1

            Does nothing.

    Returns
    -------
    X, Y, C : ndarray
        Output formatted to work with :func:`~matplotlib.pyplot.pcolormesh`.
    """
    if iteration_order is not None:
        warnings.warn(
            deprecated_keyword_doing_nothing_msg("iteration_order"),
            DeprecationWarning,
            stacklevel=2,
        )

    x_, y_, z_ = columns_to_meshgrid(
        np.asarray(xcenters),
        np.asarray(ycenters),
        np.asarray(z),
        ignore_duplicates=ignore_duplicates,
        ignore_missing=ignore_missing,
        fill_value_missing=fill_value_missing,
    )

    xedges = centers_to_edges(x_, xmin, xmax)
    yedges = centers_to_edges(y_, ymin, ymax)

    return xedges, yedges, z_.T


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
    for_pcolormesh_from_txt
    """
    with uproot.open(fname) as file:  # type: ignore
        values, xedges, yedges = file[hname].to_numpy()
    return xedges, yedges, values.T


def for_pcolormesh_from_txt(
    fname: str | PathLike,
    *,
    data_layout: Literal["rows", "columns"] = "columns",
    xyz_indices: tuple[int, int, int] = (0, 1, 2),
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    ignore_duplicates: bool = False,
    ignore_missing: bool = False,
    fill_value_missing: float = np.nan,
    iteration_order: None = None,
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

    ignore_duplicates : bool, default False
        If False, duplicate (x, y) pairs cause a ValueError.

        If True, the last z-value for each duplicate pair is used.

    ignore_missing : bool, default False
        If False, missing (x, y) combinations cause a ValueError.

        If True, missing combinations are filled with `fill_value_missing`.

    fill_value_missing : scalar, default np.nan
        Value used to fill missing (x, y) combinations.

    Other parameters
    ----------------
    iteration_order : None
        .. version-deprecated:: 5.10.1

            Does nothing.

    **loadtxt_kwargs
        Extra keyword arguments for :func:`numpy.loadtxt`.

        This can be used, e.g., to skip a certain number of lines, or to
        change the column separator.

    See also
    --------
    for_pcolormesh_from_root
    """
    if iteration_order is not None:
        warnings.warn(
            deprecated_keyword_doing_nothing_msg("iteration_order"),
            DeprecationWarning,
            stacklevel=2,
        )
    data = np.loadtxt(fname, **loadtxt_kwargs)
    if data_layout == "columns":
        data = data.T
    elif data_layout != "rows":
        raise ValueError(f"{data_layout=}, but it must be 'rows' or 'columns'")
    return for_pcolormesh(
        data[xyz_indices[0]],
        data[xyz_indices[1]],
        data[xyz_indices[2]],
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        ignore_duplicates=ignore_duplicates,
        ignore_missing=ignore_missing,
        fill_value_missing=fill_value_missing,
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
        integral = np.trapezoid(exp, x)
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
        line = f"\r{line0}: {100 * output_size / size} percent done."
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
    print(f"\r{line0}. Total runtime: {t1 - t0:.2f}s                           ")

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
        line = f"\r{line0}: {100 * output_size / size} percent done."
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
    print(f"\r{line0}. Total runtime: {t1 - t0:.2f}s                           ")

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

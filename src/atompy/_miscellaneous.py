import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Callable, Union, Literal, overload
import matplotlib.pyplot as plt
import time
import warnings
from scipy.optimize import curve_fit, Bounds
from scipy.special import sph_harm
from dataclasses import dataclass
from . import _errors


def get_all_dividers(
    n: int
) -> tuple[int, ...]:
    """
    Return possible rebins of the integer `n`.

    Parameters
    ----------
    n : int

    Returns
    -------
    all_dividers : tuple
        A tuple of all dividers of `n`.

    Examples
    --------
    ::

        >>> get_all_dividers(12)
        (1, 2, 3, 4, 6)

    """
    all_dividers = []
    for divider in range(1, n // 2 + 1):
        if n % divider == 0:
            all_dividers.append(divider)
    return tuple(all_dividers)


def crop(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    lower: float = -np.inf,
    upper: float = np.inf
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Return x,y data where lower <= x <= upper

    Parameters
    ----------
    x, y : array_like
        The x and y data

    lower : float
        lower limit, inclusive

    upper : float
        upper limit, inclusive

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
        >>> ap.crop_data(x, y, 1, 4)
        (array([1, 2, 3, 4]), array([1, 2, 3, 4]))

    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    xi = np.flatnonzero(np.logical_and(x >= lower, x <= upper))
    xout = x[xi[0]:xi[-1] + 1]
    yout = y[xi[0]:xi[-1] + 1]
    return xout, yout


def convert_cosine_to_angles(
    cos_angles: npt.ArrayLike,
    y_data: npt.ArrayLike,
    full_range: bool = False
) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
    """
    Convert data given as a cosine to radians

    Parameters
    ----------
    cos_angles : array_like
        cosines of angles, within [-1, 1]

    y_data : array_like
        the corresponding y-data

    full_range : bool
        The range of the output data
        - `True`: 0 .. 2*pi
        - `False`: 0 .. pi

    Returns
    -------
    angles : ndarray

    y_data : ndarray

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


def integral_sum(
    bincenters: npt.ArrayLike,
    y_data: npt.ArrayLike,
    lower: float = -np.inf,
    upper: float = np.inf
) -> float:
    """
    Get the integral of a histogram by summing the counts weighted by the
    binsize. The binsize needs to be constant.

    Parameters
    ----------
    bincenters : ArrayLike
        center of bins. All bins should have equal size, otherwise return
        doesn't make sense

    y_data : ArrayLike
        corresponding data

    lower, upper : float
        only calculate integral within these bounds, including edges

    Returns
    -------
    integral : float
        The value of the intergral

    Examples
    --------
    ::
        >>> import numpy as np
        >>> import atompy as ap
        >>> x, y = np.linspace(0, 2, 5), np.linspace(0, 4, 5)**2
        >>> x, y
        >>> ap.integral_sum(x, y)
        15.0
    """
    x, y = crop(bincenters, y_data, lower, upper)
    binsize = x[1] - x[0]
    return np.sum(y) * binsize


def integral_polyfit(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    lower: float = -np.inf,
    upper: float = np.inf,
    fit_degree: int = 5,
    showfit: bool = False
) -> float:
    """
    Get the integral of the data. The integral is determined with by
    integrating a polynomial fit.

    Parameters
    ----------
    x, y : array_like
        x, y data

    lower/upper : float, default -/+ np.inf
        if specified, only calculate integral within the given range

    fit_degree : int, default 5
        Degree of the polynomial used for the fit

    showfit : bool, default False
        Show a fit for each set of ydata (to check if fit is any good)

    Returns
    -------
    integral : float
        The value of the integral

    Examples
    --------
    ::
        >>> import numpy as np
        >>> import atompy as ap
        >>> x, y = np.linspace(0, 2, 5), np.linspace(0, 4, 5)**2
        >>> x, y
        ([0.  0.5 1.  1.5 2. ], [ 0.  1.  4.  9. 16.])
        >>> ap.integral_polyfit(x, y, fit_degree=2)
        10.66 
    """
    x, y = crop(x, y, lower, upper)
    if lower == -np.inf:
        lower = np.min(x)
    if upper == np.inf:
        upper = np.max(x)
    coeffs = np.polynomial.polynomial.polyfit(x, y, deg=fit_degree)
    upper_lim, lower_lim = 0.0, 0.0
    for i, coeff in enumerate(coeffs):
        upper_lim += coeff / (i + 1) * upper**(i + 1)
        lower_lim += coeff / (i + 1) * lower**(i + 1)
    integral = upper_lim - lower_lim
    if showfit:
        xt = np.linspace(lower, upper, 500)
        yt = np.polynomial.polynomial.polyval(xt, coeffs)
        sum_integral = integral_sum(x, y, lower, upper)
        plt.plot(x, y, "o", label=f"Int. sum={sum_integral:.2f}")
        plt.plot(xt, yt, label=f"Int. fit={integral:.2f}")
        plt.legend()
        plt.xlim(lower, upper)
        plt.show()
    return integral


def edges_to_centers(
        edges: npt.NDArray[np.float64]
) -> npt.NDArray:
    """
    Return centers of bins discribed by ``edges``.

    Parameters
    ----------
    edges : ndarray, shape(n+1,)

    Returns
    -------
    centers : ndarray, shape(n,)
    """
    return edges[:-1] + 0.5 * np.diff(edges)


def centers_to_edges(
    centers: npt.NDArray[np.float64],
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """
    Work out bin edges from bin centers.

    If the bins don't have constant size, at least one limit has to be
    provided, from which the edges can be determined

    Parameters
    ----------
    centers : ndarray, shape(n)
        centers of the bins

    lower, uppper : float, optional
        Lower and upper limits of the bins.

        At least one limit must be provided if bins don't have a constant 
        size. If both lower and upper limits are provided, the lower one
        will be prioritized.

    Returns
    -------
    edges : ndarray, shape(n+1)
        Edges of the bins.

    See also
    --------
    work_out_bin_edges
    """
    # if bins don't have a constant size, determine xbinedges differently
    edges = np.empty(centers.size + 1)
    binsize = centers[1] - centers[0]
    if np.any(np.abs(np.diff(centers) - binsize) > binsize * 0.001):
        if lower is not None:
            # take lower edge and work out binsize forward
            edges[0] = lower
            for i in range(len(centers)):
                edges[i+1] = 2.0 * centers[i] - edges[i]

        elif upper is not None:
            # take upper edge and work out binsize backward
            edges[-1] = upper
            for i in reversed(range(len(centers))):
                edges[i] = 2.0 * centers[i] - edges[i+1]
        else:
            # cannot determine binsize, throw exception
            raise _errors.UnderdeterminedBinsizeError
    else:  # bins have equal size
        edges[:-1] = centers - 0.5 * binsize
        edges[-1] = centers[-1] + 0.5 * binsize

    return edges


def work_out_bin_edges(
    centers: npt.NDArray[np.float64],
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """
    Alias for :func:`.centers_to_edges`.
    """
    return centers_to_edges(centers, lower, upper)


def sample_distribution(
    edges: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    size: int
) -> npt.NDArray:
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
    See also :doc:`/examples/tutorials/rand_distr`.

    See also
    --------
    sample_distribution_func
    sample_distribution_discrete
    """

    output = np.empty(size)
    output_size = 0

    line0 = f"Creating a distribution of {size} samples"

    rng = np.random.default_rng()

    t0 = time.time()
    while output_size < size:
        line = f"\r{line0}: {100 * output_size/size} percent done."
        print(line, end="")
        buffer = size - output_size
        sample = rng.uniform(edges[0], edges[-1], buffer)
        test = rng.uniform(0.0, np.max(values), buffer)

        edges_index = np.digitize(sample, edges[1:-2])

        sample = np.ma.compressed(np.ma.masked_array(
            sample, test > values[edges_index]))

        output[output_size:output_size + sample.size] = sample
        output_size += sample.size

    t1 = time.time()
    print(f"\r{line0}. Total runtime: {t1-t0:.2f}s                           ")

    return output


def sample_distribution_func(
    f: Callable,
    size: int,
    xlimits: tuple[float, float],
    ylimits: Union[tuple[float, float], Literal["auto"]],
    *args,
    **kwargs,
) -> npt.NDArray[np.float64]:
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
    See also :doc:`/examples/tutorials/rand_distr`.

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

    rng = np.random.default_rng()

    line0 = f"Creating a distribution of {size} samples"

    t0 = time.time()
    while output_size < size:
        line = f"\r{line0}: {100 * output_size/size} percent done."
        print(line, end="")
        buffer = size - output_size
        sample = rng.uniform(xlimits[0], xlimits[-1], buffer)
        test = rng.uniform(ylimits[0], ylimits[1], buffer)

        sample = np.ma.compressed(np.ma.masked_array(
            sample, test > f(sample, *args, **kwargs)))

        output[output_size:output_size + sample.size] = sample
        output_size += sample.size

    t1 = time.time()
    print(f"\r{line0}. Total runtime: {t1-t0:.2f}s                           ")

    return output


def sample_distribution_discrete(
    values: npt.NDArray[np.float64],
    probabilities: npt.NDArray[np.float64],
    size: int
) -> npt.NDArray[np.float64]:
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

    See also :doc:`/examples/tutorials/rand_distr`.
    """
    probabilities_ = probabilities / np.sum(probabilities)
    rng = np.random.default_rng()
    return rng.choice(values, size, p=probabilities_)


@dataclass
class _ImshowDataIter:
    image: npt.NDArray[np.float64]
    extent: npt.NDArray[np.float64]

    def __post_init__(self):
        self.index = 0

    def __iter__(self) -> "_ImshowDataIter":
        return self

    def __next__(self) -> npt.NDArray[np.float64]:
        self.index += 1
        if self.index == 1:
            return self.image
        elif self.index == 2:
            return self.extent
        raise StopIteration


@dataclass
class ImshowData:
    """
    Store an image with its extents, ready to be plotted with imshow.

    Parameters
    ----------
    image : ndarray
        Data. A 2D pixel map of values from bins.

    extent : ndarray
        Array (xmin, xmax, ymin, ymax)

    Examples
    --------
    ::

        # "imdata" is an ImshowData object
        image, extent = imdata
        plt.imshow(image, extent=extent)
        plt.imshow(imdata.image, extent=imdata.extent)
        plt.imshow(imdata(0), extent=imdata(1))
        plt.imshow(imdata[0], extent=imdata[1])
        plt.imshow(**imdata())
    """
    image: npt.NDArray[np.float64]
    extent: npt.NDArray[np.float64]

    def __iter__(self) -> _ImshowDataIter:
        return _ImshowDataIter(self.image, self.extent)

    def __getitem__(
        self,
        index: Literal[0, 1]
    ) -> npt.NDArray[np.float64]:
        if index == 0:
            return self.image
        elif index == 1:
            return self.extent
        else:
            msg = f"{index=}, but it must be 0 (image) or 1 (extent)"
            raise IndexError(msg)

    @overload
    def __call__(self, index: Literal[0, 1]) -> npt.NDArray[np.float64]: ...

    @overload
    def __call__(self, index: None = None) -> dict: ...

    def __call__(
        self,
        index: Optional[Literal[0, 1]] = None
    ) -> Union[npt.NDArray[np.float64], dict]:
        """ Get image, extent, or a dictionary of both.

        The dictionary can be unpacked to conveniently call `plt.imshow 
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.
        imshow.html>`_.

        Parameters
        ----------
        index : 0 or 1, optional
            Specify what to return

            - 0: image
            - 1: extent
            - `None`: A dictionary :code:`{"X": image, "extent":extent}`

        Returns
        -------
        output : `np.ndarray <https://numpy.org/doc/stable/reference/ \
        generated/numpy.ndarray.html>`_ or dict
            See *index*
        """
        if index is None:
            return dict(X=self.image, extent=self.extent)
        else:
            return self[index]


@dataclass
class _PcolormeshDataIter:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]

    def __post_init__(self):
        self.index = 0

    def __iter__(self) -> "_PcolormeshDataIter":
        return self

    def __next__(self) -> npt.NDArray[np.float64]:
        self.index += 1
        if self.index == 1:
            return self.x
        if self.index == 2:
            return self.y
        if self.index == 3:
            return self.c
        raise StopIteration


@dataclass
class PcolormeshData:
    """
    Store 2D data such that it can be plotted with pcolormesh

    See :obj:`matplotlib.pyplot.pcolormesh`

    Parameters
    ----------
    x : ndarray

    y : ndarray

    c : ndarray

    Examples
    --------
    ::

        # 'pcolormesh_data' is a PcolormeshData object
        # following are examples on how to use it to plot things
        X, Y, C = pcolormesh_data
        plt.pcolormesh(X, Y, C)
        plt.pcolormesh(pcolormesh_data.x,
                       pcolormesh_data.y,
                       pcolormesh_data.c)
        plt.pcolormesh(pcolormesh_data[0],
                       pcolormesh_data[1],
                       pcolormesh_data[2])
        plt.pcolormesh(pcolormesh_data(0),
                       pcolormesh_data(1),
                       pcolormesh_data(2))
        plt.pcolormesh(*pcolormesh_data)
        plt.pcolormesh(**pcolormesh_data())
    """
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]

    @property
    def z(self):
        return self.c

    @z.setter
    def z(self, arr: npt.NDArray[np.float64]):
        self.c

    def __getitem__(
        self,
        index
    ) -> npt.NDArray[np.float64]:
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.c
        raise IndexError

    def __call__(
        self,
        index: Optional[Literal[0, 1, 2]] = None
    ) -> Union[npt.NDArray[np.float64], dict]:
        """ Return x, y, c or a dictionary of all three.

        The dictionary can be unpacked to conveniently call
        :obj:`matplotlib.pyplot.pcolormesh`.

        Parameters
        ----------
        index : {0, 1, 2}, optional
            Specify what to return

            - 0: x
            - 1: y
            - 2: c
            - `None`: A dictionary :code:`{"X": x, "Y": y, "C", c}`

        Returns
        -------
        output : ndarray
            See *index*
        """
        if index is None:
            return dict(X=self.x, Y=self.y, C=self.c)
        else:
            return self[index]


def eval_yl0_polynomial(
    thetas: npt.ArrayLike,
    *coeffs
):
    r"""
    Evaluate a polynomial of spherical harmonics.

    Parameters
    ----------
    thetas : ArrayLike
        Polar angle(s) in rad. :math:`0 \le \theta \le \pi`.

    *coeffs
        Coefficients corresponding to the terms. Length of ``coeffs`` determines
        the degree of the polynomial.

    Returns
    -------
    sum_yl0 : ArrayLike
        Sum of spherical harmonics evaluated at ``thetas``.

    See also
    --------
    add_polar_guideline
    fit_yl0_polynomial

    Notes
    -----
    See `scipy.special.sph_harm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html>`__.
    """
    if hasattr(thetas, "__len__"):
        size = len(thetas) # type: ignore
    else:
        size = 1

    output = np.zeros(size)
    for l, coeff in enumerate(coeffs):
        output += coeff * np.real(sph_harm(0, l, 0.0, thetas))
    return output


def fit_yl0_polynomial(
    thetas: npt.ArrayLike,
    intensity: npt.ArrayLike,
    degree: int,
    odd_terms: bool = True
):
    r"""
    Fit data to a polynomial of :math:`\sum_l Y_l^0(\theta, 0)`

    Parameters
    ----------
    thetas : ArrayLike
        Polar angles of data in rad. :math:`0 \le \theta \le \pi`.
    
    intensity : ArrayLike
        Corresponding y-data.

    degree : int
        Maximum degree of the polynomial.

    odd_terms : bool, default ``True``
        If ``False``, restrict to even ``l`` only, ensuring forward, backward
        symmetry.

    Returns
    -------
    coeffs : ndarray
        Coefficients of the polynomial, starting with the lowest order.

    Notes
    -----
    Fit uses `scipy.special.sph_harm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html>`__.

    See also
    --------
    add_polar_guideline
    eval_yl0_polynomial
    """
    p0 = np.full(degree, 0.0)

    lower = np.full(degree, -np.inf)
    upper = np.full(degree, np.inf)

    if not odd_terms:
        for i in range(1, degree+1, 2):
            lower[i] = 0.0
            upper[i] = 0.0 + 1.0e-300

    bounds = Bounds(lower, upper) # type: ignore

    coeffs, _ = curve_fit(
        eval_yl0_polynomial, thetas, intensity, p0=p0, bounds=bounds)
    return coeffs


def fit_polar(
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        deg: int,
        odd_terms: bool = True
) -> npt.NDArray[np.float64]:
    """
    DEPRECATED. Alias for :func:`.fit_yl0`.

    Parameters
    ----------
    x, y : ndarray
        data

    deg : int
        degree of the fit

    odd_terms : bool, default ``True``
        if ``True`` use even and odd terms in the Legendre polynomial.
    """
    warnings.warn("fit_polar is deprecated. Use fit_pl0 instead",
                  DeprecationWarning)
    return fit_yl0_polynomial(x, y, deg, odd_terms=odd_terms)


def eval_polarfit(
    theta: npt.ArrayLike,
    *coeffs
) -> npt.ArrayLike:
    """
    DEPRECATED. Alias for :func:`.eval_yl0`.

    Evaluate legendre polyomial with coefficients ``coeffs``.

    See ``scipy.special.legendre``.

    Parameters
    ----------
    theta : ArrayLike
        angle(s) in rad

    *coeffs
        Coefficients of the Legendre polynomial

    Returns
    -------
    output : ArrayLike
        Evaluate Legendre polynomial squared.
    """
    warnings.warn("eval_polarfit is deprecated. Use eval_pl0 instead",
                  DeprecationWarning)
    return eval_yl0_polynomial(theta, *coeffs)


def eval_polarfit_even(
    theta: npt.ArrayLike,
    *coeffs
) -> npt.ArrayLike:
    """
    DEPRECATED. Use :func:`.eval_yl0` instead.

    Evaluate legendre polyomial with coefficients ``coeffs`` of only even terms.

    See ``scipy.special.legendre``.

    Parameters
    ----------
    theta : ArrayLike
        angle(s) in rad

    *coeffs
        Coefficients of the Legendre polynomial

    Returns
    -------
    output : ArrayLike
        Evaluate Legendre polynomial squared.
    """
    warnings.warn("eval_polarfit_even is deprecated. Use eval_pl0 instead",
                  DeprecationWarning)
    coeffs_all = []
    for coeff_even in coeffs:
        coeffs_all.append(coeff_even)
        coeffs_all.append(0.0)
    return eval_yl0_polynomial(theta, *coeffs_all)



def eval_sph_harm(
    theta: npt.ArrayLike,
    phi: float,
    l: int,
    m: int,
) -> npt.ArrayLike:
    return np.real(sph_harm(m, l, phi, theta))


def fit_sph_harm(
    theta: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    phi: float,
    l: int,
    m: int,
    odd_terms: bool = True
) -> npt.NDArray[np.float64]:
    raise NotImplementedError


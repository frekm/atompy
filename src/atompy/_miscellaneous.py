import numpy as np
import numpy.typing as npt
from typing import Any, Literal, overload, Optional, Union
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from ._histogram import *


class NonconstantBinsizeError(Exception):
    def __init__(
        self,
        fname: str,
        which: Literal["x", "y", ""]
    ) -> None:
        self.fname = fname
        self.which = which

    def __str__(self):
        return (
            f"{self.which}binsizes from {self.fname} are not constant and "
            f"no {self.which}-limits are provided. Provide either "
            f"{self.which}lim[0] or {self.which}lim[1]"
        )


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
) -> tuple[npt.NDArray[np.float_], npt.NDArray[Any]]:
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


def sample_distribution(
    edges: npt.NDArray,
    values: npt.NDArray,
    size: int
) -> npt.NDArray:
    """ 
    Create a sample of `size` that follows a discrete distribution.

    Parameters
    ----------
    edges : ndarray, shape(n,)
        The eft edges of the bins from the input distribution. Monotnoically
        increasing.

    values : ndarray, shape(n,)
        The correpsonding values. Must be >=0 everywhere

    size : int
        size of the output sample distribution

    Returns
    -------
    sample: ndarray, shape(size,)
        A sample ranging from ``distr_edges[0]`` to ``distr_edges[-1]`` with 
        a distribution corresponding to ``distr_values``.
    """

    output = np.empty(size)
    output_size = 0

    line0 = f"Creating a distribution of {size} samples"

    t0 = time.time()
    while output_size < size:
        line = f"\r{line0}: {100 * output_size/size} percent done."
        print(line, end="")
        buffer = size - output_size
        sample = np.random.uniform(edges[0], edges[-1], buffer)
        test = np.random.uniform(0.0, np.max(values), buffer)

        edges_index = np.digitize(sample, edges[1:-1])

        sample = np.ma.compressed(np.ma.masked_array(
            sample, test > values[edges_index]))

        output[output_size:output_size + sample.size] = sample
        output_size += sample.size

    t1 = time.time()
    print(f"\r{line0}. Total runtime: {t1-t0:.2f}s                           ")

    return output


@dataclass
class _ImshowDataIter:
    image: NDArray[np.float64]
    extent: NDArray[np.float64]

    def __post_init__(self):
        self.index = 0

    def __iter__(self) -> "_ImshowDataIter":
        return self

    def __next__(self) -> NDArray[np.float64]:
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

        # "image_data" is an ImshowData object
        image, extent = image_data
        plt.imshow(image, extent=extent)
        plt.imshow(imdata.image, extent=imdata.extent)
        plt.imshow(imdata(0), extent=imdata(1))
        plt.imshow(imdata[0], extent=imdata[1])
        plt.imshow(**imdata())
    """
    image: NDArray[np.float64]
    extent: NDArray[np.float64]

    def __iter__(self) -> _ImshowDataIter:
        return _ImshowDataIter(self.image, self.extent)

    def __getitem__(
        self,
        index: Literal[0, 1]
    ) -> NDArray[np.float64]:
        if index == 0:
            return self.image
        elif index == 1:
            return self.extent
        else:
            msg = f"{index=}, but it must be 0 (image) or 1 (extent)"
            raise IndexError(msg)

    @overload
    def __call__(self, index: Literal[0, 1]) -> NDArray[np.float64]: ...

    @overload
    def __call__(self, index: None = None) -> dict: ...

    def __call__(
        self,
        index: Optional[Literal[0, 1]] = None
    ) -> Union[NDArray[np.float64], dict]:
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
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    c: NDArray[np.float64]

    def __post_init__(self):
        self.index = 0

    def __iter__(self) -> "_PcolormeshDataIter":
        return self

    def __next__(self) -> NDArray[np.float64]:
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
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    c: NDArray[np.float64]

    def __getitem__(
        self,
        index
    ) -> NDArray[np.float64]:
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
    ) -> Union[NDArray[np.float64], dict]:
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

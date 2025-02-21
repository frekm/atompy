import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from dataclasses import dataclass
from . import _io as apio
from . import _miscellaneous as _misc
from typing import Literal, Union


def _check_hist1d_compatibility(
    h1: "Hist1d",
    h2: "Hist1d"
) -> None:
    if h1.edges.shape != h2.edges.shape:
        raise ValueError("histograms do not match")
    if np.any(h1.edges != h2.edges):
        raise ValueError("edges of histograms do not match")


def _check_hist2d_compatibility(
    h1: "Hist2d",
    h2: "Hist2d"
) -> None:
    if h1.xedges.shape != h2.xedges.shape or h1.yedges.shape != h2.yedges.shape:
        raise ValueError("histograms do not match")
    if np.any(h1.xedges != h2.xedges):
        raise ValueError("xedges of histograms do not match")
    if np.any(h1.yedges != h2.yedges):
        raise ValueError("yedges of histograms do not match")


@dataclass
class _Hist1dIterator:
    histogram: NDArray[np.float64]
    edges: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.index = 0

    def __iter__(self) -> "_Hist1dIterator":
        return self

    def __next__(self) -> NDArray[np.float64]:
        self.index += 1
        if self.index == 1:
            return self.histogram
        if self.index == 2:
            return self.edges
        raise StopIteration


@dataclass
class Hist1d:
    _histogram: NDArray[np.float64]
    _edges: NDArray[np.float64]

    def __iter__(self) -> _Hist1dIterator:
        return _Hist1dIterator(self.histogram, self.edges)

    def __add__(self, other: "Hist1d") -> "Hist1d":
        if not isinstance(other, Hist1d):
            raise NotImplemented
        _check_hist1d_compatibility(self, other)
        return Hist1d(self._histogram + other._histogram, self._edges)

    def __sub__(self, other: "Hist1d") -> "Hist1d":
        if not isinstance(other, Hist1d):
            raise NotImplemented
        _check_hist1d_compatibility(self, other)
        return Hist1d(self._histogram - other._histogram, self._edges)

    def __mul__(self, other: "Hist1d") -> "Hist1d":
        if not isinstance(other, Hist1d):
            raise NotImplemented
        _check_hist1d_compatibility(self, other)
        return Hist1d(self._histogram * other._histogram, self._edges)

    def __truediv__(self, other: "Hist1d") -> "Hist1d":
        if not isinstance(other, Hist1d):
            raise NotImplemented
        _check_hist1d_compatibility(self, other)
        return Hist1d(self._histogram / other._histogram, self._edges)

    def __floordiv__(self, other: "Hist1d") -> "Hist1d":
        if not isinstance(other, Hist1d):
            raise NotImplemented
        _check_hist1d_compatibility(self, other)
        return Hist1d(self._histogram // other._histogram, self._edges)

    def __neg__(self) -> "Hist1d":
        return Hist1d(-1.0 * self._histogram, self._edges)

    def __pos__(self) -> "Hist1d":
        return self

    def __eq__(self, other: "Hist1d") -> bool:
        if not isinstance(other, Hist1d):
            raise NotImplemented
        return bool(
            np.all(self._histogram == other._histogram)
            and np.all(self.edges == other.edges)
        )

    def __ne__(self, other: "Hist1d") -> bool:
        if not isinstance(other, Hist1d):
            raise NotImplemented
        return bool(
            np.any(self._histogram != other._histogram)
            and np.any(self.edges != other.edges)
        )

    @property
    def histogram(self) -> NDArray[np.float64]:
        """
        The values of the histogram.

        Returns
        -------
        histogram : ndarray
        """
        return self._histogram

    @histogram.setter
    def histogram(self, _histogram) -> None:
        """
        Set the values of the histogram.

        Parameters
        ----------
        _histogram : ndarray
            Must be of length ``(len(Hist1d.edges) - 1)``
        """
        self._histogram = _histogram

    @property
    def edges(self) -> NDArray[np.float64]:
        """
        Return the bin edges.

        Returns
        -------
        edges : ndarray
        """
        return self._edges

    @edges.setter
    def edges(self, _edges) -> None:
        """
        Set the bin edges.

        Parameters
        ----------
        _edges : ndarray
            Must be of length ``(len(Hist1d.histogram) + 1)``
        """
        self._edges = _edges

    @property
    def centers(
        self
    ) -> NDArray[np.float64]:
        """ Return centers of the histogram's bins """
        return self.edges[:-1] + 0.5 * np.diff(self.edges)

    def rebinned(
        self,
        factor: int
    ) -> "Hist1d":
        """
        Rebin histogram

        Parameters
        ----------
        factor : int
            This is how many old bins will be combined to a new bin.
            Number of old bins divided by factor must be an integer

        Returns
        -------
        new_histogram : Hist1d
            The new, rebinned histogram
        """
        old_n = self.edges.size - 1

        if old_n % factor != 0:
            raise ValueError(
                f"Invalid {factor=}. Possible factors for this "
                f"histogram are {_misc.get_all_dividers(old_n)}."
            )

        new_hist = np.empty(self.histogram.size // factor)
        for i in range(new_hist.size):
            new_hist[i] = np.sum(
                self.histogram[i * factor: i * factor + factor])

        new_edges = np.full(new_hist.size + 1, self.edges[-1])
        for i in range(new_edges.size - 1):
            new_edges[i] = self.edges[i * factor]

        return Hist1d(new_hist, new_edges)

    def save_to_file(
            self,
            fname: str,
            **kwargs
    ) -> None:
        """
        Save histogram to file (i.e., centers and bin-values)

        Parameters
        ----------
        fname : str
            Filename

        Other Parameters
        ----------------
        kwargs
            Additional keyword arguments for :obj:`numpy.savetxt`.
        """
        apio.save_1d_as_txt(self.histogram, self.edges, fname, **kwargs)

    @property
    def for_step(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ Returns right edges and bin-values

        By default, :meth:`matplotlib.axes.Axes.step` needs the right edges of a
        bin and the corresponding bin value. See the *where* keyword argument.

        Returns
        -------
        right_edges : ndarray
            right edges, that is, ``Hist1d.edges[1:]``.

        bin_values : ndarray
            bin values, that is, ``Hist1d.histogram``.

        Examples
        --------
        ::

            # 'hist' is a Hist1d object
            plt.step(*hist.for_step)

            # doesn't work if you specify anything else but 'pre' as a
            # plt.step keyword
            plt.step(*hist.for_step, where="mid") # this will have shifted bins
        """
        return self.edges, np.append(self.histogram[0], self.histogram)

    @property
    def for_plot(self):
        """ Returns bin-centers and bin-values

        Returns
        -------
        centers : ndarray

        bin_values : ndarray

        Examples
        --------
        ::

            # 'hist' is a Hist1d object
            plt.plot(*hist.for_plot)
            # ... which is equivalent to 
            plt.plot(hist.centers, hist.histogram)
        """
        return self.centers, self.histogram

    @property
    def integral(self) -> float:
        """
        Calculate integral of histogram.

        Returns
        -------
        integral : float
            integral = sum(binsizes * histogram_values)
        """
        return np.sum(self.binwidths * self.histogram)  # type: ignore

    @property
    def bins(self) -> int:
        """
        Returns the number of bins
        """
        return self.edges.size - 1
    
    @property
    def binwidths(self) -> NDArray[np.float64]:
        """
        Returns the widths of each bin.
        """
        return np.diff(self.edges)

    @property
    def normalized_to_integral(self) -> "Hist1d":
        """
        Returns the histogram normalized to its integral.

        Calculates integral as bin-width * histogram.

        Returns
        -------
        hist1d : :class:`.Hist1d`
            New, normalized histogram.

        Examples
        --------

        .. plot:: _examples/histogram/normalize_hist1d.py
            :include-source:

        """
        return Hist1d(self.histogram / self.integral, self.edges)

    @property
    def sum(self) -> float:
        """
        Returns sum of histogram.
        """
        return np.sum(self.histogram).astype(float)
    
    @property
    def normalized_to_sum(self) -> "Hist1d":
        return Hist1d(self.histogram / self.sum, self.edges)

    @property
    def max(self) -> float:
        """
        Returns maximum value of the histogram.
        """
        return self.histogram.max()

    @property
    def min(self) -> float:
        """
        Returns minimum value of the histogram.
        """
        return self.histogram.min()

    @property
    def normalized_to_max(self) -> "Hist1d":
        return Hist1d(self.histogram / self.max, self.edges)

    def within_range(
        self,
        range_: tuple[float, float],
        keepdims: bool = False
    ) -> "Hist1d":
        """
        Return a :class:`.Hist1d` only within `range_[0]` and `range_[1]`.

        Parameters
        ----------
        range_ : (float, float)
            left/right edge to be included in the final histogram. Edges
            are included.

        keepdims : bool, default False
            If *True*, keep original dimensions, i.e., the length of xedges
            won't change

        Returns
        -------
        hist1d : :class:`.Hist1d`

        Examples
        --------

        .. plot:: _examples/histogram/within_range.py
            :include-source:
        """
        idx = np.flatnonzero(np.logical_and(
            self.edges >= range_[0], self.edges <= range_[1]))
        if keepdims:
            _H = np.zeros_like(self.histogram)
            _H[idx[0]:idx[-1]] = self.histogram[idx[0]:idx[-1]]
            rtn = Hist1d(_H, self.edges)

        else:
            rtn = Hist1d(
                self.histogram[idx[0]:idx[-1]],
                self.edges[idx[0]:idx[-1] + 1])
        return rtn

    def without_range(
            self,
            range_: tuple[float, float]
    ) -> "Hist1d":
        """
        Return a :class:`.Hist1d` excluding everything within `range_[0]`
        and `range_[1]`.

        Parameters
        ----------
        range_ : (float, float)
            left/right edge of region to be excluded in the final histogram.
            Edges are also excluded.

        keepdims : bool, default False
            If *True*, keep original dimensions, i.e., the length of xedges
            won't change

        Returns
        -------
        hist1d : :class:`.Hist1d`

        Examples
        --------

        .. plot:: _examples/histogram/without_range.py
            :include-source:
        """
        idx = np.flatnonzero(np.logical_and(
            self.edges >= range_[0], self.edges <= range_[1]))
        _H = self.histogram.copy()
        _H[idx[0]:idx[-1]] = np.nan
        return Hist1d(_H, self.edges)


@dataclass
class _Hist2dIterator:
    H: NDArray[np.float64]
    xedges: NDArray[np.float64]
    yedges: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.index = 0

    def __iter__(self) -> "_Hist2dIterator":
        return self

    def __next__(self) -> NDArray[np.float64]:
        self.index += 1
        if self.index == 1:
            return self.H
        if self.index == 2:
            return self.xedges
        if self.index == 3:
            return self.yedges
        raise StopIteration


@dataclass
class Hist2d:
    """
    A numpy wrapper class for the return of :func:`numpy.histogram2d`.

    Parameters
    ----------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram as returned by
        `np.histogram2d <https://numpy.org/doc/stable/reference/generated/
        numpy.histogram2d.html>`_

    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension of *H*

    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension of *H*

    Examples
    --------
    :: 

        import numpy as np
        import atompy as ap
        import matplotlib.pyplot as plt

        # get some random data going
        rng = np.random.default_rng()
        sample_x, sample_y = rng.normal(size=(2, 1_000))

        # initiate a Hist2d instance
        H, x, y = np.histogram2d(sample_x, sample_y)
        hist2d = ap.Hist2d(H, x, y)
        # or do a one-liner
        hist2d = Hist2d(*np.histogram2d(sample_x, sample_y))

        # plot
        plt.pcolormesh(*hist2d.for_pcolormesh)

        # or rebin, then plot
        plt.pcolormesh(*hist2d.rebinned_x(2).for_pcolormesh)

    """
    _H: NDArray[np.float64]
    _xedges: NDArray[np.float64]
    _yedges: NDArray[np.float64]

    def __iter__(self) -> _Hist2dIterator:
        return _Hist2dIterator(self.H, self.xedges, self.yedges)

    def __add__(self, other: "Hist2d") -> "Hist2d":
        if not isinstance(other, Hist2d):
            raise NotImplemented
        _check_hist2d_compatibility(self, other)
        return Hist2d(self._H + other._H, self._xedges, self._yedges)

    def __sub__(self, other: "Hist2d") -> "Hist2d":
        if not isinstance(other, Hist2d):
            raise NotImplemented
        _check_hist2d_compatibility(self, other)
        return Hist2d(self._H - other._H, self._xedges, self._yedges)

    def __mul__(self, other: "Hist2d") -> "Hist2d":
        if not isinstance(other, Hist2d):
            raise NotImplemented
        _check_hist2d_compatibility(self, other)
        return Hist2d(self._H * other._H, self._xedges, self._yedges)

    def __truediv__(self, other: "Hist2d") -> "Hist2d":
        if not isinstance(other, Hist2d):
            raise NotImplemented
        _check_hist2d_compatibility(self, other)
        return Hist2d(self._H / other._H, self._xedges, self._yedges)

    def __floordiv__(self, other: "Hist2d") -> "Hist2d":
        if not isinstance(other, Hist2d):
            raise NotImplemented
        _check_hist2d_compatibility(self, other)
        return Hist2d(self._H // other._H, self._xedges, self._yedges)

    def __neg__(self) -> "Hist2d":
        return Hist2d(-1.0 * self._H, self._xedges, self._yedges)

    def __pos__(self) -> "Hist2d":
        return self

    def __eq__(self, other: "Hist2d") -> bool:
        return bool(
            np.all(self._H == other._H)
            and np.all(self._xedges == other._xedges)
            and np.all(self._yedges == other._yedges)
        )

    def __ne__(self, other: "Hist2d") -> bool:
        if not isinstance(other, Hist2d):
            raise NotImplemented
        return bool(
            np.any(self._H != other._H)
            and np.any(self._xedges != other._xedges)
            and np.any(self._yedges != other._yedges)
        )

    @property
    def H(self) -> NDArray[np.float64]:
        """
        The bi-dimensional histogram.

        Returns
        -------
        H : ndarray
        """
        return self._H

    @H.setter
    def H(self, _H: NDArray[np.float64]) -> None:
        """
        Set the bi-dimensional histogram to `_H`.

        Parameters
        ----------
        _H : ndarray
        """
        self._H = _H

    @property
    def xedges(self) -> NDArray[np.float64]:
        """
        The bin edges along the first dimension of *Hist2d.H*

        Returns
        -------
        xedges : ndarray
        """
        return self._xedges

    @xedges.setter
    def xedges(self, _xedges: NDArray[np.float64]) -> None:
        """
        Set the bin edges along the first dimension of *Hist2d.H*

        Parameters
        ----------
        _xedges : ndarray
        """
        self._xedges = _xedges

    @property
    def yedges(self) -> NDArray[np.float64]:
        """
        The bin edges along the second dimension of *Hist2d.H*

        Returns
        -------
        yedges : ndarray
        """
        return self._yedges

    @yedges.setter
    def yedges(self, _yedges: NDArray[np.float64]) -> None:
        """
        Set the bin edges along the second dimension of *Hist2d.H*

        Parameters
        ----------
        _yedges : ndarray
        """
        self._yedges = _yedges

    @property
    def xcenters(self) -> NDArray[np.float64]:
        """
        Get center of bins along first dimension of ``Hist2d.H``.

        Returns
        -------
        xcenters : ndarray
        """
        return self.xedges[:-1] + 0.5 * np.diff(self.xedges)

    @property
    def ycenters(self) -> NDArray[np.float64]:
        """
        Get center of bins along second dimension of ``Hist2d.H``

        Returns
        -------
        xcenters : ndarray
        """
        return self.yedges[:-1] + 0.5 * np.diff(self.yedges)

    def rebinned_x(self, factor: int) -> "Hist2d":
        """
        Rebin x-dimension of histogram.

        Return a *Hist2d* where the first dimension was rebinned with a
        factor of *factor*

        Parameters
        ----------
        factor : int
            This is how many old bins will be combined to a new bin. The number
            of old bins divided by *factor* must be integer.

        Returns
        -------
        rebinned_hist2d : Hist2d
            the new (rebinned) histogram
        """
        old_n = self.xedges.size - 1
        if old_n % factor != 0:
            raise ValueError(
                f"Invalid {factor=}. Possible factors for this "
                f"histogram are {_misc.get_all_dividers(old_n)}."
            )
        new_hist = np.empty((self.H.shape[0] // factor, self.H.shape[1]))
        for i in range(new_hist.shape[0]):
            new_hist[i, :] = np.sum(
                self.H[i * factor:i * factor + factor, :], axis=0)
        new_xedges = np.full(new_hist.shape[0] + 1, self.xedges[-1])
        for i in range(new_xedges.size):
            new_xedges[i] = self.xedges[i * factor]
        return Hist2d(new_hist, new_xedges, self.yedges)

    def rebinned_y(self, factor: int) -> "Hist2d":
        """
        Rebin y-dimension of histogram.

        Return a *Hist2d* where the second dimension was rebinned with a
        factor of *factor*

        Parameters
        ----------
        factor : int
            This is how many old ybins will be combined to a new bin.
            Number of old ybins divided by factor must be an integer

        Returns
        -------
        rebinned_hist2d : Hist2d
            the new (rebinned) histogram
        """
        old_n = self.yedges.size - 1
        if old_n % factor != 0:
            raise ValueError(
                f"Invalid {factor=}. Possible factors for this "
                f"histogram are {_misc.get_all_dividers(old_n)}."
            )
        new_hist = np.empty((self.H.shape[0], self.H.shape[1] // factor))
        for i in range(new_hist.shape[1]):
            new_hist[:, i] = np.sum(
                self.H[:, i * factor:i * factor + factor], axis=1)
        new_yedges = np.full(new_hist.shape[1] + 1, self.yedges[-1])
        for i in range(new_yedges.size):
            new_yedges[i] = self.yedges[i * factor]
        return Hist2d(new_hist, self.xedges, new_yedges)

    @property
    def for_pcolormesh(
        self
    ) -> _misc.PcolormeshData:
        """
        Return such that it can be plotted using
        :obj:`matplotlib.pyplot.pcolormesh`

        The input order for `pcolormesh` is xedges, yedges, H.T

        Returns
        -------
        xedges : ndarray

        yedges : ndarray

        H.T : ndarray
            Transposed matrix of input *Hist2d.H*

        Examples
        --------
        ::

            hist = Hist2d(*np.histogram2d(xsamples, ysamples))
            plt.pcolormesh(*hist.for_pcolormesh)
        """
        return _misc.PcolormeshData(self.xedges, self.yedges, self.H.T)

    @property
    def for_imshow(
        self,
    ) -> _misc.ImshowData:
        """
        Return corresponding :class:`.ImshowData` object.

        Assumes that the origin of the image is specified by 
        `matplotlib.rcParams["image.origin"]`

        Returns
        -------
        imshow_data : :class:`.ImshowData`
            A data type storing an image and the extent of the image.

            - :code:`imshow_data.image`: 2d pixel array
            - :code:`imshow_data_extent`: the extents of the image

        Examples
        --------
        ::

            hist = Hist2d(*np.histogram2d(xsamples, ysamples))
            image, extent = hist.for_imshow
            plt.imshow(image, extent=extent)

            # or (calling *imshow_data* returns a dictionary)
            plt.imshow(**hist.for_imshow())

            # or
            imshow_data = hist.for_imshow
            plt.imshow(imshow_data.image, extent=imshow_data.extent)
        """
        if np.any(np.diff(
                self.xedges) < (self.xedges[1] - self.xedges[0]) * 0.001):
            raise ValueError("xbinsize not constant, use pcolormesh instead")
        if np.any(np.diff(
                self.yedges) < (self.yedges[1] - self.yedges[0]) * 0.001):
            raise ValueError("ybinsize not constant, use pcolormesh instead")
        origin = plt.rcParams["image.origin"]
        if origin == "lower":
            H_ = self.H.T
        else:
            H_ = np.flip(self.H.T, axis=0)
        edges = [self.xedges.min(),
                 self.xedges.max(),
                 self.yedges.min(),
                 self.yedges.max()]
        return _misc.ImshowData(H_, np.array(edges))

    def within_xrange(
        self,
        xrange: tuple[float, float],
        keepdims: bool = False
    ) -> "Hist2d":
        """
        Apply an inclusive gate along x.

        Return a histogram where *xrange[0]* <= xedges <= *xrange[1]*

        Parameters
        ----------
        xrange : (float, float)
            lower/upper xedge to be included in the final histogram. Edges
            are included.

        keepdims : bool, default False
            If *True*, keep original dimensions, i.e., the length of xedges
            won't change

        Returns
        -------
        new_histogram : :class:`Hist2d`
        """
        idx = np.flatnonzero(np.logical_and(
            self.xedges >= xrange[0], self.xedges <= xrange[1]))
        if keepdims:
            _H = np.zeros_like(self.H)
            _H[idx[0]:idx[-1], :] = self.H[idx[0]:idx[-1], :]
            rtn = Hist2d(_H, self.xedges, self.yedges)

        else:
            rtn = Hist2d(
                self.H[idx[0]:idx[-1], :],
                self.xedges[idx[0]:idx[-1] + 1],
                self.yedges)
        return rtn

    def within_yrange(
        self,
        yrange: tuple[float, float],
        keepdims: bool = False
    ) -> "Hist2d":
        """
        Apply an inclusive gate along y.

        Return a histogram where *yrange[0]* <= yedges <= *yrange[1]*

        Parameters
        ----------
        yrange : (float, float)
            lower/upper xedge to be included in the final histogram. Edges
            are included.

        keepdims : bool, default False
            If *True*, keep original dimensions, i.e., the length of yedges
            won't change

        Returns
        -------
        new_histogram : :class:`.Hist2d`
        """
        idx = np.flatnonzero(np.logical_and(
            self.yedges >= yrange[0], self.yedges <= yrange[1]))
        if keepdims:
            _H = np.zeros_like(self.H)
            _H[:, idx[0]:idx[-1]] = self.H[:, idx[0]:idx[-1]]
            rtn = Hist2d(_H, self.xedges, self.yedges)

        else:
            rtn = Hist2d(
                self.H[:, idx[0]:idx[-1]],
                self.xedges,
                self.yedges[idx[0]:idx[-1] + 1])
        return rtn

    def without_xrange(
            self,
            xrange: tuple[float, float],
    ) -> "Hist2d":
        """
        Apply an exclusive gate along x.

        Return a histogram where every bin with
        *xedges[0]* <= bin xedges <= *xedges[1]* is numpy.nan.

        Parameters
        ----------
        xrange : (float, float)
            lower/upper xedge to be excluded from the final histogram. Edges
            are excluded.

        Returns
        -------
        new_histogram : :class:`Hist2d`

        Examples
        --------
        .. plot:: _examples/histogram/without_xrange.py
            :include-source:
        """
        idx = np.flatnonzero(np.logical_and(
            self.xedges >= xrange[0], self.xedges <= xrange[1]))
        _H = self.H.copy()
        _H[idx[0]:idx[-1], :] = np.nan
        return Hist2d(_H, self.xedges, self.yedges)

    def without_yrange(
            self,
            yrange: tuple[float, float],
    ) -> "Hist2d":
        """
        Apply an exclusive gate along y.

        Return a histogram where every bin with
        yxedges[0]* <= bin yedges <= *yedges[1]* is numpy.nan.

        Parameters
        ----------
        yrange : (float, float)
            lower/upper yedge to be excluded from the final histogram. Edges
            are excluded.

        Returns
        -------
        new_histogram : :class:`.Hist2d`

        Examples
        --------
        .. plot:: _examples/histogram/without_yrange.py
            :include-source:
        """
        idx = np.flatnonzero(np.logical_and(
            self.yedges >= yrange[0], self.yedges <= yrange[1]))
        _H = self.H.copy()
        _H[:, idx[0]:idx[-1]] = np.nan
        return Hist2d(_H, self.xedges, self.yedges)

    @property
    def projected_onto_x(self) -> "Hist1d":
        """
        Project histogram onto its x-axis

        Returns
        -------
        hist1d : :class:`.Hist1d`
            A 1D histogram where the *bins* are the x-bins of the original
            2D histogram, and the *bin-values* are the projection.

        Examples
        --------

        .. plot:: _examples/histogram/projection_x.py
            :include-source:
        """
        return Hist1d(np.sum(self.H, axis=1), self.xedges)

    @property
    def prox(self) -> "Hist1d":
        """
        Alias for :meth:`.projected_onto_x`.
        """
        return self.projected_onto_x

    @property
    def projected_onto_y(self) -> "Hist1d":
        """
        Project histogram onto its y-axis

        Returns
        -------
        hist1d : :class:`.Hist1d`
            A 1D histogram where the *bins* are the x-bins of the original
            2D histogram, and the *bin-values* are the projection.

        Examples
        --------

        .. plot:: _examples/histogram/projection_y.py
            :include-source:
        """
        return Hist1d(np.sum(self.H, axis=0), self.yedges)

    @property
    def proy(self) -> "Hist1d":
        """
        Alias for :meth:`.projected_onto_y`.
        """
        return self.projected_onto_y

    def __get_profile(
        self,
        counts: NDArray,
        bin_centers: NDArray,
        option: Literal["", "s", "i", "g"] = ""
    ):
        """
        See `TProfile <https://root.cern.ch/doc/master/classTProfile.html>`_
        of the ROOT Data Analysis Framework 
        """
        H = np.sum(counts * bin_centers)
        E = np.sum(counts * bin_centers**2)
        W = np.sum(counts)
        h = H / W
        s = np.sqrt(E / W - h**2)
        e = s / np.sqrt(W)
        if option == "":
            out = e
        elif option == "s":
            out = s
        elif option == "i":
            raise NotImplementedError
        elif option == "g":
            out = 1. / np.sqrt(W)
        else:
            msg = f"{option=}, but it needs to be '', 's', 'i', or 'g'"
            raise ValueError(msg)
        return h, out

    def get_profile_along_x(
        self,
        option: Literal["", "s", "i", "g"] = ""
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Get the x-profile.

        Calculate the mean value and its error per column. Which type of error
        is controled with *option*.

        Parameters
        ----------
        option : :code:`""`, :code:`"s"`, :code:`"i"`, or :code:`"g"`, default :code:`""`
            Control type of errors, see *Notes*.

            - :code:`option == ""`:  Error of the mean of all Y values
            - :code:`option == "s"`: Standard deviation of all Y
            - :code:`option == "i"`: See *Notes*
            - :code:`option == "g"`: Error of a weighted mean for combining
              measurements with variances of :math:`w`.


        Returns
        -------
        h : ndarray

            Mean values of each Y (i.e., column)

        errors : ndarray

            Errors of the mean values. Depends on `option`.

        Notes
        -----
        See `TProfile <https://root.cern.ch/doc/master/classTProfile.html>`_
        of the ROOT Data Analysis Framework.

        For a histogram :math:`X` vs :math:`Y`, the following is calculated for
        each :math:`X` value.

        .. math::

            \begin{align}
            H(j) &= \sum w Y \\
            E(j) &= \sum w Y^2 \\
            W(j) &= \sum w \\
            h(j) &= H(j) / W(j) \\
            s(j) &= \sqrt{E(j) / W(j) - h(j)^2} \\
            e(j) &= s(j) / \sqrt{W(j)}
            \end{align}

        Here, :math:`w` are the counts of bin `j`.

        Examples
        --------

        .. plot:: _examples/histogram/profile_x.py
            :include-source:

        """
        means = np.zeros(self.H.shape[0])
        errors = np.zeros(self.H.shape[0])
        for idx_col in range(self.H.shape[0]):
            col = self.H[idx_col]
            mean, error = self.__get_profile(col, self.ycenters, option=option)
            means[idx_col] = mean
            errors[idx_col] = error
        return means, errors

    @property
    def profile_along_x(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Alias for :meth:`get_profile_along_x(option="") <.Hist2d.get_profile_along_x>`
        """
        return self.get_profile_along_x(option="")

    def get_profile_along_y(
        self,
        option: Literal["", "s", "i", "g"] = ""
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Get the y-profile.

        Calculate the mean value and its error per row. Which type of error
        is controled with *option*.

        Parameters
        ----------
        option : :code:`""`, :code:`"s"`, :code:`"i"`, or :code:`"g"`, default :code:`""`
            Control type of errors, see *Notes*.

            - :code:`option == ""`:  Error of the mean of all Y values
            - :code:`option == "s"`: Standard deviation of all Y
            - :code:`option == "i"`: See *Notes*
            - :code:`option == "g"`: Error of a weighted mean for combining
              measurements with variances of :math:`w`.


        Returns
        -------
        h : ndarray

            Mean values of each X (i.e., row)

        errors : ndarray

            Errors of the mean values. Depends on *option*.

        Notes
        -----
        See `TProfile <https://root.cern.ch/doc/master/classTProfile.html>`_
        of the ROOT Data Analysis Framework.

        For a histogram :math:`X` vs :math:`Y`, the following is calculated for
        each :math:`Y` value.

        .. math::

            \begin{align}
            H(j) &= \sum w X \\
            E(j) &= \sum w X^2 \\
            W(j) &= \sum w \\
            h(j) &= H(j) / W(j) \\
            s(j) &= \sqrt{E(j) / W(j) - h(j)^2} \\
            e(j) &= s(j) / \sqrt{W(j)}
            \end{align}

        Here, :math:`w` are the counts of bin `j`.

        Examples
        --------

        .. plot:: _examples/histogram/profile_y.py
            :include-source:

        """
        means = np.zeros(self.H.shape[1])
        errors = np.zeros(self.H.shape[1])
        for idx_row in range(self.H.shape[1]):
            row = self.H[:, idx_row]
            mean, error = self.__get_profile(row, self.xcenters, option=option)
            means[idx_row] = mean
            errors[idx_row] = error
        return means, errors

    @property
    def profile_along_y(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Alias for :meth:`get_profile_along_y(option="") <.Hist2d.get_profile_along_y>`
        """
        return self.get_profile_along_y(option="")

    @property
    def without_zeros(
        self
    ) -> "Hist2d":
        """
        Replace zeros with :code:`None`, removing them from colormaps

        Examples
        --------

        .. plot:: _examples/histogram/without_zeros.py
            :include-source:

        """
        _H = self.H.copy()
        _H[_H == 0] = None
        return Hist2d(_H, self.xedges, self.yedges)

    def save_to_file(
        self,
        fname: str,
        **kwargs
    ) -> None:
        """
        Save the histogram to a file.

        The first column in the file will be y, the second x, the third z.
        (this is chosen as such because the the standard hist2ascii-macro of 
        our group outputs this order)

        Parameters
        ----------
        fname : str
            Filename, including filetype
        **kwargs
            keyword arguments for :func:`numpy.savetxt`
        """
        apio.save_2d_as_txt(
            self.H, self.xedges, self.yedges, fname, **kwargs)

    @property
    def column_normalized_to_sum(self) -> "Hist2d":
        """ Normalize each column to their sum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the columns are normalized.
        """
        return Hist2d(
            self.H / self.H.sum(axis=1, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def column_normalized_to_max(self) -> "Hist2d":
        """ Normalize each column to their maximum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the columns are normalized.
        """
        return Hist2d(
            self.H / self.H.max(axis=1, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def row_normalized_to_sum(self) -> "Hist2d":
        """ Normalize each row to their sum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the rows are normalized.
        """
        return Hist2d(
            self.H / self.H.sum(axis=0, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def row_normalized_to_max(self) -> "Hist2d":
        """ Normalize each row to their maximum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the rows are normalized.
        """
        return Hist2d(
            self.H / self.H.max(axis=0, keepdims=True),
            self.xedges,
            self.yedges)


if __name__ == "__main__":
    print("achwas")

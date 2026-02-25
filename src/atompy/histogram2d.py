from os import PathLike
import copy

import numpy as np
from numpy.typing import NDArray, ArrayLike

from typing import Any, Literal, Self

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colorbar import Colorbar
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import uproot

from ._core import (
    raise_unmatching_edges,
    get_topmost_figure,
    deprecated_keyword_doing_nothing_msg,
)
from .utils import (
    centers_to_edges,
    edges_to_centers,
    get_all_dividers,
    for_pcolormesh_from_txt,
)
from .histogram1d import Hist1d


class Hist2d:
    """
    A histogram class providing basic 2D-histogram methods.

    .. tip

        Histogram your data using :func:`numpy.histogram`, then wrap the results
        in :class:`.Hist2d`::

            hist = ap.Hist2d(*np.histogram2d(data))

    Parameters
    ----------
    values : array_like, shape (m, n)
        The values of the histogram given as a 2D-array, where
        ``values[m, n]`` corresponds to the bin whose edges are given by
        [``xedges[m]``, ``xedges[m+1]``] and [``yedges[n]``, ``yedges[n+1]``].

    xedges : array_like, shape (m+1,)
        The x-edges of the histogram.

    yedges : array_like, shape (n+1,)
        The y-edges of the histogram.

    Attributes
    ----------
    values : ndarray, shape(m, n)

    H : ndarray, shape(m, n)

    xedges : ndarray, shape(m+1,)

    yedges : ndarray, shape(n+1,)

    xcenters : ndarray, shape(m,)

    ycenters : ndarray, shape(n,)

    xbins : int

    ybins : int

    nbins : (int, int)

    xlim : (float, float)

    ylim : (float, float)

    limits : ((float, float), (float, float))
    """

    def __init__(
        self,
        values: ArrayLike,
        xedges: ArrayLike,
        yedges: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
    ):
        self._values = np.asarray(values).astype(np.float64)
        self._xedges = np.asarray(xedges).astype(np.float64)
        self._yedges = np.asarray(yedges).astype(np.float64)
        if len(self._xedges) - 1 != self._values.shape[0]:
            raise ValueError("xedges and values don't match")
        if len(self._yedges) - 1 != self._values.shape[1]:
            raise ValueError("yedges and values don't match")
        self._xcenters = edges_to_centers(self._xedges)
        self._ycenters = edges_to_centers(self._yedges)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

    @classmethod
    def from_centers(
        cls,
        xcenters: ArrayLike,
        ycenters: ArrayLike,
        values: ArrayLike,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
    ) -> Self:
        """
        Initiate a :class:`.Hist2d` from centers (rather than edges).

        Parameters
        ----------
        xcenters : ArrayLike
            The x-coordinates of the bin centers.

        ycenters : ArrayLike
            The y-coordinates of the bin centers.

        values : ArrayLike
            The values associated with each bin.

        xmin/xmax/ymin/ymax : float, optional
            The lower (upper) x (y) bound to use when converting centers to edges.
            Either the lower or the upper bound must be provided if the bins have
            unequal size. Unnecessary for equally sized bins.

        Returns
        -------
        Hist2d
            A new :class:`.Hist2d` instance.

        Examples
        --------
        >>> xcenters = [0.5, 1.5, 2.5]
        >>> ycenters = [10.0, 20.0]
        >>> values = [[1, 2], [3, 4], [5, 6]]
        >>> hist = Hist2d.from_centers(xcenters, ycenters, values)
        >>> print(hist.xedges)
        [0., 1., 2., 3.]
        >>> print(hist.yedges)
        [5., 15., 25.]
        >>> print(hist.values)
        [[1, 2], [3, 4], [5, 6]]

        >>> xcenters = [0.5, 1.0, 2.0]
        >>> ycenters = [10.0, 20.0]
        >>> values = [[1, 2], [3, 4], [5, 6]]
        >>> hist = Hist2d.from_centers(xcenters, ycenters, values, xmin=0.0)
        >>> print(hist.xedges)
        [0.0, 0.75, 1.5, 2.5]
        """
        xedges = centers_to_edges(xcenters, lower=xmin, upper=xmax)
        yedges = centers_to_edges(ycenters, lower=ymin, upper=ymax)
        return cls(values, xedges, yedges, title, xlabel, ylabel, zlabel)

    @classmethod
    def from_txt(
        cls,
        fname: str | PathLike,
        *,
        iteration_order: Literal["x_first", "y_first"] = "x_first",
        data_layout: Literal["rows", "columns"] = "columns",
        xyz_indices: tuple[int, int, int] = (0, 1, 2),
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        **loadtxt_kwargs,
    ) -> Self:
        """
        Initiate a :class:`.Hist2d` from a text file.

        Parameters
        ----------
        fname : str | PathLike
            The path to the text file.

        iteration_order : {"x_first", "y_first"}, default: "x_first"
            The order in which the data iterates through the x and y dimensions.

        data_layout : {"rows", "columns"}, default: "columns"
            The layout of the data in the file, either row-major or column-major.

        xyz_indices : tuple[int, int, int], default: (0, 1, 2)
            A tuple specifying the column (or row, depending on `data_layout`)
            indices for x, y, and z (values) data.

        xmin/xmax/ymin/ymax : float, optional
            The lower (upper) x (y) bound to use when converting centers to edges.
            Either the lower or the upper bound must be provided if the bins have
            unequal size. Unnecessary for equally sized bins.

        **loadtxt_kwargs
            Additional keyword arguments to pass to :meth:`numpy.loadtxt`.

        Returns
        -------
        Hist2d
            A new :class:`.Hist2d` instance.

        Examples
        --------
        Given a file named `data.txt` with the following content:

        .. code-block::

            #x  y  z
            0.5  10.0  1
            1.5  10.0  3
            2.5  10.0  5
            0.5  20.0  2
            1.5  20.0  4
            2.5  20.0  6

        >>> # Create a dummy data.txt file for demonstration
        >>> with open("data.txt", "w") as f:
        ...     f.write("#x  y  z\\n")
        ...     f.write("0.5  10.0  1\\n")
        ...     f.write("1.5  10.0  3\\n")
        ...     f.write("2.5  10.0  5\\n")
        ...     f.write("0.5  20.0  2\\n")
        ...     f.write("1.5  20.0  4\\n")
        ...     f.write("2.5  20.0  6\\n")

        >>> hist = Hist2d.from_txt("data.txt")
        >>> print(hist.xedges)
        [0., 1., 2., 3.]
        >>> print(hist.yedges)
        [5., 15., 25.]
        >>> print(hist.values)
        [[1, 2], [3, 4], [5, 6]]
        """

        data = for_pcolormesh_from_txt(
            fname,
            iteration_order=iteration_order,
            data_layout=data_layout,
            xyz_indices=xyz_indices,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            **loadtxt_kwargs,
        )
        return cls(data[2].T, data[0], data[1], title, xlabel, ylabel, zlabel)

    @classmethod
    def from_root(
        cls,
        fname: str | PathLike,
        hname: str,
        title: str | Literal["__auto__"] = "__auto__",
        xlabel: str | Literal["__auto__"] = "__auto__",
        ylabel: str | Literal["__auto__"] = "__auto__",
        zlabel: str = "",
    ) -> Self:
        """
        Initiate a :class:`.Hist2d` from a `ROOT <https://root.cern/>`__ file.

        Parameters
        ----------
        fname : str | PathLike
            The path to the ROOT file.

        hname : str
            The path to the 2D histogram within the root file.

        Returns
        -------
        Hist2d
            A new :class:`.Hist2d` instance.
        """
        with uproot.open(fname) as file:  # type: ignore
            hist: Any = file[hname]
            values, xedges, yedges = hist.to_numpy()  # type: ignore
            title_: str = hist.title if title == "__auto__" else title
            xlabel_: str = (
                hist.member("fXaxis").member("fTitle")
                if xlabel == "__auto__"
                else xlabel
            )
            ylabel_: str = (
                hist.member("fYaxis").member("fTitle")
                if ylabel == "__auto__"
                else ylabel
            )
            zlabel_ = zlabel if zlabel != "__auto__" else ""
        return cls(values, xedges, yedges, title_, xlabel_, ylabel_, zlabel_)

    @property
    def values(self) -> NDArray[np.float64]:
        """
        The histogram values.
        """
        return self._values

    @values.setter
    def values(self, new_values: ArrayLike) -> None:
        new_values = np.asarray(new_values).astype(np.float64)
        if new_values.shape != self._values.shape:
            raise ValueError(
                f"{new_values.shape=}, but it must be {self._values.shape}"
            )
        self._values = new_values

    @property
    def H(self) -> NDArray[np.float64]:
        """Alias for :attr:`.Hist2d.values`"""
        return self.values

    @H.setter
    def H(self, new_H: ArrayLike) -> None:
        self.values = new_H

    @property
    def xedges(self) -> NDArray[np.float64]:
        """
        The histogram x-edges.
        """
        return self._xedges

    @xedges.setter
    def xedges(self, new_xedges: ArrayLike) -> None:
        new_xedges = np.asarray(new_xedges).astype(np.float64)
        if new_xedges.shape != self._xedges.shape:
            raise ValueError(
                f"{new_xedges.shape=}, but it must be {self._xedges.shape}"
            )
        self._xedges = new_xedges
        self._xcenters = edges_to_centers(new_xedges)

    @property
    def xcenters(self) -> NDArray[np.float64]:
        """
        The centers of the histogram x-edges.
        """
        return self._xcenters

    @property
    def yedges(self) -> NDArray[np.float64]:
        """
        The histogram y-edges.
        """
        return self._yedges

    @property
    def ycenters(self) -> NDArray[np.float64]:
        """
        The centers of the histogram y-edges.
        """
        return self._ycenters

    @yedges.setter
    def yedges(self, new_yedges: ArrayLike) -> None:
        new_yedges = np.asarray(new_yedges).astype(np.float64)
        if new_yedges.shape != self._yedges.shape:
            raise ValueError(
                f"{new_yedges.shape=}, but it must be {self._yedges.shape}"
            )
        self._yedges = new_yedges
        self._ycenters = edges_to_centers(new_yedges)

    @property
    def nbins(self) -> tuple[int, int]:
        """(nxbins, nybins)"""
        return self.xbins, self.ybins

    @property
    def xbins(self) -> int:
        """The number of x bins."""
        return len(self._xcenters)

    @property
    def ybins(self) -> int:
        """The number of y bins."""
        return len(self._ycenters)

    @property
    def xlim(self) -> tuple[float, float]:
        """The x-range of the histogram."""
        return float(np.min(self._xedges)), float(np.max(self._xedges))

    @property
    def ylim(self) -> tuple[float, float]:
        """The y-range of the histogram."""
        return float(np.min(self._yedges)), float(np.max(self._yedges))

    @property
    def limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(xlim, ylim)"""
        return self.xlim, self.ylim

    def __add__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        raise_unmatching_edges(self.xedges, other.xedges, "x")
        raise_unmatching_edges(self.yedges, other.yedges, "y")
        return type(self)(
            self.values + other.values,
            self.xedges.copy(),
            self.yedges.copy(),
            f"{self.title} + {other.title}",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def __iadd__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        self.values += other.values
        self.title = f"{self.title} + {other.title}"
        return self

    def __sub__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        raise_unmatching_edges(self.xedges, other.xedges, "x")
        raise_unmatching_edges(self.yedges, other.yedges, "y")
        return type(self)(
            self.values - other.values,
            self.xedges.copy(),
            self.yedges.copy(),
            f"{self.title} $-$ {other.title}",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def __isub__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        self.values -= other.values
        self.title = f"{self.title} $-$ {other.title}"
        return self

    def __mul__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        raise_unmatching_edges(self.xedges, other.xedges, "x")
        raise_unmatching_edges(self.yedges, other.yedges, "y")
        return type(self)(
            self.values * other.values,
            self.xedges.copy(),
            self.yedges.copy(),
            rf"{self.title} $\times$ {other.title}",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def __imul__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        self.values *= other.values
        self.title = rf"{self.title} $\times$ {other.title}"
        return self

    def __truediv__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        raise_unmatching_edges(self.xedges, other.xedges, "x")
        raise_unmatching_edges(self.yedges, other.yedges, "y")
        return type(self)(
            self.values / other.values,
            self.xedges.copy(),
            self.yedges.copy(),
            rf"{self.title} / {other.title}",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def __itruediv__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        self.values /= other.values
        self.title = rf"{self.title} / {other.title}"
        return self

    def __floordiv__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        raise_unmatching_edges(self.xedges, other.xedges, "x")
        raise_unmatching_edges(self.yedges, other.yedges, "y")
        return type(self)(
            self.values // other.values,
            self.xedges.copy(),
            self.yedges.copy(),
            rf"$\lfloor${self.title} / {other.title}$\rfloor$",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def __ifloordiv__(self, other: "Hist2d") -> Self:
        if not isinstance(other, Hist2d):
            return NotImplemented
        self.values //= other.values
        self.title = rf"$\lfloor${self.title} / {other.title}$\rfloor$"
        return self

    def __neg__(self) -> Self:
        return type(self)(
            -self.values,
            self.xedges,
            self.yedges,
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def __pos__(self) -> Self:
        return self

    def __eq__(self, other: "Hist2d") -> np.bool:
        if not isinstance(other, Hist2d):
            return NotImplemented
        equal_values = self.values == other.values
        equal_xedges = self.xedges == other.xedges
        equal_yedges = self.yedges == other.yedges
        return np.all([equal_values, equal_xedges, equal_yedges])

    def __str__(self) -> str:
        xedges_str = str(self.xedges)
        yxedges_str = str(self.yedges)
        values_str = str(self.values)
        hist_str = (
            "Hist1d with (values, x-edges, y-edges)\n"
            f"{values_str}\n{xedges_str}\n{yxedges_str}"
        )
        return hist_str

    def norm_diff(self, other: "Hist2d") -> Self:
        """
        Return the normalized difference between two histograms.

        Calculates (self - other) / (self + other).

        Parameters
        ----------
        other : :class:`Hist2d`
            The other histogram.

            Both histograms must have matching edges.

        Returns
        -------
        norm_diff : :class:`.Hist2d`
            A new histgram of the normalized difference.

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_diff.py
            :include-source:

        """
        raise_unmatching_edges(self.xedges, other.xedges, "x")
        raise_unmatching_edges(self.yedges, other.yedges, "y")
        new_xedges = self.xedges.copy()
        new_yedges = self.yedges.copy()
        new_values = (self.values - other.values) / (self.values + other.values)
        return type(self)(
            new_values,
            new_xedges,
            new_yedges,
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def rebin_x(self, fac: int) -> Self:
        """
        Rebin the histogram along the x-axis by an integer factor.

        This method reduces the number of x bins by aggregating adjacent bins
        along the x-axis. The number of x bins must be divisible by `fac`.
        The bin contents are summed, and new x-bin edges are constructed accordingly.

        Parameters
        ----------
        fac : int
            Factor by which to reduce the number of x bins. Must be a divisor of
            the current number of x bins (`self.xbins`).

        Returns
        -------
        :class:`.Hist2d`
            A new `Hist2d` instance with rebinned x-axis and updated values.

        Raises
        ------
        ValueError
            If `fac` is not a divisor of the number of x bins.

        Examples
        --------
        >>> h = Hist2d(values, xedges, yedges)
        >>> h_rebinned = h.rebin_x(2)
        >>> h_rebinned.values.shape[0] == h.values.shape[0] // 2
        True
        """
        old_n = self.xbins
        if old_n % fac != 0:
            msg = f"Invalid {fac=}. Possible factors for x-rebinning are {get_all_dividers(old_n)}"
            raise ValueError(msg)

        new_values = np.empty((self.xbins // fac, self.ybins))
        for i in range(new_values.shape[0]):
            new_values[i] = np.sum(self.values[i * fac : (i + 1) * fac], axis=0)
        new_xedges = np.empty(new_values.shape[0] + 1)
        new_xedges[-1] = self.xedges[-1]
        for i in range(new_xedges.size):
            new_xedges[i] = self.xedges[i * fac]
        return type(self)(
            new_values,
            new_xedges,
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def rebin_y(self, fac: int) -> Self:
        """
        Rebin the histogram along the y-axis by an integer factor.

        This method reduces the number of y bins by aggregating adjacent bins
        along the y-axis. The number of y bins must be divisible by `fac`.
        The bin contents are summed, and new y-bin edges are constructed accordingly.

        Parameters
        ----------
        fac : int
            Factor by which to reduce the number of y bins. Must be a divisor of
            the current number of y bins (`self.ybins`).

        Returns
        -------
        :class:`.Hist2d`
            A new `Hist2d` instance with rebinned y-axis and updated values.

        Raises
        ------
        ValueError
            If `fac` is not a divisor of the number of y bins.

        Examples
        --------
        >>> h = Hist2d(values, xedges, yedges)
        >>> h_rebinned = h.rebin_y(2)
        >>> h_rebinned.values.shape[1] == h.values.shape[1] // 2
        True
        """
        old_n = self.ybins
        if old_n % fac != 0:
            msg = f"Invalid {fac=}. Possible factors for x-rebinning are {get_all_dividers(old_n)}"
            raise ValueError(msg)

        new_values = np.empty((self.xbins, self.ybins // fac))
        for i in range(new_values.shape[1]):
            new_values[:, i] = np.sum(self.values[:, i * fac : (i + 1) * fac], axis=1)
        new_yedges = np.empty(new_values.shape[1] + 1)
        new_yedges[-1] = self.yedges[-1]
        for i in range(new_yedges.size):
            new_yedges[i] = self.yedges[i * fac]
        return type(self)(
            new_values,
            self.xedges,
            new_yedges,
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def binsizes_x(self) -> NDArray[np.float64]:
        """
        Compute the widths of the bins along the x-axis.

        Returns
        -------
        ndarray
            Array of bin widths along the x-axis, with shape `(self.xbins,)`.

        Examples
        --------
        >>> h = Hist2d(values, xedges, yedges)
        >>> xwidths = h.binsizes_x()
        >>> xwidths.shape == (h.xbins,)
        True
        """
        return np.diff(self.xedges)

    def binsizes_y(self) -> NDArray[np.float64]:
        """
        Compute the widths of the bins along the y-axis.

        Returns
        -------
        ndarray
            Array of bin widths along the y-axis, with shape `(self.ybins,)`.

        Examples
        --------
        >>> h = Hist2d(values, xedges, yedges)
        >>> ywidths = h.binsizes_y()
        >>> ywidths.shape == (h.ybins,)
        True
        """
        return np.diff(self.yedges)

    def binsizes(self) -> NDArray[np.float64]:
        """
        Compute the binsizes along x and y.

        Returns
        -------
        ndarray
            Array of x and y binsizes. Shape (`self.xbins, self.ybins`)
        """
        return np.array((self.binsizes_x, self.binsizes_y))

    def binareas(self) -> NDArray[np.float64]:
        """
        Compute the area of each 2D bin in the histogram.

        Returns
        -------
        ndarray
            2D array of shape (self.xbins, self.ybins) containing the area of each bin,
            computed as the outer product of x-bin widths and y-bin widths.

        Examples
        --------
        >>> h = Hist2d(values, xedges, yedges)
        >>> areas = h.binareas()
        >>> areas.shape == h.values.shape
        True
        """
        xwidths = self.binsizes_x()
        ywidths = self.binsizes_y()
        return np.outer(xwidths, ywidths).astype(np.float64)

    def for_pcolormesh(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Prepare the histogram data for plotting with :obj:`matplotlib.pyplot.pcolormesh`.

        Returns the bin edges and values formatted for direct use with `plt.pcolormesh`,
        which expects bin edges for the x and y axes, and a 2D array of values where
        shape = (len(yedges) - 1, len(xedges) - 1). The histogram values are transposed
        to match this expected orientation.

        Returns
        -------
        tuple of numpy.ndarray
            (xedges, yedges, values.T) where:
            - xedges : ndarray of shape (xbins + 1,)
            - yedges : ndarray of shape (ybins + 1,)
            - values.T : 2D ndarray of shape (ybins, xbins), ready for `pcolormesh`

        Examples
        --------
        >>> h = Hist2d(values, xedges, yedges)
        >>> xedges, yedges, data = h.for_pcolormesh()
        >>> plt.pcolormesh(xedges, yedges, data)
        """
        return self.xedges, self.yedges, self.values.T

    def keep(
        self,
        xlower: float = -np.inf,
        xupper: float = np.inf,
        ylower: float = -np.inf,
        yupper: float = np.inf,
        squeeze: bool = True,
        setval: float = 0.0,
    ) -> Self:
        """
        Return a new histogram with only bins within the specified (x, y) range.

        Parameters
        ----------
        xlower : float
            Lower bound of x-axis (inclusive).
        xupper : float
            Upper bound of x-axis (exclusive).
        ylower : float
            Lower bound of y-axis (inclusive).
        yupper : float
            Upper bound of y-axis (exclusive).
        squeeze : bool, optional
            If True, return a cropped histogram containing only the selected region.
            If False, retain original shape and set bins outside the range to `setval`.
        setval : float, optional
            Value to assign to bins outside the region if `squeeze=False`.

        Returns
        -------
        Hist2d
            A new histogram with the selected region either cropped or masked.

        Raises
        ------
        ValueError
            If no bins are selected and `squeeze=True`.
        """
        xmask = (self.xedges[:-1] >= xlower) & (self.xedges[1:] < xupper)
        ymask = (self.yedges[:-1] >= ylower) & (self.yedges[1:] < yupper)

        if not np.any(xmask) or not np.any(ymask):
            if squeeze:
                raise ValueError(
                    "Selected region does not overlap with any histogram bins."
                )
            else:
                new_values = np.full_like(self.values, fill_value=setval)
                return type(self)(
                    new_values,
                    self.xedges.copy(),
                    self.yedges.copy(),
                    self.title,
                    self.xlabel,
                    self.ylabel,
                    self.zlabel,
                )

        if squeeze:
            new_values = self.values[np.ix_(xmask, ymask)]
            new_xedges = self.xedges[
                np.concatenate(([False], xmask)) | np.concatenate((xmask, [False]))
            ]
            new_yedges = self.yedges[
                np.concatenate(([False], ymask)) | np.concatenate((ymask, [False]))
            ]
            return type(self)(
                new_values,
                new_xedges,
                new_yedges,
                self.title,
                self.xlabel,
                self.ylabel,
                self.zlabel,
            )
        else:
            new_values = np.full_like(self.values, fill_value=setval)
            new_values[np.ix_(xmask, ymask)] = self.values[np.ix_(xmask, ymask)]
            return type(self)(
                new_values,
                self.xedges.copy(),
                self.yedges.copy(),
                self.title,
                self.xlabel,
                self.ylabel,
                self.zlabel,
            )

    def remove(
        self,
        xlower: float | None = None,
        xupper: float | None = None,
        ylower: float | None = None,
        yupper: float | None = None,
        setval: float = 0.0,
    ) -> Self:
        """
        Return a new histogram with bins within the specified (x, y) range set to `setval`.

        Parameters
        ----------
        xlower : float or None
            Lower bound of x-axis to remove (inclusive). If None, no lower bound.
        xupper : float or None
            Upper bound of x-axis to remove (exclusive). If None, no upper bound.
        ylower : float or None
            Lower bound of y-axis to remove (inclusive). If None, no lower bound.
        yupper : float or None
            Upper bound of y-axis to remove (exclusive). If None, no upper bound.
        setval : float, optional
            Value to assign to bins within the removed region.

        Returns
        -------
        Hist2d
            A new histogram with the specified region "removed" (set to `setval`).
        """
        new_values = self.values.copy()

        # Determine the masks for the bins to be removed
        x_remove_mask = np.ones(self.xedges.shape[0] - 1, dtype=bool)
        if xlower is not None:
            x_remove_mask = x_remove_mask & (self.xedges[:-1] >= xlower)
        if xupper is not None:
            x_remove_mask = x_remove_mask & (self.xedges[1:] < xupper)

        y_remove_mask = np.ones(self.yedges.shape[0] - 1, dtype=bool)
        if ylower is not None:
            y_remove_mask = y_remove_mask & (self.yedges[:-1] >= ylower)
        if yupper is not None:
            y_remove_mask = y_remove_mask & (self.yedges[1:] < yupper)

        new_values[np.ix_(x_remove_mask, y_remove_mask)] = setval

        return type(self)(
            new_values,
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def integrate(self) -> float:
        """
        Return the integral of the histogram.

        Returns
        -------
        integral : float
            The integral, that is, counts-in-bin times area-of-bin.

        See also
        --------
        sum
        min
        max
        norm_to_integral
        """
        return float(np.sum(self.values * self.binareas()))

    def sum(self) -> float:
        """
        Return the sum of all bins of the histogram.

        Returns
        -------
        sum : float
            The sum, that is, sum(counts-in-bins)

        See also
        --------
        integrate
        min
        max
        norm_to_sum
        """
        return float(np.sum(self.values))

    def max(self) -> float:
        """
        Returns the maximum value of the histogram.

        Returns
        -------
        max : float

        See also
        --------
        integrate
        sum
        min
        norm_to_max
        """

        return float(np.amax(self.values))

    def min(self) -> float:
        """
        Returns the minimum value of the histogram.

        Returns
        -------
        min : float

        See also
        --------
        integrate
        sum
        max
        norm_to_min
        """

        return float(np.amin(self.values))

    def project_onto_x(
        self,
    ) -> Hist1d:
        """
        Project histogram onto its x-axis.

        .. note::

            If you want to only want to project within some x/y gate, use
            :meth:`.Hist2d.keep` or :meth:`.Hist2d.remove`.

        Returns
        -------
        hist1d : :class:`.Hist1d`
            A 1D histogram where the *bins* are the x-bins of the original
            2D histogram, and the *bin-values* are the projection.

        Examples
        --------

        .. plot:: _examples/histogram2d/project_onto_x.py
            :include-source:

        """
        return Hist1d(
            np.sum(self.values, axis=1),
            self.xedges.copy(),
            self.title,
            self.xlabel,
            self.zlabel,
        )

    def project_onto_y(self) -> Hist1d:
        """
        Project histogram onto its x-axis.

        .. note::

            If you want to only want to project within some x/y gate, use
            :meth:`.Hist2d.keep` or :meth:`.Hist2d.remove`.

        Returns
        -------
        hist1d : :class:`.Hist1d`
            A 1D histogram where the *bins* are the x-bins of the original
            2D histogram, and the *bin-values* are the projection.

        Examples
        --------

        .. plot:: _examples/histogram2d/project_onto_y.py
            :include-source:

        """
        return Hist1d(
            np.sum(self.values, axis=0),
            self.yedges.copy(),
            self.title,
            self.ylabel,
            self.zlabel,
        )

    def _calculate_profile(
        self,
        counts: NDArray,
        bin_centers: NDArray,
        option: Literal["", "s", "i", "g"] = "",
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
            out = 1.0 / np.sqrt(W)
        else:
            msg = f"{option=}, but it needs to be '', 's', 'i', or 'g'"
            raise ValueError(msg)
        return h, out

    def profile_along_x(
        self, option: Literal["", "s", "i", "g"] = ""
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Get the x-profile.

        Calculate the mean value and its error per column. Which type of error
        is controled with *option*.

        Parameters
        ----------
        option : `""`, `"s"`, `"i"`, or `"g"`, default `""`
            Control type of errors, see *Notes*.

            - `""`:  Error of the mean of all Y values
            - `"s"`: Standard deviation of all Y
            - `"i"`: See *Notes*
            - `"g"`: Error of a weighted mean for combining
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

            H(j) &= \sum w Y \\
            E(j) &= \sum w Y^2 \\
            W(j) &= \sum w \\
            h(j) &= H(j) / W(j) \\
            s(j) &= \sqrt{E(j) / W(j) - h(j)^2} \\
            e(j) &= s(j) / \sqrt{W(j)}

        Here, :math:`w` are the counts of bin `j`.

        Examples
        --------

        .. plot:: _examples/histogram2d/profile_x.py
            :include-source:

        """
        means = np.zeros(self.values.shape[0])
        errors = np.zeros(self.values.shape[0])
        for idx_col in range(self.values.shape[0]):
            col = self.values[idx_col]
            mean, error = self._calculate_profile(col, self.ycenters, option=option)
            means[idx_col] = mean
            errors[idx_col] = error
        return means, errors

    def profile_along_y(
        self, option: Literal["", "s", "i", "g"] = ""
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Get the y-profile.

        Calculate the mean value and its error per row. Which type of error
        is controled with *option*.

        Parameters
        ----------
        option : `""`, `"s"`, `"i"`, or `"g"`, default `""`
            Control type of errors, see *Notes*.

            - `""`:  Error of the mean of all Y values
            - `"s"`: Standard deviation of all Y
            - `"i"`: See *Notes*
            - `"g"`: Error of a weighted mean for combining
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

            H(j) &= \sum w X \\
            E(j) &= \sum w X^2 \\
            W(j) &= \sum w \\
            h(j) &= H(j) / W(j) \\
            s(j) &= \sqrt{E(j) / W(j) - h(j)^2} \\
            e(j) &= s(j) / \sqrt{W(j)}

        Here, :math:`w` are the counts of bin `j`.

        Examples
        --------

        .. plot:: _examples/histogram2d/profile_y.py
            :include-source:

        """
        means = np.zeros(self.values.shape[1])
        errors = np.zeros(self.values.shape[1])
        for idx_row in range(self.values.shape[1]):
            row = self.values[:, idx_row]
            mean, error = self._calculate_profile(row, self.xcenters, option=option)
            means[idx_row] = mean
            errors[idx_row] = error
        return means, errors

    def remove_zeros(self) -> Self:
        """
        Replace zeros with `NaN`, removing them from colormaps.

        Examples
        --------

        .. plot:: _examples/histogram2d/remove_zeros.py
            :include-source:

        """
        no_zeros = self.values.copy()
        no_zeros[no_zeros == 0.0] = np.nan
        return type(self)(
            no_zeros,
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_to_sum(self) -> Self:
        """
        Normalize histogram to it's maximum.

        Returns
        -------
        normalized_hist2d : :class:`.Hist2d`
            A new histogram where each bin value is divided by the histogram's sum.

        See also
        --------
        norm_to_integral
        norm_to_max

        Examples
        --------

        .. plot:: _examples/histogram2d/normalize.py
            :include-source:
        """
        return type(self)(
            self.values / np.sum(self.values),
            self.xedges.copy(),
            self.yedges.copy(),
            f"{self.title} / sum",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_to_integral(self) -> Self:
        """
        Normalize histogram to it's integral.

        The integral is calculated as the sum of all bins weighted by their area.

        Returns
        -------
        normalized_hist2d : :class:`.Hist2d`
            The normalized histogram.

        See also
        --------
        norm_to_max
        norm_to_sum

        Examples
        --------

        .. plot:: _examples/histogram2d/normalize.py
            :include-source:
        """
        return type(self)(
            self.values / self.integrate(),
            self.xedges.copy(),
            self.yedges.copy(),
            f"{self.title} / integral",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_to_max(self) -> Self:
        """
        Normalize histogram to it's maximum.

        Returns
        -------
        normalized_hist2d : :class:`.Hist2d`
            A new histogram where each bin value is divided by the histogram's maximum.

        See also
        --------
        norm_to_integral
        norm_to_sum

        Examples
        --------

        .. plot:: _examples/histogram2d/normalize.py
            :include-source:
        """
        return type(self)(
            self.values / np.amax(self.values),
            self.xedges.copy(),
            self.yedges.copy(),
            f"{self.title} / max",
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_col_to_sum(self) -> Self:
        """Normalize each column to their sum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the columns are normalized.

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_col_to_sum.py
            :include-source:

        """
        return type(self)(
            self.values / self.values.sum(axis=1, keepdims=True),
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_to_xbins(self) -> Self:
        """
        Normalize values to the number of x bins.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where all values are divided by the number of x bins.

        See also
        --------
        norm_to_ybins

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_to_nbins.py
            :include-source:

        """
        return type(self)(
            self.values / self.xbins,
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_to_ybins(self) -> Self:
        """
        Normalize values to the number of y bins.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where all values are divided by the number of y bins.

        See also
        --------
        norm_to_xbins

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_to_nbins.py
            :include-source:

        """
        return type(self)(
            self.values / self.ybins,
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_col_to_integral(self) -> Self:
        """Normalize each column to their integral.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the columns are normalized.

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_col_to_integral.py
            :include-source:

        """
        sums = self.values.sum(axis=1, keepdims=True)
        areas = self.binsizes_x() * np.diff(self.ylim)
        integrals = sums * areas
        return type(self)(
            self.values / integrals,
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_col_to_max(self) -> Self:
        """Normalize each column to their maximum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the columns are normalized.

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_col_to_max.py
            :include-source:

        """
        return type(self)(
            self.values / self.values.max(axis=1, keepdims=True),
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_row_to_sum(self) -> Self:
        """Normalize each row to their sum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the row are normalized.

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_row_to_sum.py
            :include-source:

        """
        return type(self)(
            self.values / self.values.sum(axis=0, keepdims=True),
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_row_to_integral(self) -> Self:
        """Normalize each row to their integral.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the rows are normalized.

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_row_to_integral.py
            :include-source:

        """
        sums = self.values.sum(axis=0, keepdims=True)
        areas = self.binsizes_y() * np.diff(self.xlim)
        integrals = sums * areas
        return type(self)(
            self.values / integrals,
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def norm_row_to_max(self) -> Self:
        """Normalize each row to their maximum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the rows are normalized.

        Examples
        --------

        .. plot:: _examples/histogram2d/norm_row_to_max.py
            :include-source:

        """
        return type(self)(
            self.values / self.values.max(axis=0, keepdims=True),
            self.xedges.copy(),
            self.yedges.copy(),
            self.title,
            self.xlabel,
            self.ylabel,
            self.zlabel,
        )

    def save_to_file(self, fname: str, **savetxt_kwargs) -> None:
        """
        Save the histogram to a file.

        The first column in the file will be x, the second y, the third z.

        Parameters
        ----------
        fname : str
            Filename, including filetype.

        **savetxt_kwargs
            :func:`numpy.savetxt` keyword arguments. Useful to, e.g., set a header
            with the ``header`` keyword.
        """
        xbinsizes = np.diff(self.xedges, 1)
        ybinsizes = np.diff(self.yedges, 1)
        xbincenters = self.xedges[:-1] + xbinsizes / 2.0
        ybincenters = self.yedges[:-1] + ybinsizes / 2.0
        nx = xbincenters.shape[0]
        ny = ybincenters.shape[0]

        out = np.zeros((nx * ny, 3))
        for ix, x in enumerate(xbincenters):
            for iy, y in enumerate(ybincenters):
                out[ix + iy * nx, 0] = x
                out[ix + iy * nx, 1] = y
                out[ix + iy * nx, 2] = self.values[ix, iy]
        savetxt_kwargs.setdefault("delimiter", "\t")
        xlabel = "x" if self.xlabel == "" else self.xlabel
        ylabel = "y" if self.ylabel == "" else self.ylabel
        zlabel = "values" if self.zlabel == "" else self.zlabel
        savetxt_kwargs.setdefault("header", f"{xlabel}\t{ylabel}\t{zlabel}")
        np.savetxt(fname, out, **savetxt_kwargs)

    def plot(
        self,
        ax: Axes | None = None,
        fname: str | None = None,
        xlabel: str | Literal["__auto__"] = "__auto__",
        ylabel: str | Literal["__auto__"] = "__auto__",
        zlabel: str | Literal["__auto__"] = "__auto__",
        title: str | Literal["__auto__"] = "__auto__",
        logscale: bool = False,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        colorbar_kwargs: dict[str, Any] = {},
        savefig_kwargs: dict[str, Any] = {},
        use_fixed_layout: None = None,
        fixed_layout_kwargs: None = None,
        make_me_nice: None = None,
        make_me_nice_kwargs: None = None,
        **pcolormesh_kwargs,
    ) -> tuple[Figure, Axes, Colorbar]:
        """
        Plot the 2D histogram using :obj:`matplotlib.pyplot.pcolormesh`.

        Parameters
        ----------
        fname : str, optional
            If provided, the plot will be saved to this file.

        xlabel : str, default "__auto__"
            Label for the x-axis.

            If "__auto__", use `Hist2d.xlabel`.

        ylabel : str, default "__auto__"
            Label for the y-axis.

            If "__auto__", use `Hist2d.ylabel`.

        zlabel : str, default "__auto__"
            Label for the colorbar (z-axis).

            If "__auto__", use `Hist2d.zlabel`.

        title : str, default "__auto__"
            Title of the plot.

            If "__auto__", use `Hist2d.title`.

        logscale : bool, optional
            If True, use a logarithmic color scale.

        xlim : tuple[float, float], optional
            Limits for the x-axis.

        ylim : tuple[float, float], optional
            Limits for the y-axis.

        colorbar_kwargs: dict, optional
            Additional keyword arguments passed to
            :meth:`~matplotlib.figure.Figure.add_colorbar`.

        savefig_kwargs : dict, optional
            Additional keyword arguments passed to
            :meth:`~matplotlib.figure.Figure.savefig`.

        Other parameters
        ----------------
        pcolormesh_kwargs : dict, optional
            Additional keyword arguments passed to :obj:`~matplotlib.pyplot.pcolormesh`.


        use_fixed_layout
            .. version-deprecated:: 5.4.2

                Does nothing.

        fixed_layout_kwargs
            .. version-deprecated:: 5.4.2

                Does nothing.

        make_me_nice
            .. version-deprecated:: 5.4.2

                Does nothing.

        make_me_nice_kwargs
            .. version-deprecated:: 5.4.2

                Does nothing.

        Returns
        -------
        tuple of Figure, Axes, Colorbar
            A tuple containing the matplotlib Figure, Axes, and Colorbar
            objects.

        Examples
        --------

        .. plot:: _examples/histogram2d/plot.py
            :include-source:

        .. plot:: _examples/histogram2d/plot_in_axes.py
            :include-source:

        """
        if make_me_nice is not None:
            warnings.warn(
                deprecated_keyword_doing_nothing_msg("make_me_nice"),
                DeprecationWarning,
                stacklevel=2,
            )
        if make_me_nice_kwargs is not None:
            warnings.warn(
                deprecated_keyword_doing_nothing_msg("make_me_nice_kwargs"),
                DeprecationWarning,
                stacklevel=2,
            )
        if use_fixed_layout is not None:
            warnings.warn(
                deprecated_keyword_doing_nothing_msg("use_fixed_layout"),
                DeprecationWarning,
                stacklevel=2,
            )
        if fixed_layout_kwargs is not None:
            warnings.warn(
                deprecated_keyword_doing_nothing_msg("fixed_layout_kwargs"),
                DeprecationWarning,
                stacklevel=2,
            )

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = get_topmost_figure(ax)
        norm = LogNorm() if logscale else None
        pcolormesh_kwargs_ = pcolormesh_kwargs.copy()
        pcolormesh_kwargs_.setdefault("norm", norm)
        pcolormesh_kwargs_.setdefault("rasterized", True)
        im = ax.pcolormesh(*self.for_pcolormesh(), **pcolormesh_kwargs_)

        cbar_kwargs = colorbar_kwargs.copy()
        cbar_kwargs.setdefault("use_gridspec", False)
        cb = fig.colorbar(im, ax=ax, **cbar_kwargs)
        cb.set_label(zlabel if zlabel != "__auto__" else self.zlabel)

        title_ = title if title != "__auto__" else self.title
        fig.canvas.manager.set_window_title(title_)  # type: ignore
        ax.set_title(title_)

        ax.set_xlabel(xlabel if xlabel != "__auto__" else self.xlabel)
        ax.set_ylabel(ylabel if ylabel != "__auto__" else self.ylabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if fname is not None:
            fig.savefig(fname, **savefig_kwargs)

        return fig, ax, cb

    def copy(self) -> Self:
        """Return a copy of the histogram."""
        return type(self)(
            self.values.copy(),
            self.xedges.copy(),
            self.yedges.copy(),
            copy.copy(self.title),
            copy.copy(self.xlabel),
            copy.copy(self.ylabel),
        )

import copy

from typing import Any, Iterator, Literal, Self
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import uproot

from os import PathLike
import warnings

import numpy as np
from numpy.typing import NDArray, ArrayLike

from ._core import (
    raise_unmatching_edges,
    get_topmost_figure,
    deprecated_keyword_doing_nothing_msg,
)

from .utils import get_all_dividers, centers_to_edges


class Hist1d:
    """
    A histogram class providing basic histogram methods.

    .. tip::

        Histogram your data using :func:`numpy.histogram`, then wrap the results
        in :class:`.Hist1d`::

            hist = ap.Hist1d(*np.histogram(data))

    Parameters
    ----------
    values : array_like
        The histogram values, e.g., counts.

    edges : array_like
        The edges of the histogram bins. Note that
        ``len(values) = len(edges) + 1``

        .. note::

            If you want to initialize a :class:`.Hist1d` from centers instead of edges,
            use :meth:`.Hist1d.from_centers`.

    title : str, default ""
        Optional title of the histogram.

    xlabel : str, default ""
        Optional x-label of the histogram.

    ylabel : str, default ""
        Optional y-label of the histogram.

    Attributes
    ----------
    edges : ndarray

    bin_edges : ndarray

    values : ndarray

    hist : ndarray

    centers : ndarray

    limits : (float, float)

    nbins : int
    """

    def __init__(
        self,
        values: ArrayLike,
        edges: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        self._values = np.asarray(values).astype(np.float64)
        self._edges = np.asarray(edges).astype(np.float64)
        if len(self._values) != len(self._edges) - 1:
            raise ValueError("shape of values does not match shape of edges")
        self._centers = self._calculate_centers(self._edges)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    @classmethod
    def from_centers(
        cls,
        values: ArrayLike,
        centers: ArrayLike,
        lower: None | float = None,
        upper: None | float = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> Self:
        """
        Initiate a :class:`.Hist1d` instance from values and bin-centers.

        If the bins don't have constant size, at least one limit has to be
        provided, from which the edges can be determined

        .. attention::

            If `centers` are not the centers of *all* bins, or if `lower` or `upper`
            are not indeed the lower or upper edge, `from_centers` will silently
            produce nonsense.

        Parameters
        ----------
        centers : ndarray, shape(n)
            centers of the bins

        lower, uppper : float, optional
            Lower/upper limits of the range.

            At least one limit must be provided if bins don't have a constant
            size. If both lower and upper limits are provided, the lower one
            will be prioritized.

        title : str, default ""
            Optional title of the histogram.

        xlabel : str, default ""
            Optional x-label of the histogram.

        ylabel : str, default ""
            Optional y-label of the histogram.

        See also
        --------
        centers_to_edges

        Examples
        --------
        Initiate a histogram with constant bin sizes::

            >>> x = 0.5, 1.5, 2.5, 3.5, 4.5
            >>> y =   1,   2,   3,   4,   5
            >>> hist = ap.Hist1d.from_centers(x, y)
            >>> hist.edges
            [0. 1. 2. 3. 4. 5.]

        Initiate a histogram with non-constant bin sizes. Then, a lower (or upper)
        bound has to be passed::

            >>> x = 0.5, 1.5, 3.0, 4.5, 5.5
            >>> y =   1,   2,   3,   4,   5
            >>> hist = ap.Hist1d.from_centers(x, y, lower=0.0)
            >>> hist.edges
            [0. 1. 2. 4. 5. 6.]
        """
        edges = centers_to_edges(centers, lower, upper)
        values = np.asarray(values, copy=True).astype(np.float64)
        if len(values) != len(edges) - 1:
            raise ValueError("shape of values does not match shape of centers")
        return cls(values, edges, title, xlabel, ylabel)

    @classmethod
    def from_txt(
        cls,
        fname: str | PathLike,
        data_layout: Literal["rows", "columns"] = "columns",
        idx_centers: int = 0,
        idx_values: int = 1,
        lower: None | float = None,
        upper: None | float = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **loadtxt_kwargs,
    ) -> Self:
        """
        Initiate a :class:`.Hist1d` from a text file.

        Assumes that the histogram is saved as bin-centers, bin-values. The file is
        loaded using :func:`numpy.loadtxt`.

        Parameters
        ----------
        fname : str or PathLike
            The filename.

        data_layout : "rows" or "columns", default "columns"
            Specify if centers and values are saved in the text file in rows or
            in columns.

        idx_centers : int, default 0
            The index that corresponds to the histogram bin centers.

        idx_values : int, default 1
            The index that corresponds to the histogram values.

        lower, upper : float, optional
            If the histogram bins do not have equal size, at least `lower` or `upper`
            has to be provided in order to properly calculate the bin edges.

            See :func:`.centers_to_edges`.

        title : str, default ""
            Optional title of the histogram.

        xlabel : str, default ""
            Optional x-label of the histogram.

        ylabel : str, default ""
            Optional y-label of the histogram.

        Other parameters
        ----------------
        **loadtxt_kwargs
            Additional :func:`numpy.loadtxt` keyword arguments.

        Examples
        --------
        Assume a ``data.txt`` with::

            # data.txt
            0.5    1
            1.5    2
            2.5    3
            3.5    4
            4.5    5

        Initiate a histogram from it::

            >>> hist = ap.Hist1d.from_txt("data.txt")
            >>> hist.values
            array([1. 2. 3. 4. 5.])
            >>> hist.edges
            array([0. 1. 2. 3. 4. 5])

        If the bins do not have a constant binsize, e.g.::

            # data.txt
            0.5    1
            1.5    2
            3.0    3
            4.5    4
            5.5    5

        one can load it, but needs to specify either `lower` or `upper`::

            >>> hist = ap.Hist1d.from_txt("data.txt", lower=0.0)
            >>> hist.values
            array([1. 2. 3. 4. 5.])
            >>> hist.edges
            array([0. 1. 2. 4. 5. 6])

        If multiple datasets are within one textfile, e.g.::

            # manydata.txt
            # values1   values2   centers
              1         11        0.5
              2         12        1.5
              3         13        2.5
              4         14        3.5
              5         15        4.5

        one can load specify which data to load using the `idx_*` keywords::

            >>> hist1 = ap.Hist1d.from_txt("manydata.txt", idx_centers=2, idx_values=0)
            >>> hist2 = ap.Hist1d.from_txt("manydata.txt", idx_centers=2, idx_values=1)
            >>> hist1.values
            array([1. 2. 3. 4. 5.])
            >>> hist2.values
            array([11. 12. 13. 14. 15.])
        """
        data = np.loadtxt(fname, **loadtxt_kwargs)
        if data_layout == "columns":
            data = data.T
        elif data_layout != "rows":
            raise ValueError(f"{data_layout=}, but it should be 'rows' or 'columns'")
        return cls.from_centers(
            data[idx_values], data[idx_centers], lower, upper, title, xlabel, ylabel
        )

    @classmethod
    def from_root(
        cls,
        fname: str | PathLike,
        hname: str,
        title: str | Literal["__auto__"] = "__auto__",
        xlabel: str | Literal["__auto__"] = "__auto__",
        ylabel: str | Literal["__auto__"] = "__auto__",
    ) -> Self:
        """
        Initiate a :class:`.Hist1d` from a `ROOT <https://root.cern.ch/>`__ file.

        Parameters
        ----------
        fname : str or PathLike
            The filename of the ROOT file, e.g., ``important_data.root``

        hname : str
            The name of the histogram within the ROOT file,
            e.g., ``path/to/histogram1d``.
        """
        with uproot.open(fname) as file:  # type: ignore
            hist: Any = file[hname]
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
            values, edges = hist.to_numpy()
            return cls(values, edges, title_, xlabel_, ylabel_)

    @staticmethod
    def _calculate_centers(edges: NDArray[np.float64]) -> NDArray[np.float64]:
        return edges[:-1] + 0.5 * np.diff(edges).astype(np.float64)

    @property
    def values(self) -> NDArray[np.float64]:
        """Histogram's values (e.g., counts)."""
        return self._values

    @values.setter
    def values(self, values: ArrayLike) -> None:
        values = np.asarray(values, copy=True).astype(np.float64)
        if len(values) != len(self.values):
            raise ValueError(f"shape of new values does not match shape of old values")
        self._values = values

    @property
    def hist(self) -> NDArray[np.float64]:
        """Alias for :attr:`.Hist1d.values`"""
        return self.values

    @hist.setter
    def hist(self, new_hist: ArrayLike) -> None:
        self.values = new_hist

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        """Alias for :attr:`.Hist1d.edges`"""
        return self.edges

    @bin_edges.setter
    def bin_edges(self, new_edges: ArrayLike) -> None:
        self.edges = new_edges

    @property
    def edges(self) -> NDArray[np.float64]:
        """Edges of the histogram's bins."""
        return self._edges

    @edges.setter
    def edges(self, edges: ArrayLike) -> None:
        edges = np.asarray(edges, copy=True).astype(np.float64)
        if len(edges) != len(self.edges):
            raise ValueError(f"shape of new edges does not match shape of old edges")
        self._edges = edges
        self._centers = self._calculate_centers(self._edges)

    @property
    def centers(self) -> NDArray[np.float64]:
        """Centers of the histogram's bins."""
        return self._centers

    @property
    def nbins(self) -> int:
        """Number of bins."""
        return len(self.values)

    @property
    def limits(self) -> tuple[float, float]:
        """Limits of the histogram's edges."""
        return float(self.edges[0]), float(self.edges[-1])

    def __add__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        raise_unmatching_edges(self.edges, other.edges)
        return type(self)(
            self.values + other.values,
            self.edges.copy(),
            f"{self.title} + {other.title}",
            self.xlabel,
            self.ylabel,
        )

    def __iadd__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        self.title = f"{self.title} + {other.title}"
        self.values += other.values
        return self

    def __sub__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        raise_unmatching_edges(self.edges, other.edges)
        return type(self)(
            self.values - other.values,
            self.edges.copy(),
            f"{self.title} $-$ {other.title}",
            self.xlabel,
            self.ylabel,
        )

    def __isub__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        self.values -= other.values
        self.title = f"{self.title} $-$ {other.title}"
        return self

    def __mul__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        raise_unmatching_edges(self.edges, other.edges)
        return type(self)(
            self.values * other.values,
            self.edges.copy(),
            rf"{self.title} $\times$ {other.title}",
            self.xlabel,
            self.ylabel,
        )

    def __imul__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        self.values *= other.values
        self.title = rf"{self.title} $\times$ {other.title}"
        return self

    def __truediv__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        raise_unmatching_edges(self.edges, other.edges)
        return type(self)(
            self.values / other.values,
            self.edges.copy(),
            f"{self.title} / {other.title}",
            self.xlabel,
            self.ylabel,
        )

    def __itruediv__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        self.values /= other.values
        self.title = f"{self.title} / {other.title}"
        return self

    def __floordiv__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        raise_unmatching_edges(self.edges, other.edges)
        return type(self)(
            self.values // other.values,
            self.edges.copy(),
            rf"$\lfloor${self.title} / {other.title}$\rfloor$",
            self.xlabel,
            self.ylabel,
        )

    def __ifloordiv__(self, other: "Hist1d") -> Self:
        if not isinstance(other, Hist1d):
            return NotImplemented
        self.values //= other.values
        self.title = rf"$\lfloor${self.title} / {other.title}$\rfloor$"
        return self

    def __neg__(self) -> Self:
        return type(self)(
            -self.values, self.edges, f"$-$ {self.title}", self.xlabel, self.ylabel
        )

    def __iter__(self) -> Iterator[NDArray[Any]]:
        return iter([self.values, self.centers])

    def __str__(self) -> str:
        edges_str = str(self.edges)
        values_str = str(self.values)
        hist_str = f"Hist1d with (values, xedges, yedges) =\n{values_str}\n{edges_str}"
        return hist_str

    def convert_cosine_to_angles(self, full_range: bool = True) -> Self:
        """
        Convert edges which represent `cosine(angle)` to `angle`.

        Edges must not exceed the interval [-1, 1].

        Parameters
        ----------
        full_range : bool, default True
            Control the output range.

            - If False, return a histogram ranging from 0 to π.
            - If True, return a histogram ranging from 0 to 2π, where the second half
              is a mirror image of the first.

        Raises
        ------
        ValueError
            Raised if the edges do not represent cosine values.

        Returns
        -------
        hist : :class:`.Hist1d`

        See also
        --------
        convert_cosine_to_angles
        """
        cosines = np.flip(self.edges)
        values = np.flip(self.values)

        angles = np.arccos(cosines)

        if full_range:
            angles = np.append(angles, angles[1:] + np.pi)
            values = np.append(values, np.flip(values))

        return type(self)(values, angles, self.title, self.xlabel, self.ylabel)

    def rebin(self, factor: int) -> Self:
        """
        Rebin histogram.

        Parameters
        ----------
        factor : int
            This is how many old bins will be combined to a new bin.
            Number of old bins divided by factor must be an integer.

            .. note::

                Use :func:`.get_all_dividers` to find all possible rebin factors.

        Returns
        -------
        new_histogram : :class:`.Hist1d`
            The new, rebinned histogram

        Examples
        --------

        .. plot:: _examples/histogram1d/rebin.py
            :include-source:
        """
        old_n = self.nbins

        if old_n % factor != 0:
            raise ValueError(
                f"Invalid {factor=}. Possible factors for this histogram are {get_all_dividers(old_n)}."
            )

        new_hist = np.empty(self.values.size // factor)
        for i in range(new_hist.size):
            new_hist[i] = np.sum(self.values[i * factor : i * factor + factor])

        new_edges = np.full(new_hist.size + 1, self.edges[-1])
        for i in range(new_edges.size - 1):
            new_edges[i] = self.edges[i * factor]

        return type(self)(new_hist, new_edges, self.title, self.xlabel, self.ylabel)

    def binsizes(self) -> NDArray[np.float64]:
        """
        Return the widths of all bins.

        Returns
        -------
        binsizes : ndarray

        Examples
        --------
        ::

            >>> ap.Hist1d((1, 2, 3), (0, 1, 2, 3).binsizes()
            [1. 1. 1.]
            >>> ap.Hist1d((1, 2, 3), (0, 1, 3, 4).binsizes()
            [1. 2. 1.]
        """
        return np.diff(self.edges)

    def integrate(self) -> float:
        """
        Return the integral of the histogram.

        The integral is calculated as bin-value * bin-width.

        Returns
        -------
        integral : float

        See also
        --------
        max
        min
        sum
        """
        return np.sum(self.values * self.binsizes())

    def sum(self) -> float:
        """
        Return the sum of the histogram's values.

        Returns
        -------
        sum : float

        See also
        --------
        integrate
        max
        min
        """
        return np.sum(self.values)

    def max(self) -> float:
        """
        Return the maximum value of the histogram.

        Returns
        -------
        max : float

        See also
        --------
        integrate
        min
        sum
        """
        return np.amax(self.values)

    def min(self) -> float:
        """
        Return the minimum value of the histogram.

        Returns
        -------
        min : float

        See also
        --------
        integrate
        max
        sum
        """
        return np.amin(self.values)

    def norm_to_integral(self) -> Self:
        """
        Return the histogram normalized to :meth:`.Hist1d.integrate`.

        Returns
        -------
        hist : :class:`Hist1d`
            The normalized histogram.

        See also
        --------
        norm_to_max
        norm_to_sum
        """
        new_values = np.divide(self.values, self.integrate()).copy()
        new_edges = self.edges.copy()
        new_title = f"{self.title} / integral"
        return type(self)(new_values, new_edges, new_title, self.xlabel, self.ylabel)

    def norm_to_max(self) -> Self:
        """
        Return the histogram normalized to :meth:`.Hist1d.max`.

        Returns
        -------
        hist : :class:`Hist1d`
            The normalized histogram.

        See also
        --------
        norm_to_integral
        norm_to_sum
        """
        new_values = np.divide(self.values, self.values.max()).copy()
        new_edges = self.edges.copy()
        new_title = f"{self.title} / max"
        return type(self)(new_values, new_edges, new_title, self.xlabel, self.ylabel)

    def norm_to_sum(self) -> Self:
        """
        Return the histogram normalized to :meth:`.Hist1d.sum`.

        .. note::

            When comparing two histograms, :meth:`.Hist1d.norm_to_integral` may be more
            appropriate!

        Returns
        -------
        hist : :class:`Hist1d`
            The normalized histogram.

        See also
        --------
        norm_to_integral
        norm_to_max
        """
        new_values = np.divide(self.values, self.sum()).copy()
        new_edges = self.edges.copy()
        new_title = f"{self.title} / sum"
        return type(self)(new_values, new_edges, new_title, self.xlabel, self.ylabel)

    def for_step(
        self, extent_to: None | float = None
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Return arrays appropriate for plotting with :obj:`plt.step <matplotlib.pyplot.step>`.

        By default, ``plt.step`` needs the right edges of a
        bin and the corresponding bin value. See the *where* keyword argument.

        .. attention ::
            Don't use anything else but ``where="pre"`` (which is the default) in
            ``plt.step``. Otherwise the histogram will be shifted.

        Parameters
        ----------
        extent_to : float, optional
            Extent the edges to this value (useful if the resulting plot should start
            at, e.g., zero).

        Returns
        -------
        right_edges : ndarray
            Right edges, that is, ``Hist1d.edges[1:]``.

        values : ndarray
            Bin values, that is, ``Hist1d.values``.

        Examples
        --------
        Plot ``hist: Hist1d``::

            plt.step(*hist.for_step())

        If ``where != "pre"``, the resulting histogram will be shifted!::

            plt.step(*hist.for_step(), where="mid") # this will have shifted bins

        See also
        --------
        for_bar
        for_plot
        """
        if extent_to is not None:
            edges = np.append(self.edges, self.edges[-1])
            values = np.concatenate([[extent_to], self.values, [extent_to]])
            return edges, values
        else:
            return self.edges, np.append(self.values[0], self.values)

    def for_plot(self) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Return arrays appropriate for plotting with :obj:`plt.plot <matplotlib.pyplot.plot>`.

        Returns
        -------
        centers : ndarray

        values : ndarray

        Examples
        --------
        Plot ``hist: Hist1d``::

            plt.plot(*hist.for_plot())

        Convinient, if you want to chain histogram operations and then plot them.
        E.g., this::

            plt.plot(hist.keep(lower, upper).rebin(2).centers,
                     hist.keep(lower, upper).rebin(2).norm_to_integral().values)

        becomes::

            plt.plot(*hist.keep(lower, upper).rebin(2).norm_to_integral().for_plot())


        See also
        --------
        for_bar
        for_step
        """
        return self.centers, self.values

    def for_bar(self) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """
        Return arrays appropriate for plotting with :obj:`plt.bar <matplotlib.pyplot.bar>`.

        .. attention::

            When using ``for_bar``, you cannot provide the ``widths`` keyword
            in :obj:`plt.bar <matplotlib.pyplot.bar>`!

        Returns
        -------
        centers : ndarray

        values : ndarray

        binwidths : ndarray

        Examples
        --------
        Plot ``hist: Hist1d``::

            plt.bar(*hist.for_bar())

        Note that you cannot provide the ``widths`` keyword when using this::

            plt.bar(*hist.for_bar(), widths=widths) # invalid!!!

        If you want to provide your own binwidths, use :meth:`.Hist1d.for_plot`
        instead::

            plt.bar(*hist.for_plot(), widths=widths)

        See also
        --------
        for_plot
        for_step
        """
        return self.centers, self.values, self.binsizes()

    def copy(self) -> Self:
        """Return a copy of the histogram."""
        return type(self)(
            self.values.copy(),
            self.edges.copy(),
            copy.copy(self.title),
            copy.copy(self.xlabel),
            copy.copy(self.ylabel),
        )

    def plot(
        self,
        ax: Axes | None = None,
        fname: str | None = None,
        xlabel: str | Literal["__auto__"] = "__auto__",
        ylabel: str | Literal["__auto__"] = "__auto__",
        title: str | Literal["__auto__"] = "__auto__",
        logscale: bool = False,
        xlim: tuple[None | float, None | float] | None = None,
        ylim: tuple[None | float, None | float] | None = None,
        plot_fmt: str | None = None,
        savefig_kwargs: dict[str, Any] = {},
        use_fixed_layout: None = None,
        fixed_layout_kwargs: None = None,
        make_me_nice: None = None,
        make_me_nice_kwargs: None = None,
        **plot_kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot the 1D histogram using :obj:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        fname : str, optional
            If provided, the plot will be saved to this file
            using :func:`matplotlib.figure.Figure.savefig`.

        xlabel : str, default "__auto__"
            Label for the x-axis.

            If "__auto__", use `Hist1d.xlabel`.

        ylabel : str, default "__auto__"
            Label for the y-axis.

            If "__auto__", use `Hist1d.ylabel`.

        title : str, default "__auto__"
            Title of the plot.

            If "__auto__", use `Hist1d.title`.

        logscale : bool, optional
            If True, use a logarithmic y scale.

        xlim : tuple[float | None, float | None], optional
            Limits for the x-axis.

        ylim : tuple[float | None, float | None], optional
            Limits for the y-axis.

        plot_fmt : str, optional
            format string for :obj:`matplotlib.pyplot.plot`.

        savefig_kwargs : dict, optional
            Additional keyword arguments passed to
            :func:`~matplotlib.figure.Figure.savefig`.

        Other parameters
        ----------------
        plot_kwargs : dict, optional
            Additional keyword arguments passed to
            :obj:`matplotlib.pyplot.plot`.

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
        tuple of :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            A tuple containing the matplotlib Figure and Axes.

        Examples
        --------

        .. plot:: _examples/histogram1d/plot.py
            :include-source:

        .. plot:: _examples/histogram1d/plot_in_axes.py
            :include-source:

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = get_topmost_figure(ax)
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
        if plot_fmt is None:
            ax.plot(*self.for_plot(), **plot_kwargs)
        else:
            ax.plot(*self.for_plot(), plot_fmt, **plot_kwargs)
        ax.set_xlabel(xlabel if xlabel != "__auto__" else self.xlabel)
        ax.set_ylabel(ylabel if ylabel != "__auto__" else self.ylabel)
        title_ = title if title != "__auto__" else self.title
        ax.set_title(title_)
        if title_ != "":
            fig.canvas.manager.set_window_title(title_)  # type: ignore
        ax.set_xlim(xlim)  # type: ignore
        ax.set_ylim(ylim)  # type: ignore
        if logscale:
            ax.set_yscale("log")
        if fname is not None:
            fig.savefig(fname, **savefig_kwargs)
        return fig, ax

    def plot_step(
        self,
        ax: Axes | None = None,
        fname: str | None = None,
        xlabel: str | Literal["__auto__"] = "__auto__",
        ylabel: str | Literal["__auto__"] = "__auto__",
        title: str | Literal["__auto__"] = "__auto__",
        logscale: bool = False,
        xlim: tuple[None | float, None | float] | None = None,
        ylim: tuple[None | float, None | float] | None = None,
        start_at: float | Literal["auto"] = 0.0,
        savefig_kwargs: dict[str, Any] = {},
        use_fixed_layout: None = None,
        fixed_layout_kwargs: None = None,
        make_me_nice: None = None,
        make_me_nice_kwargs: None = None,
        **plot_kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot the 1D histogram using :obj:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        fname : str, optional
            If provided, the plot will be saved to this file
            using :meth:`matplotlib.figure.Figure.savefig`.

        xlabel : str, default "__auto__"
            Label for the x-axis.

            If "__auto__", use `Hist1d.xlabel`.

        ylabel : str, default "__auto__"
            Label for the y-axis.

            If "__auto__", use `Hist1d.ylabel`.

        title : str, default "__auto__"
            Title of the plot.

            If "__auto__", use `Hist1d.title`.

        logscale : bool, optional
            If True, use a logarithmic y scale.

        xlim : tuple[float | None, float | None], optional
            Limits for the x-axis.

        ylim : tuple[float | None, float | None], optional
            Limits for the y-axis.

        start_at : float or "auto", default 0.0
            Value at which the steps will start and end.

            If "auto", start (end) at first (last) value of `Hist1d.values`.

        savefig_kwargs : dict, optional
            Additional keyword arguments passed to
            :func:`~matplotlib.figure.Figure.savefig`.

        Other parameters
        ----------------
        plot_kwargs : dict, optional
            Additional keyword arguments passed to :obj:`matplotlib.pyplot.plot`.

            Should not contain the `drawstyle` keyword.

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
        tuple of :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            A tuple containing the matplotlib Figure and Axes.

        Examples
        --------

        .. plot:: _examples/histogram1d/plot_step.py
            :include-source:

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = get_topmost_figure(ax)
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
        ax.set_xlabel(xlabel if xlabel != "__auto__" else self.xlabel)
        ax.set_ylabel(ylabel if ylabel != "__auto__" else self.ylabel)
        title_ = title if title != "__auto__" else self.title
        ax.set_title(title_)
        if title_ != "":
            fig.canvas.manager.set_window_title(title_)  # type: ignore
        x = np.repeat(self.edges, 2)
        y = np.empty_like(x, dtype=np.float64)
        y[0] = start_at if start_at != "auto" else self.values[0]
        y[1:-1] = np.repeat(self.values, 2)
        y[-1] = start_at if start_at != "auto" else self.values[-1]
        ax.plot(x, y, "-", **plot_kwargs)
        ax.set_xlim(xlim)  # type: ignore
        ax.set_ylim(ylim)  # type: ignore
        if logscale:
            ax.set_yscale("log")
        if fname is not None:
            fig.savefig(fname, **savefig_kwargs)
        return fig, ax

    def keep(
        self,
        lower: float,
        upper: float,
        squeeze: bool = False,
        setval: float = 0.0,
    ) -> Self:
        """
        Keep every entry of the histogram in-between `lower` and `upper`

        Parameters
        ----------
        lower : float
            Keep all data where the left :attr:`edges <Hist1d.edges>` are greater
            or equal to `lower`.

        upper : float
            Keep all data where the right :attr:`edges <Hist1d.edges>` are lesser
            or equal to `upper`.

        squeeze : bool, default False
            Controls if the resulting :class:`.Hist1d` has the same number of bins
            as the original histogram.

        setval : float, default 0.0
            If `squeeze` is False, fill removed data with `setval`.

        Returns
        -------
        hist : :class:`.Hist1d`

        See also
        --------
        remove

        Examples
        --------
        Create a histogram::

            >>> hist = ap.Hist1d([1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])

        Only keep values within the interval [1, 4]::

            >>> hist.keep(1, 4).values
            [0. 2. 3. 4. 0.]

        Fill removed values with a :data:`numpy.nan`

            >>> hist.keep(1, 4, setval=np.nan).values
            [nan 2. 3. 4. nan]

        Squeeze length to only include kept values::

            >>> hist.keep(1, 4, squeeze=True).edges
            [1. 2. 3.]
            >>> hist.keep(1, 4, squeeze=True).values
            [2. 3]

        Example plot:

        .. plot:: _examples/histogram1d/keep.py
            :include-source:
        """
        idx = np.flatnonzero(
            np.logical_and(lower <= self.edges[:-1], self.edges[1:] <= upper)
        )
        if squeeze:
            new_values = self.values.copy()[idx]
            new_edges = self.edges.copy()[np.append(idx, idx[-1] + 1)]
            return type(self)(
                new_values, new_edges, self.title, self.xlabel, self.ylabel
            )
        else:
            new_values = np.full(self.values.shape, setval)
            new_values[idx] = self.values[idx]
            new_edges = self.edges.copy()
            return type(self)(
                new_values, new_edges, self.title, self.xlabel, self.ylabel
            )

    def remove(
        self,
        lower: float,
        upper: float,
        setval: float = 0.0,
    ) -> Self:
        """
        Remove every entry of the histogram in-between `lower` and `upper`

        By default, the interval is [lower, upper).

        Parameters
        ----------
        lower : float
            Remove all data where the left :attr:`edges <Hist1d.edges>` are greater
            (or equal) to `lower`.

        upper : float
            Remove all data where the right :attr:`edges <Hist1d.edges>` are lesser
            (or equal) to `upper`.

        setval : float, default 0.0
            If `keepdims` is True, fill removed data with `setval`.

        Returns
        -------
        hist : :class:`.Hist1d`

        See also
        --------
        keep

        Examples
        --------
        Create a histogram::

            >>> hist = ap.Hist1d([1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])

        Only keep values in within the interval [1, 4]::

            >>> hist.remove(1, 3).values
            [1. 0. 0. 0. 5.]

        Fill removed values with a :data:`numpy.nan`

            >>> hist.keep(1, 4, setval=np.nan).values
            [1. nan nan nan 5.]

        Example plot:

        .. plot:: _examples/histogram1d/remove.py
            :include-source:
        """
        idx = np.flatnonzero(
            np.logical_and(lower <= self.edges[:-1], self.edges[1:] <= upper)
        )
        new_hist = type(self)(
            self.values.copy(), self.edges.copy(), self.title, self.xlabel, self.ylabel
        )
        new_hist.values[idx] = setval
        return new_hist

    def norm_diff(self, other: "Hist1d") -> Self:
        """
        Return the normalized difference between two histograms.

        Calculates (self - other) / (self + other).

        Parameters
        ----------
        other : :class:`.Hist1d`
            The other histogram.

            Both histograms must have matching edges.

        Returns
        -------
        norm_diff : :class:`.Hist1d`
            A new histgram of the normalized difference.

        Examples
        --------

        .. plot:: _examples/histogram1d/norm_diff.py
            :include-source:

        """
        raise_unmatching_edges(self.edges, other.edges, "x")
        new_edges = self.edges.copy()
        new_values = (self.values - other.values) / (self.values + other.values)
        return type(self)(
            new_values,
            new_edges,
            f"({self.title} $-$ {other.title}) / ({self.title} + {other.title})",
            self.xlabel,
            self.ylabel,
        )

    def pad_with(self, value: float) -> Self:
        """
        Extent histogram left and right with `value`.

        A bin is inserted before and after the histogram. The bin has the value `value`.

        Parameters
        ----------
        value : floaa

        Returns
        -------
        :class:`.Hist1d`
            A new histogram with padding.

        Examples
        --------

        .. plot:: _examples/histogram1d/pad_with.py
            :include-source:

        """
        binwidths = self.binsizes()
        new_edges = np.empty(len(self.edges) + 2, dtype=np.float64)
        new_edges[0] = self.edges[0] - binwidths[0]
        new_edges[1:-1] = self.edges
        new_edges[-1] = self.edges[-1] + binwidths[-1]
        new_values = np.empty(self.nbins + 2, dtype=np.float64)
        new_values[0] = value
        new_values[1:-1] = self.values
        new_values[-1] = value
        return type(self)(new_values, new_edges, self.title, self.xlabel, self.ylabel)

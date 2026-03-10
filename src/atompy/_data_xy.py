from typing import Iterator, Any, Literal, Self, TypedDict, Callable
from os import PathLike
import copy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray

from atompy import _core


class DataXYKwargs(TypedDict, total=True):
    title: str
    xlabel: str
    ylabel: str
    plot_kwargs: dict[str, Any]


class DataXY:
    """
    A class representing xy-data.

    Parameters
    ----------
    x : array_like
        The histogram values, e.g., counts.

    y : array_like
        The edges of the histogram bins. Note that
        ``len(values) = len(edges) + 1``

        .. note::

            If you want to initialize a :class:`.Hist1d` from centers instead of edges,
            use :meth:`.Hist1d.from_centers`.

    title : str, default ""
        Optional title of the data.

    xlabel : str, default ""
        Optional x-label of the data.

    ylabel : str, default ""
        Optional y-label of the data.

    **plot_kwargs
        Other keyword parameters will be stored in :attr:`.DataXY.plot_kwargs`,
        which is used by :meth:`.DataXY.plot`.

    Attributes
    ----------
    x : ndarray

    y : ndarray

    xy : tuple[ndarray, ndarray]

    title : str

    xlabel : str

    ylabel : str

    plot_kwargs : dict

    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **plot_kwargs,
    ) -> None:
        _x = np.asarray(x, dtype=np.float64, copy=True)
        _y = np.asarray(y, dtype=np.float64, copy=True)
        if _x.size != _y.size:
            raise ValueError("x and y values don't match")
        self._data = np.array((_x, _y))
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._plot_kwargs: dict[str, Any] = plot_kwargs

    @classmethod
    def from_txt(
        cls,
        fname: str | PathLike,
        data_layout: Literal["rows", "columns"] = "columns",
        idx_x: int = 0,
        idx_y: int = 1,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **loadtxt_kwargs,
    ) -> Self:
        """
        Initiate :class:`.DataXY` from a text file.

        The file is loaded using :func:`numpy.loadtxt`.

        Parameters
        ----------
        fname : str or PathLike
            The filename.

        data_layout : "rows" or "columns", default "columns"
            Specify if centers and values are saved in the text file in rows or
            in columns.

        idx_x : int, default 0
            The index that corresponds to the x values.

        idx_y : int, default 1
            The index that corresponds to the y values.

        title : str, default ""
            Optional title of the data.

        xlabel : str, default ""
            Optional x-label of the data.

        ylabel : str, default ""
            Optional y-label of the data.

        Other parameters
        ----------------
        **loadtxt_kwargs
            Additional keyword arguments are passed to :func:`numpy.loadtxt`.

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

            >>> data = ap.DataXY.from_txt("data.txt")
            >>> data.x
            array([1. 2. 3. 4. 5.])
            >>> data.y
            array([0. 1. 2. 3. 4. 5])

        If multiple datasets are within one textfile, e.g.::

            # manydata.txt
            # y1   y2   x
              1         11        0.5
              2         12        1.5
              3         13        2.5
              4         14        3.5
              5         15        4.5

        one can load specify which data to load using the `idx_*` keywords::

            >>> data1 = ap.DataXY.from_txt("manydata.txt", idx_x=2, idx_y=0)
            >>> data2 = ap.DataXY.from_txt("manydata.txt", idx_x=2, idx_y=1)
            >>> data1.y
            array([1. 2. 3. 4. 5.])
            >>> data2.y
            array([11. 12. 13. 14. 15.])
        """
        data = np.loadtxt(fname, **loadtxt_kwargs)
        if data_layout == "columns":
            data = data.T
        elif data_layout != "rows":
            raise ValueError(f"{data_layout=}, but it should be 'rows' or 'columns'")
        return cls(data[idx_x], data[idx_y], title=title, xlabel=xlabel, ylabel=ylabel)

    @classmethod
    def from_function(
        cls,
        f: Callable,
        x: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **fkwargs,
    ) -> Self:
        r"""
        Instantiate from a function.


        Parameters
        ----------
        f : Callable
            *y*-data will be *f*\ (*x*, \*\*\ *fkwargs*).

        x : array_like
            x-points at which *f* will be evaluated.

        title : str, default ""
            Optional title of the data.

        xlabel : str, default ""
            Optional x-label of the data.

        ylabel : str, default ""
            Optional y-label of the data.

        **fkwargs
            Other keyword arguments will be passed to *f*.

        Returns
        -------
        :class:`.DataXY`

        Examples
        --------

        ::

            >>> data = ap.DataXY.from_function(lambda x : x**2, (0, 1, 2))
            >>> data.x
            array([0., 1., 2.])
            >>> data.y
            array([0., 1., 4.])

        """
        y = f(np.asarray(x), **fkwargs)
        return cls(x, y, title=title, xlabel=xlabel, ylabel=ylabel)

    @property
    def _kwargs(self) -> DataXYKwargs:
        return {
            "title": copy.copy(self.title),
            "xlabel": copy.copy(self.xlabel),
            "ylabel": copy.copy(self.ylabel),
            "plot_kwargs": self.plot_kwargs.copy(),
        }

    @property
    def title(self) -> str:
        """
        Title of the data.

        May be used for :meth:`.DataXY.plot`.
        """
        return self._title

    @title.setter
    def title(self, val: str) -> None:
        self._title = val

    @property
    def xlabel(self) -> str:
        """
        X label of the data.

        May be used for :meth:`.DataXY.plot`.
        """
        return self._xlabel

    @xlabel.setter
    def xlabel(self, val: str) -> None:
        self._xlabel = val

    @property
    def ylabel(self) -> str:
        """
        Y label of the data.

        May be used for :meth:`.DataXY.plot`.
        """
        return self._ylabel

    @ylabel.setter
    def ylabel(self, val: str) -> None:
        self._ylabel = val

    @property
    def x(self) -> NDArray[np.float64]:
        """
        x-data of the data set.
        """
        return self._data[0]

    @x.setter
    def x(self, values: ArrayLike) -> None:
        self._data[0] = values

    @property
    def y(self) -> NDArray[np.float64]:
        """
        y-data of the data set.
        """
        return self._data[1]

    @y.setter
    def y(self, values: ArrayLike) -> None:
        self._data[1] = values

    @property
    def xy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Tuple (x, y) data.

        Use, for example, like

        .. code-block:: python

            plt.plot(*data.xy, **data.plot_kwargs)
        """
        return self.x, self.y

    @property
    def plot_kwargs(self) -> dict[str, Any]:
        """
        Keyword arguments for :func:`matplotlib.pyplot.plot`.

        Use, for example, like

        .. code-block:: python

            plt.plot(*data.xy, **data.plot_kwargs)
        """
        return self._plot_kwargs

    @plot_kwargs.setter
    def plot_kwargs(self, new_dict) -> None:
        self._plot_kwargs = new_dict.copy()

    def __len__(self) -> int:
        return self._data.shape[1]

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        return iter(self._data.T)

    def asarray(self, copy: bool = False) -> NDArray[np.float64]:
        """
        Return a 2d ndarray of the data.

        Returns
        -------
        data : ndarray, shape (2, N)
            data[0] corresponds to x, data[1] to y.
        """
        return self._data.copy() if copy else self._data

    def xmin(self) -> float:
        """
        Get the minimum x-value.
        """
        return float(self.x.min())

    def xmax(self) -> float:
        """
        Get the maximum x-value.
        """
        return float(self.x.max())

    def ymin(self) -> float:
        """
        Get the minimum y-value.
        """
        return float(self.y.min())

    def ymax(self) -> float:
        """
        Get the maximum y-value.
        """
        return float(self.y.max())

    def xlims(self) -> tuple[float, float]:
        """
        Get (xmin, xmax).
        """
        return self.xmin(), self.xmax()

    def ylims(self) -> tuple[float, float]:
        """
        Get (ymin, ymax).
        """
        return self.ymin(), self.ymax()

    def sum(self) -> float:
        """
        Calculate sum of y values.
        """
        return float(np.sum(self.y))

    def integrate(self) -> float:
        """
        Integrate data.

        Integration is done using the
        `trapezoid rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`__.

        See also
        --------
        norm_to_integral
        """
        return float(np.trapezoid(self.y, self.x))

    def norm_to_integral(self) -> Self:
        """
        Return the data normalized to :meth:`.DataXY.integrate`.

        Returns
        -------
        data : :class:`.DataXY`
            The normalized data.

        See also
        --------
        norm_to_max
        norm_to_sum
        """
        new_x = self.x.copy()
        new_y = np.divide(self.y, self.integrate()).copy()
        new_title = f"{self.title} / integral"
        return type(self)(new_x, new_y, new_title, self.xlabel, self.ylabel)

    def norm_to_max(self) -> Self:
        """
        Return the data normalized to :meth:`.DataXY.max`.

        Returns
        -------
        data : :class:`.DataXY`
            The normalized data.

        See also
        --------
        norm_to_integral
        norm_to_sum
        """
        new_x = self.x.copy()
        new_y = np.divide(self.y, self.ymax()).copy()
        new_title = f"{self.title} / max"
        return type(self)(new_x, new_y, new_title, self.xlabel, self.ylabel)

    def norm_to_sum(self) -> Self:
        """
        Return the data normalized to :meth:`.DataXY.sum`.

        Returns
        -------
        data : :class:`.DataXY`
            The normalized data.

        See also
        --------
        norm_to_integral
        norm_to_max
        """
        new_x = self.x.copy()
        new_y = np.divide(self.y, self.sum()).copy()
        new_title = f"{self.title} / sum"
        return type(self)(new_x, new_y, new_title, self.xlabel, self.ylabel)

    def copy(self) -> Self:
        """Return a copy of data."""
        return type(self)(self.x.copy(), self.y.copy(), **self._kwargs)

    def _get_gated_data(
        self, cond: NDArray[np.bool_], squeeze: bool, ysetval: float
    ) -> Self:
        if squeeze:
            new_x = self.x[cond].copy()
            new_y = self.y[cond].copy()
        else:
            new_x = self.x.copy()
            new_y = self.y.copy()
            new_y[~cond] = ysetval
        return type(self)(new_x, new_y, **self._kwargs)

    def keep_x(
        self, xmin: float, xmax: float, squeeze: bool = True, ysetval: float = 0.0
    ) -> Self:
        """
        Only keep data within [xmin, xmax).

        Parameters
        ----------
        xmin, xmax : float
            Minimum (inclusive) and maximum (exclusive) x values.

        squeeze : bool, default True
            If true, remove data outside of gate. Otherwise, keep data but set
            y values to *ysetvat*.

        ysetval : float, default 0.0
            See *squeeze*.

            Only applies if squeeze is False.
        """
        cond = (xmin <= self.x) & (self.x < xmax)
        return self._get_gated_data(cond, squeeze, ysetval)

    def keep_y(
        self, ymin: float, ymax: float, squeeze: bool = True, ysetval: float = 0.0
    ) -> Self:
        """
        Only keep data within [ymin, ymax).

        Parameters
        ----------
        ymin, ymax : float
            Minimum (inclusive) and maximum (exclusive) y values.

        squeeze : bool, default True
            If true, remove data outside of gate. Otherwise, keep data but set
            y values to *ysetvat*.

        ysetval : float, default 0.0
            See *squeeze*.

            Only applies if squeeze is False.
        """
        cond = (ymin <= self.y) & (self.y < ymax)
        return self._get_gated_data(cond, squeeze, ysetval)

    def remove_x(
        self, xmin: float, xmax: float, squeeze: bool = True, ysetval: float = 0.0
    ) -> Self:
        """
        Remove data within [xmin, xmax).

        Parameters
        ----------
        xmin, xmax : float
            Minimum (inclusive) and maximum (exclusive) x values where data is removed.

        squeeze : bool, default True
            If true, remove data outside of gate. Otherwise, keep data but set
            y values to *ysetvat*.

        ysetval : float, default 0.0
            See *squeeze*.

            Only applies if squeeze is False.
        """
        cond = (xmin <= self.x) & (self.x < xmax)
        return self._get_gated_data(~cond, squeeze, ysetval)

    def remove_y(
        self, ymin: float, ymax: float, squeeze: bool = True, ysetval: float = 0.0
    ) -> Self:
        """
        Remove data within [ymin, ymax).

        Parameters
        ----------
        ymin, ymax : float
            Minimum (inclusive) and maximum (exclusive) y values where data is removed.

        squeeze : bool, default True
            If true, remove data outside of gate. Otherwise, keep data but set
            y values to *ysetvat*.

        ysetval : float, default 0.0
            See *squeeze*.

            Only applies if squeeze is False.
        """
        cond = (ymin <= self.x) & (self.x < ymax)
        return self._get_gated_data(~cond, squeeze, ysetval)

    def plot(
        self,
        ax: Axes | None = None,
        fname: str | None = None,
        xlabel: str | Literal["__auto__"] = "__auto__",
        ylabel: str | Literal["__auto__"] = "__auto__",
        title: str | Literal["__auto__"] = "__auto__",
        logscale_x: bool = False,
        logscale_y: bool = False,
        xlim: tuple[None | float, None | float] | None = None,
        ylim: tuple[None | float, None | float] | None = None,
        plot_fmt: str | None = None,
        savefig_kwargs: dict[str, Any] = {},
        **plot_kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot the data using :meth:`matplotlib.axes.Axes.plot`.

        .. note::

            :attr:`DataXY.plot_kwargs` will also be passed to
            :meth:`~!matplotlib.axes.Axes.plot`.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            If a matplotilb axes is passed, plot data into it, else create a new axes.

        fname : str, optional
            If provided, the plot will be saved to this file
            using :meth:`matplotlib.figure.Figure.savefig`.

        xlabel : str, default "__auto__"
            Label for the x-axis.

            If "__auto__", use :attr:`.DataXY.xlabel`.

        ylabel : str, default "__auto__"
            Label for the y-axis.

            If "__auto__", use :attr:`.DataXY.ylabel`.

        title : str, default "__auto__"
            Title of the plot.

            If "__auto__", use :attr:`.DataXY.title`.

        logscale_x : bool, default False
            If True, use a logarithmic x scale.

        logscale_y : bool, default False
            If True, use a logarithmic y scale.

        xlim : tuple[float | None, float | None], optional
            Limits for the x-axis.

        ylim : tuple[float | None, float | None], optional
            Limits for the y-axis.

        plot_fmt : str, optional
            format string for :meth:`~matplotlib.axes.Axes.plot`.

        savefig_kwargs : dict, optional
            Dictionary of keyword arguments passed to
            :func:`~matplotlib.figure.Figure.savefig`.

        Other parameters
        ----------------
        plot_kwargs : dict, optional
            Additional keyword arguments are merged with
            :attr:`.DataXY.plot_kwargs` and passed to
            :obj:`matplotlib.pyplot.plot`.

        Returns
        -------
        tuple of :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            A tuple containing the matplotlib Figure and Axes.

        Examples
        --------

        .. plot:: _examples/dataxy/plot.py
            :include-source:

        """
        _plot_kwargs = self.plot_kwargs | plot_kwargs
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = _core.get_topmost_figure(ax)
        if plot_fmt is None:
            ax.plot(*self.xy, **_plot_kwargs)
        else:
            ax.plot(*self.xy, plot_fmt, **_plot_kwargs)
        ax.set_xlabel(xlabel if xlabel != "__auto__" else self.xlabel)
        ax.set_ylabel(ylabel if ylabel != "__auto__" else self.ylabel)
        title_ = title if title != "__auto__" else self.title
        ax.set_title(title_)
        if title_ != "":
            fig.canvas.manager.set_window_title(title_)  # type: ignore
        ax.set_xlim(xlim)  # type: ignore
        ax.set_ylim(ylim)  # type: ignore
        if logscale_x:
            ax.set_xscale("log")
        if logscale_y:
            ax.set_yscale("log")
        if fname is not None:
            fig.savefig(fname, **savefig_kwargs)
        return fig, ax

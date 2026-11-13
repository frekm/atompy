from collections.abc import Callable
from os import PathLike
from typing import Any, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from atompy._core import get_topmost_figure, normalize_columns_or_row_data
from atompy._plottable import Plottable2d
from atompy._utils import interpolate


class Function(Plottable2d):
    """
    A class representing a mathematical function.

    Parameters
    ----------
    f : Callable
        Function to evaluate. Must accept an array of x-values.

    xlims : tuple[float, float]
        Minimum and maximum x-values.

    steps : int, default 100
        Number of points at which the function is evaluated.

    xlog : bool, default False
        If True, use logarithmically spaced x-values.

    ylog : bool, default False
        If True, use logarithmic interpolation when applicable.

    interval : "inclusive" or "exclusive", default "inclusive"
        Specify whether the upper x-limit is included.

    title : str, default ""
        Optional title of the plot.

    xlabel : str, default ""
        Optional x-axis label.

    ylabel : str, default ""
        Optional y-axis label.

    f_kwargs : dict, optional
        Additional keyword arguments passed to `f`.

    plot_kwargs : dict, optional
        Keyword arguments stored in :attr:`.Function.plot_kwargs`,
        which are used by :attr:`.Function.plot`.
    """
    def __init__(
        self,
        f: Callable,
        xlims: tuple[float, float],
        *,
        steps: int = 100,
        xlog: bool = False,
        ylog: bool = False,
        interval: Literal["inclusive", "exclusive"] = "inclusive",
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        f_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
    ) -> None:
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        super().__init__(title, xlabel, ylabel, **plot_kwargs)
        self._f = f
        self._f_kwargs = {} if f_kwargs is None else f_kwargs
        self._xlog = xlog
        self._ylog = ylog
        self._xlims = xlims
        self._steps = steps
        if interval not in ("inclusive", "exclusive"):
            raise ValueError(f"{interval=}, but it must be 'inclusive' or 'exclusive'")
        self._endpoint = interval == "inclusive"
        self._update()

    @classmethod
    def from_xy(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        *,
        steps: int = 100,
        xlog: bool = False,
        ylog: bool = False,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        plot_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """
        Instantiate from x- and y-data.

        The function is constructed using interpolation.

        Parameters
        ----------
        x, y : array_like
            One-dimensional x- and y-values. The x-values must be
            monotonically increasing and both arrays must have the same size.

        steps : int, default 100
            Number of points at which the interpolated function is evaluated.

        xlog : bool, default False
            If True, use logarithmically spaced x-values.

        ylog : bool, default False
            If True, use logarithmic interpolation.

        title : str, default ""
            Optional title of the plot.

        xlabel : str, default ""
            Optional x-axis label.

        ylabel : str, default ""
            Optional y-axis label.

        plot_kwargs : dict, optional
            Keyword arguments stored in :attr:`.Function.plot_kwargs`.

        Returns
        -------
        :class:`.Function`
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"{x.ndim=} and {y.ndim=}, but must be one-dimensional")
        if x.size != y.size:
            raise ValueError(f"f{x.size=} != {y.size=}, but must be the same")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be monotonically increasing")

        f = interpolate(x, y, xlog, ylog)
        return cls(
            f,
            (x[0], x[-1]),
            steps=steps,
            xlog=xlog,
            ylog=ylog,
            interval="inclusive",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_kwargs=plot_kwargs,
        )

    @classmethod
    def from_txt(
        cls,
        fname: str | PathLike,
        *,
        xy_indices: tuple[int, int] = (0, 1),
        data_layout: Literal["columns", "rows"] = "columns",
        steps: int = 100,
        xlog: bool = False,
        ylog: bool = False,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        plot_kwargs: dict[str, Any] | None = None,
        **loadtxt_kwargs,
    ) -> Self:
        """
        Instantiate from a text file.

        The file is loaded using :func:`numpy.loadtxt`.

        Parameters
        ----------
        fname : str or PathLike
            Filename of the input file.

        xy_indices : tuple[int, int], default (0, 1)
            Indices corresponding to the x- and y-data.

        data_layout : "columns" or "rows", default "columns"
            Specify whether data is stored in columns or rows.

        steps : int, default 100
            Number of points at which the interpolated function is evaluated.

        xlog : bool, default False
            If True, use logarithmically spaced x-values.

        ylog : bool, default False
            If True, use logarithmic interpolation.

        title : str, default ""
            Optional title of the plot.

        xlabel : str, default ""
            Optional x-axis label.

        ylabel : str, default ""
            Optional y-axis label.

        plot_kwargs : dict, optional
            Keyword arguments stored in :attr:`.Function.plot_kwargs`.

        Other parameters
        ----------------
        **loadtxt_kwargs
            Additional keyword arguments passed to :func:`numpy.loadtxt`.

        Returns
        -------
        :class:`.Function`
        """
        data = normalize_columns_or_row_data(
            np.loadtxt(fname, **loadtxt_kwargs), data_layout
        )
        return cls.from_xy(
            *data[xy_indices],
            steps=steps,
            xlog=xlog,
            ylog=ylog,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_kwargs=plot_kwargs,
        )

    def _update(self) -> None:
        self._x = self._lin_or_logspace(self.xlog)
        self._y = self(self._x)

    def _lin_or_logspace(self, log: bool) -> NDArray[np.number]:
        if log:
            xlims = np.log10(self.xlims)
            x = np.logspace(*xlims, num=self.steps, endpoint=self._endpoint)
        else:
            x = np.linspace(*self.xlims, self.steps, endpoint=self._endpoint)
        return x

    def __call__(self, x: ArrayLike) -> NDArray[np.number]:
        """Evaluate the function at the given x-values."""
        return self._f(np.asarray(x), **self._f_kwargs)

    @property
    def steps(self) -> int:
        """Number of points at which the function is evaluated."""
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = value
        self._update()

    @property
    def dx(self) -> float:
        """Spacing between consecutive x-values."""
        return (self.xlims[1] - self.xlims[0]) / self.steps

    @property
    def xlog(self) -> bool:
        """Whether x-values are logarithmically spaced."""
        return self._xlog

    @xlog.setter
    def xlog(self, value: bool) -> None:
        if value != self.xlog:
            self._xlog = value
            self._update()

    @property
    def ylog(self) -> bool:
        """Wether y-values are in logspace."""
        return self._ylog

    @ylog.setter
    def ylog(self, value: bool) -> None:
        if value != self.ylog:
            self._ylog = value
            self._update()

    @property
    def xlims(self) -> tuple[float, float]:
        """Minimum and maximum x-values."""
        return self._xlims

    @property
    def ylims(self) -> tuple[float, float]:
        """Minimum and maximum y-values."""
        return float(np.amin(self.y)), float(np.amax(self.y))

    @property
    def x(self) -> NDArray[np.number]:
        """x-data of the function."""
        return self._x

    @property
    def y(self) -> NDArray[np.number]:
        """y-data of the function."""
        return self._y

    @property
    def xy(self) -> tuple[NDArray[np.number], NDArray[np.number]]:
        """Tuple (x, y) data."""
        return self.x, self.y

    def scaled(self, factor: float) -> Self:
        """
        Get a scaled version of the function.

        Parameters
        ----------
        factor : float
            Scale factor.

        Returns
        -------
        scaled_function : :class:`.Function`
        """
        return type(self)(
            lambda factor: self._f * factor,
            self.xlims,
            steps=self.steps,
            xlog=self.xlog,
            ylog=self.ylog,
            interval="inclusive" if self._endpoint else "exclusive",
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            f_kwargs=self._f_kwargs,
            plot_kwargs=self.plot_kwargs,
        )

    def integrate(
        self,
        xmin: float | None = None,
        xmax: float | None = None,
    ) -> float:
        """
        Integrate the function over an interval.

        Integration is done using the trapezoid rule.

        Parameters
        ----------
        xmin : float, optional
            Lower integration limit. Defaults to the minimum x-value.

        xmax : float, optional
            Upper integration limit. Defaults to the maximum x-value.

        Returns
        -------
        float
            The definite integral.
        """
        xmin = self.xlims[0] if xmin is None else xmin
        xmax = self.xlims[1] if xmax is None else xmax

        if xmin >= xmax:
            raise ValueError("xmin must be smaller than xmax")

        if xmin < self.xlims[0] or xmax > self.xlims[1]:
            raise ValueError("integration limits must lie within xlims")

        mask = (self.x > xmin) & (self.x < xmax)
        x = np.concatenate(([xmin], self.x[mask], [xmax]))
        y = self(np.asarray(x))

        return float(np.trapezoid(y, x))

    def is_monotonically_increasing(self) -> np.bool:
        """Check whether the function is strictly monotonically increasing."""
        return np.all(self.y[:-1] < self.y[1:])

    def is_monotonically_decreasing(self) -> np.bool:
        """Check whether the function is strictly monotonically decreasing."""
        return np.all(self.y[:-1] > self.y[1:])

    def is_invertible(self) -> np.bool:
        """Check whether the function is invertible."""
        return self.is_monotonically_increasing() or self.is_monotonically_decreasing()

    def inverted(
        self,
        ylims: tuple[float, float] | None = None,
        *,
        steps: int | Literal["same"] = "same",
        title: str = "__auto__",
        xlabel: str = "__auto__",
        ylabel: str = "__auto__",
        plot_kwargs: dict[str, Any] | Literal["same"] = "same",
    ) -> Self:
        """
        Return the inverse function.

        The function must be strictly monotonically increasing or decreasing.

        Parameters
        ----------
        ylims : tuple[float, float], optional
            Minimum and maximum input values of the inverse function.
            Must lie within the current y-range.

        steps : int or "same", default "same"
            Number of points at which the inverse function is evaluated.
            If "same", use :attr:`.Function.steps`.

        title : str, default "__auto__"
            Title of the inverse function. If "__auto__", use the current title.

        xlabel : str, default "__auto__"
            x-axis label. If "__auto__", use the current xlabel.

        ylabel : str, default "__auto__"
            y-axis label. If "__auto__", use the current ylabel.

        plot_kwargs : dict or "same", default "same"
            Plot keyword arguments. If "same", use an empty dictionary.

        Returns
        -------
        :class:`.Function`
            The inverse function.
        """
        if not self.is_invertible():
            raise TypeError(
                "function is not invertible as it is not monotonically increasing or decreasing"
            )

        if self.is_monotonically_increasing():
            x, y = self.xy
        else:
            x = self.x[::-1]
            y = self.y[::-1]

        current_ylims = y[0], y[-1]
        if ylims is None:
            ylims = current_ylims
        elif ylims[0] < current_ylims[0] or ylims[1] > current_ylims[1]:
            raise ValueError(f"input {ylims=} exceeds {current_ylims=}")

        f_inv = interpolate(y, x, self.ylog, self.xlog)

        steps = self.steps if steps == "same" else steps
        title = self.title if title == "__auto__" else title
        xlabel = self.xlabel if xlabel == "__auto__" else xlabel
        ylabel = self.ylabel if ylabel == "__auto__" else ylabel
        plot_kwargs = {} if plot_kwargs == "same" else plot_kwargs

        return type(self)(
            f_inv,
            ylims,
            steps=steps,
            xlog=self.ylog,
            ylog=self.xlog,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_kwargs=plot_kwargs,
        )

    def plot_default(
        self, ax: Axes | None = None, fmt="-", **kwargs
    ) -> tuple[Figure, Axes]:
        """
        Plot the function using :meth:`matplotlib.axes.Axes.plot`.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            If provided, plot into this axes; otherwise create a new figure
            and axes.

        fmt : str, default "-"
            Format string passed to :meth:`matplotlib.axes.Axes.plot`.

        **kwargs
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot`. These are merged with
            :attr:`.Function.plot_kwargs`.

        Returns
        -------
        tuple of :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            A tuple containing the matplotlib Figure and Axes.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = get_topmost_figure(ax)

        kwargs_ = self._plot_kwargs | kwargs
        ax.plot(*self.xy, fmt, **kwargs_)
        if self.xlog:
            ax.set_xscale("log")
        if self.ylog:
            ax.set_yscale("log")

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)

        return fig, ax

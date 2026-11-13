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
        return self._f(np.asarray(x), **self._f_kwargs)

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = value
        self._update()

    @property
    def dx(self) -> float:
        return (self.xlims[1] - self.xlims[0]) / self.steps

    @property
    def xlog(self) -> bool:
        return self._xlog

    @xlog.setter
    def xlog(self, value: bool) -> None:
        if value != self.xlog:
            self._xlog = value
            self._update()

    @property
    def ylog(self) -> bool:
        return self._ylog

    @ylog.setter
    def ylog(self, value: bool) -> None:
        if value != self.ylog:
            self._ylog = value
            self._update()

    @property
    def xlims(self) -> tuple[float, float]:
        return self._xlims

    @property
    def ylims(self) -> tuple[float, float]:
        return float(np.amin(self.y)), float(np.amax(self.y))

    @property
    def x(self) -> NDArray[np.number]:
        return self._x

    @property
    def y(self) -> NDArray[np.number]:
        return self._y

    @property
    def xy(self) -> tuple[NDArray[np.number], NDArray[np.number]]:
        return self.x, self.y

    def is_monotonically_increasing(self) -> np.bool:
        return np.all(self.y[:-1] < self.y[1:])

    def is_monotonically_decreasing(self) -> np.bool:
        return np.all(self.y[:-1] > self.y[1:])

    def is_invertible(self) -> np.bool:
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

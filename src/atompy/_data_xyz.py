import copy
from collections.abc import Callable
from os import PathLike
from typing import Any, Literal, Self, TypedDict

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from atompy._core import NUMBER_T


class DataXYZKwargs(TypedDict, total=True):
    title: str
    xlabel: str
    ylabel: str
    zlabel: str
    plot_kwargs: dict[str, Any]


class DataXYZ:
    """
    A class representing xyz-data.

    Parameters
    ----------
    x : array_like, shape(n)
        The x values.

    y : array_like, shape(m)
        The y values

    z : array_like, shape(n, m)
        The corresponding z values.

        `z[i,j]` corresponds to `x[i]`, `y[j]`.

    title : str, default ""
        Optional title of the data.

    xlabel : str, default ""
        Optional x-label of the data.

    ylabel : str, default ""
        Optional y-label of the data.

    zlabel : str, default ""
        Optional z-label of the data.

    **plot_kwargs
        Other keyword parameters will be stored in :attr:`.DataXY.plot_kwargs`,
        which is used by :meth:`.DataXY.plot`.

    Attributes
    ----------
    x : ndarray

    xmesh : ndarray

    y : ndarray

    ymesh : ndarray

    z : ndarray

    zmesh : ndarray

    title : str

    xlabel : str

    ylabel : str

    zlabel : str

    plot_kwargs : dict

    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        **plot_kwargs,
    ) -> None:
        _x = np.asarray(x).reshape(1)
        _y = np.asarray(y).reshape(1)
        _z = np.asarray(z)
        if _x.size != _y.size:
            raise ValueError("x and y values don't match")
        if _z.shape != (_x.size, _y.size):
            raise ValueError(
                f"shape of z {_z.shape} doesn't match x ({_x.size}) and y ({_y.size}) sizes"
            )
        xm, ym = np.meshgrid(_x, y)
        self._xmesh = xm
        self._ymesh = ym
        self._zmesh = _z
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._zlabel = zlabel
        self._plot_kwargs: dict[str, Any] = plot_kwargs

    @property
    def x(self) -> NDArray[NUMBER_T]: ...
    @x.setter
    def x(self, new_x: ArrayLike) -> None: ...
    @property
    def xmesh(self) -> NDArray[NUMBER_T]: ...

    @property
    def y(self) -> NDArray[NUMBER_T]: ...
    @y.setter
    def y(self, new_x: ArrayLike) -> None: ...
    @property
    def ymesh(self) -> NDArray[NUMBER_T]: ...

    @property
    def z(self) -> NDArray[NUMBER_T]: ...
    @z.setter
    def z(self, new_x: ArrayLike) -> None: ...
    @property
    def zmesh(self) -> NDArray[NUMBER_T]: ...

    @property
    def plot_kwargs(self) -> dict[str, Any]: ...
    @plot_kwargs.setter
    def plot_kwargs(self, new_kwargs: dict[str, Any]) -> None: ...

    @property
    def _kwargs(self) -> DataXYZKwargs:
        return {
            "title": copy.copy(self.title),
            "xlabel": copy.copy(self.xlabel),
            "ylabel": copy.copy(self.ylabel),
            "zlabel": copy.copy(self.ylabel),
            "plot_kwargs": self.plot_kwargs.copy(),
        }

    @property
    def title(self) -> str:
        """
        Title of the data.

        May be used for :meth:`.DataXYZ.plot`.
        """
        return self._title

    @title.setter
    def title(self, val: str) -> None:
        self._title = val

    @property
    def xlabel(self) -> str:
        """
        X label of the data.

        May be used for :meth:`.DataXYZ.plot`.
        """
        return self._xlabel

    @xlabel.setter
    def xlabel(self, val: str) -> None:
        self._xlabel = val

    @property
    def ylabel(self) -> str:
        """
        Y label of the data.

        May be used for :meth:`.DataXYZ.plot`.
        """
        return self._ylabel

    @ylabel.setter
    def ylabel(self, val: str) -> None:
        self._ylabel = val

    @property
    def zlabel(self) -> str:
        """
        Z label of the data.

        May be used for :meth:`.DataXYZ.plot`.
        """
        return self._ylabel

    @zlabel.setter
    def zlabel(self, val: str) -> None:
        self._zlabel = val

    @classmethod
    def from_txt(
        cls,
        fname: str | PathLike,
        xyz_idx: tuple[int, int, int] = (0, 1, 2),
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        **loadtxt_kwargs,
    ) -> Self: ...

    @classmethod
    def from_function(
        cls,
        f: Callable,
        x: ArrayLike,
        y: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        **fkwargs,
    ) -> Self: ...

    def xmin(self) -> np.number: ...
    def xmax(self) -> np.number: ...
    def xlims(self) -> tuple(np.number, np.number): ...
    def ymin(self) -> np.number: ...
    def ymax(self) -> np.number: ...
    def ylims(self) -> tuple(np.number, np.number): ...
    def zmin(self) -> np.number: ...
    def zmax(self) -> np.number: ...
    def zlims(self) -> tuple(np.number, np.number): ...

    def norm_to_integral(self) -> Self: ...
    def norm_to_max(self) -> Self: ...

    def integrate(self) -> float: ...

    def integrate_over_x(
        self, xmin: float = -np.inf, xmax: float = np.inf
    ) -> DataXY: ...

    def integrate_over_y(
        self, ymin: float = -np.inf, ymax: float = np.inf
    ) -> DataXY: ...

    def keep_x(self, xmin: float = -np.inf, xmax: float = np.inf) -> Self: ...
    def keep_y(self, ymin: float = -np.inf, ymax: float = np.inf) -> Self: ...
    def remove_x(self, xmin: float = -np.inf, xmax: float = np.inf) -> Self: ...
    def remove_y(self, ymin: float = -np.inf, ymax: float = np.inf) -> Self: ...

    def plot(
        self,
        ax: Axes | None = None,
        fname: str | None = None,
        xlabel: str | Literal["__auto__"] = "__auto__",
        ylabel: str | Literal["__auto__"] = "__auto__",
        zlabel: str | Literal["__auto__"] = "__auto__",
        title: str | Literal["__auto__"] = "__auto__",
        logscale_x: bool = False,
        logscale_y: bool = False,
        logscale_z: bool = False,
        xlim: tuple[None | float, None | float] | None = None,
        ylim: tuple[None | float, None | float] | None = None,
        zlim: tuple[None | float, None | float] | None = None,
        savefig_kwargs: dict[str, Any] | None = None,
        **plot_kwargs,
    ) -> tuple[Figure, Axes]: ...

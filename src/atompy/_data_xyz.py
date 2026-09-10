import copy
from os import PathLike
from typing import Any, Literal, Self, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from atompy._core import NUMBER_T, get_topmost_figure
from atompy._data_xy import DataXY
from atompy._utils import columns_to_meshgrid


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
        _x = np.ravel(x)
        _y = np.ravel(y)
        _z = np.asarray(z)
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
    def x(self) -> NDArray[NUMBER_T]:
        """
        Get x data.

        Returns
        -------
        x : ndarray, shape((n,))
        """
        return self._xmesh[0]

    @x.setter
    def x(self, new_x: ArrayLike) -> None:
        x = np.ravel(new_x)
        if self.x.size != x.size:
            raise ValueError("new size doesn't fit old size")
        xm, _ = np.meshgrid(x, self.y)
        self._xmesh = xm

    @property
    def xmesh(self) -> NDArray[NUMBER_T]:
        """
        Get a meshgrid of x data.
        """
        return self._xmesh

    @property
    def y(self) -> NDArray[NUMBER_T]:
        """
        Get y data.

        Returns
        -------
        y : ndarray, shape((m,))
        """
        return self._ymesh[:, 0]

    @y.setter
    def y(self, new_y: ArrayLike) -> None:
        y = np.ravel(new_y)
        if self.y.size != y.size:
            raise ValueError("new size doesn't fit old size")
        _, ym = np.meshgrid(self.x, y)
        self._ymesh = ym

    @property
    def ymesh(self) -> NDArray[NUMBER_T]:
        """
        Get a meshgrid of y data.
        """
        return self._ymesh

    @property
    def z(self) -> NDArray[NUMBER_T]:
        """
        Get z data.
        """
        return self._zmesh

    @z.setter
    def z(self, new_z: ArrayLike) -> None:
        z = np.asarray(new_z)
        if z.shape != self.z.shape:
            raise ValueError("new shape doesn't fit old shape")
        self._zmesh = z

    @property
    def zmesh(self) -> NDArray[NUMBER_T]:
        """
        Alias of :attr:`.DataXYZ.z`.
        """
        return self.z

    @property
    def plot_kwargs(self) -> dict[str, Any]:
        """
        Get keyword arguments used for :meth:`.DataXYZ.plot`.
        """
        return self._plot_kwargs

    @plot_kwargs.setter
    def plot_kwargs(self, new_kwargs: dict[str, Any]) -> None:
        self._plot_kwargs = new_kwargs

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
        data_layout: Literal["rows", "columns"] = "columns",
        xyz_indices: tuple[int, int, int] = (0, 1, 2),
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        **loadtxt_kwargs,
    ) -> Self:
        """
        Initialize a `DataXYZ` instance from a text file.

        Parameters
        ----------
        fname : str | PathLike
            The path to the text file.

        data_layout : {"rows", "columns"}, default: "columns"
            The layout of the data in the file, either row-major or column-major.

        xyz_indices : tuple[int, int, int], default: (0, 1, 2)
            A tuple specifying the column (or row, depending on `data_layout`)
            indices for x, y, and z data.

        title : str, default ""
            Optional title of the data.

        xlabel : str, default ""
            Optional x-label of the data.

        ylabel : str, default ""
            Optional y-label of the data.

        zlabel : str, default ""
            Optional z-label of the data.

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

        >>> import atompy as ap
        >>> d = ap.DataXYZ.from_txt("scratchpad.txt")
        >>> d.x
        array([1., 2.])
        >>> d.y
        array([1., 2., 3.])
        >>> d.z
        array([[11., 12., 13.],
               [21., 22., 23.]])
        """
        data = np.loadtxt(fname, **loadtxt_kwargs)
        if data_layout == "columns":
            data = data.T
        i, j, k = xyz_indices
        x, y, z = data[i], data[j], data[k]
        xm, ym, zm = columns_to_meshgrid(x, y, z)
        return cls(xm, ym, zm, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    def xmin(self) -> np.number:
        """
        Compute minimum of x = ``np.min(DataXYZ.x)``.
        """
        return np.min(self.x)

    def xmax(self) -> np.number:
        """
        Compute maximum of x = ``np.max(DataXYZ.x)``.
        """
        return np.max(self.x)

    def xlims(self) -> tuple[np.number, np.number]:
        """
        Compute x-limits = ``(DataXYZ.xmin, DataXYZ.xmax)``.
        """
        return (self.xmin(), self.xmax())

    def ymin(self) -> np.number:
        """
        Compute minimum of y = ``np.min(DataXYZ.y)``.
        """
        return np.min(self.y)

    def ymax(self) -> np.number:
        """
        Compute maximum of y = ``np.max(DataXYZ.y)``.
        """
        return np.max(self.y)

    def ylims(self) -> tuple[np.number, np.number]:
        """
        Compute y-limits = ``(DataXYZ.ymin, DataXYZ.ymax)``.
        """
        return (self.ymin(), self.ymax())

    def zmin(self) -> np.number:
        """
        Compute minimum of z = ``np.min(DataXYZ.z)``.
        """
        return np.min(self.z)

    def zmax(self) -> np.number:
        """
        Compute maximum of z = ``np.max(DataXYZ.z)``.
        """
        return np.max(self.z)

    def zlims(self) -> tuple[np.number, np.number]:
        """
        Compute z-limits = ``(DataXYZ.zmin, DataXYZ.zmax)``.
        """
        return (self.zmin(), self.zmax())

    def integrate(self) -> float:
        """
        Compute integral of data.

        It uses :func:`numpy.trapezoid` to perform the calculation.

        Returns
        -------
        integral : float
        """
        return np.trapezoid(np.trapezoid(self.z, self.y, axis=1), self.x, axis=0)

    def integrate_x(self) -> DataXY: ...
    def integrate_y(self) -> DataXY: ...

    def norm_to_integral(self) -> Self:
        normed_z = self.z / self.integrate()
        return type(self)(self.x, self.y, normed_z, **self._kwargs, **self._plot_kwargs)

    def norm_to_max(self) -> Self:
        normed_z = self.z / self.zmax()
        return type(self)(self.x, self.y, normed_z, **self._kwargs, **self._plot_kwargs)

    def keep_x(self, xmin: float = -np.inf, xmax: float = np.inf) -> Self: ...
    def keep_y(self, ymin: float = -np.inf, ymax: float = np.inf) -> Self: ...
    def remove_x(self, xmin: float = -np.inf, xmax: float = np.inf) -> Self: ...
    def remove_y(self, ymin: float = -np.inf, ymax: float = np.inf) -> Self: ...

    def copy(self) -> Self: ...

    def for_pcolormesh(
        self,
    ) -> tuple[NDArray[NUMBER_T], NDArray[NUMBER_T], NDArray[NUMBER_T]]:
        """
        Get data in the appropriate format for :func:`matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        x, y, z
            Equal to :attr:`.DataXYZ.x`, :attr:`.DataXYZ.x`,
            :attr:`DataXYZ.z.T<.DataXYZ.z>`
        """
        x = self.x
        y = self.y
        x_edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
        y_edges = np.r_[y[0], (y[:-1] + y[1:]) / 2, y[-1]]
        return x_edges, y_edges, self.z.T
        return self.xmesh, self.ymesh, self.z.T

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
        colorbar_kwargs: dict[str, Any] | None = None,
        savefig_kwargs: dict[str, Any] | None = None,
        **pcolormesh_kwargs,
    ) -> tuple[Figure, Axes, Colorbar]:
        """
        Plot the 2D data using :obj:`matplotlib.pyplot.pcolormesh`.

        Parameters
        ----------
        fname : str, optional
            If provided, the plot will be saved to this file.

        xlabel : str, default "__auto__"
            Label for the x-axis.

            If "__auto__", use `DataXYZ.xlabel`.

        ylabel : str, default "__auto__"
            Label for the y-axis.

            If "__auto__", use `DataXYZ.ylabel`.

        zlabel : str, default "__auto__"
            Label for the colorbar (z-axis).

            If "__auto__", use `DataXYZ.zlabel`.

        title : str, default "__auto__"
            Title of the plot.

            If "__auto__", use `DataXYZ.title`.

        logscale_x : bool, default False
            If True, use a logarithmic x scale.

        logscale_y : bool, default False
            If True, use a logarithmic x scale.

        logscale_z : bool, default False
            If True, use a logarithmic color scale.

        xlim : tuple[float, float], optional
            Limits for the x-axis.

        ylim : tuple[float, float], optional
            Limits for the y-axis.

        zlim : tuple[float, float], optional
            Limits of the z-axis (color scale).

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

        Returns
        -------
        tuple of Figure, Axes, Colorbar
            A tuple containing the matplotlib Figure, Axes, and Colorbar
            objects.

        Examples
        --------

        .. plot:: _examples/dataxyz/plot.py
            :include-source:

        .. plot:: _examples/dataxyz/plot_in_axes.py
            :include-source:
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = get_topmost_figure(ax)
        norm = LogNorm() if logscale_z else None
        pcolormesh_kwargs_ = pcolormesh_kwargs.copy()
        pcolormesh_kwargs_.setdefault("norm", norm)
        pcolormesh_kwargs_.setdefault("rasterized", True)
        pcolormesh_kwargs_.setdefault("shading", "flat")
        # im = ax.pcolormesh(*self.for_pcolormesh(), **pcolormesh_kwargs_)
        im = ax.tricontourf(
            self.xmesh.ravel(), self.ymesh.ravel(), self.z.ravel(), rasterized=True
        )

        cbar_kwargs = colorbar_kwargs.copy() if colorbar_kwargs else {}
        cbar_kwargs.setdefault("use_gridspec", False)
        cb = fig.colorbar(im, ax=ax, **cbar_kwargs)
        cb.set_label(zlabel if zlabel != "__auto__" else self.zlabel)

        title_ = title if title != "__auto__" else self.title
        if title_ != "":
            fig.canvas.manager.set_window_title(title_)  # type: ignore
        ax.set_title(title_)

        ax.set_xlabel(xlabel if xlabel != "__auto__" else self.xlabel)
        ax.set_ylabel(ylabel if ylabel != "__auto__" else self.ylabel)

        ax.set_xlim(xlim)  # ty: ignore[invalid-argument-type]
        ax.set_ylim(ylim)  # ty: ignore[invalid-argument-type]

        if logscale_x:
            ax.set_xscale("log")
        if logscale_y:
            ax.set_yscale("log")

        if fname is not None:
            savefig_kwargs = savefig_kwargs if savefig_kwargs else {}
            fig.savefig(fname, **savefig_kwargs)

        return fig, ax, cb

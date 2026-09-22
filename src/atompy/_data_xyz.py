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
    def _kwargs(self) -> DataXYZKwargs:
        return {
            "title": copy.copy(self.title),
            "xlabel": copy.copy(self.xlabel),
            "ylabel": copy.copy(self.ylabel),
            "zlabel": copy.copy(self.ylabel),
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
        return self._zlabel

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
        dataxyz
            A new :class:`.DataXYZ` instance.

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
        x_, y_, zm = columns_to_meshgrid(x, y, z)
        return cls(x_, y_, zm, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    @classmethod
    def from_table(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
    ) -> Self:
        """
        Initialize a `DataXYZ` instance from a look-up table.

        Each row in the LUT must list the x-coordinate, y-coordinate and
        corresponding z-value.

        Parameters
        ----------
        x : array_like, shape(n*m)
            x coordinates as *n* unique values that are repeated *m* times.

        y : array_like, shape(n*m)
            y coordinates as *m* unique values that are repeated *n* times.

        z : array_like, shape(n*m)
            Corresponding z values.

        title : str, default ""
            Optional title of the data.

        xlabel : str, default ""
            Optional x-label of the data.

        ylabel : str, default ""
            Optional y-label of the data.

        zlabel : str, default ""
            Optional z-label of the data.

        Returns
        -------
        dataxyz
            A new :class:`.DataXYZ` instance.

        See also
        --------
        from_txt
            Instead of manually loading data given in a text file and then
            calling `from_lut`, you can use :meth:`.DataXYZ.from_txt`.

        Examples
        --------
        >>> import atompy as ap
        >>> d = ap.DataXYZ.from_table((1, 1, 1, 2, 2, 2), (1, 2, 3, 1, 2, 3), (11, 12, 13, 21, 22, 23))
        >>> d.x
        array([1, 2])
        >>> d.y
        array([1, 2, 3])
        >>> d.z
        array([[11, 12, 13],
               [21, 22, 23]])
        """
        x_, y_, zm = columns_to_meshgrid(np.asarray(x), np.asarray(y), np.asarray(z))
        return cls(x_, y_, zm, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

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

        Uses :func:`numpy.trapezoid` to perform the calculation.

        Returns
        -------
        integral : float
        """
        return np.trapezoid(np.trapezoid(self.z, self.y, axis=1), self.x, axis=0)

    def integrate_x(
        self,
        *,
        xlabel: str = "__auto__",
    ) -> DataXY:
        """
        Integrate over x-axis of data.

        Uses :func:`numpy.trapezoid` to perform the calculation.

        Parameters
        ----------
        xlabel : str, default "__auto__"
            If not "__auto__", :attr:`.DataXY.xlabel` of the returned data will
            be set to this.

            If "__auto__", :attr:`.DataXYZ.ylabel` will be used.

        Returns
        -------
        data : :class:`.DataXY`
            xy-data, where the new x values are the old y values of the
            original xyz-data and the new y values are the old z values,
            integrated over each row.

        Examples
        --------

        .. plot:: _examples/dataxyz/integrate_xy.py
            :include-source:
        """
        integral = np.trapezoid(self.z, self.x, axis=0)
        xlabel = self.ylabel if xlabel == "__auto__" else ""
        return DataXY(self.y, integral, xlabel=xlabel)

    def integrate_y(
        self,
        *,
        xlabel: str = "__auto__",
    ) -> DataXY:
        """
        Integrate over y-axis of data.

        Uses :func:`numpy.trapezoid` to perform the calculation.

        Parameters
        ----------
        xlabel : str, default "__auto__"
            If not "__auto__", :attr:`.DataXY.xlabel` of the returned data will
            be set to this.

            If "__auto__", :attr:`.DataXYZ.xlabel` will be used.

        Returns
        -------
        data : :class:`.DataXY`
            xy-data, where the new x values are the old y values of the
            original xyz-data and the new y values are the old z values,
            integrated over each row.

        Examples
        --------

        .. plot:: _examples/dataxyz/integrate_xy.py
            :include-source:
        """
        integral = np.trapezoid(self.z, self.y, axis=1)
        xlabel = self.xlabel if xlabel == "__auto__" else ""
        return DataXY(self.x, integral, xlabel=xlabel)

    def norm_to_integral(
        self,
        *,
        title: str = "__auto__",
        xlabel: str = "__auto__",
        ylabel: str = "__auto__",
        zlabel: str = "",
    ) -> Self:
        """
        Get a copy of the data that is normalized to the integral.

        Parameters
        ----------
        title, xlabel, ylabel : str, default "__auto__"
            If "__auto__", use original title/xlabel/ylabel in output.

            Else, update.

        zlabel : str, default ""
            If "__auto__", use original zlabel in output.

            Else, update.

            zlabel is cleared by default.

        Returns
        -------
        normalized_data : :class:`.DataXYZ`

        See also
        --------
        integrate
        norm_to_max

        Examples
        --------

        >>> import atompy as ap
        >>> d = ap.DataXYZ((0, 1), (0, 1), ((1, 2), (3, 4)))
        >>> d.z
        array([[1, 2],
               [3, 4]])
        >>> d.norm_to_integral().z
        array([[0.4, 0.8],
               [1.2, 1.6]])
        >>> d.norm_to_integral().integrate()
        np.float64(1.0)
        """
        normed_z = self.z.copy() / self.integrate()
        kwargs = self._kwargs.copy()
        kwargs["title"] = kwargs["title"] if title == "__auto__" else title
        kwargs["xlabel"] = kwargs["xlabel"] if xlabel == "__auto__" else xlabel
        kwargs["ylabel"] = kwargs["ylabel"] if ylabel == "__auto__" else ylabel
        kwargs["zlabel"] = kwargs["zlabel"] if zlabel == "__auto__" else zlabel
        return type(self)(self.x.copy(), self.y.copy(), normed_z, **kwargs)

    def norm_to_max(
        self,
        *,
        title: str = "__auto__",
        xlabel: str = "__auto__",
        ylabel: str = "__auto__",
        zlabel: str = "",
    ) -> Self:
        """
        Get a copy of the data that is normalized to the integral.

        Parameters
        ----------
        title, xlabel, ylabel : str, default "__auto__"
            If "__auto__", use original title/xlabel/ylabel in output.

            Else, update.

        zlabel : str, default ""
            If "__auto__", use original zlabel in output.

            Else, update.

            zlabel is cleared by default.

        Returns
        -------
        normalized_data : :class:`.DataXYZ`

        See also
        --------
        zmax
        norm_to_integral

        Examples
        --------

        >>> import atompy as ap
        >>> d = ap.DataXYZ((0, 1), (0, 1), ((-3, 2), (1, 2)))
        >>> d.z
        array([[1, 2],
               [3, 4]])
        >>> d.norm_to_max().z
        array([[0.25, 0.5 ],
               [0.75, 1.  ]])
        >>> d.norm_to_max().zmax()
        np.float64(1.0)
        """
        normed_z = self.z.copy() / self.zmax()
        kwargs = self._kwargs.copy()
        kwargs["title"] = kwargs["title"] if title == "__auto__" else title
        kwargs["xlabel"] = kwargs["xlabel"] if xlabel == "__auto__" else xlabel
        kwargs["ylabel"] = kwargs["ylabel"] if ylabel == "__auto__" else ylabel
        kwargs["zlabel"] = kwargs["zlabel"] if zlabel == "__auto__" else zlabel
        return type(self)(self.x.copy(), self.y.copy(), normed_z, **kwargs)

    def get_closest(self, x: float, y: float) -> np.number:
        """
        Get closest z(x, y).

        Parameters
        ----------
        x, y : float

        Returns
        -------
        z_value : float

        Examples
        --------
        >>> import atompy as ap
        >>> d = ap.DataXYZ((1, 2), (1, 2, 3), ((11, 12, 13), (21, 22, 23)))
        >>> d.z
        array([[11, 12, 13],
               [21, 22, 23]])
        >>> d.z[0, 1]
        np.int64(12)
        >>> d.get_closest(1, 2)
        np.int64(12)
        >>> d.get_closest(1.1, 2.5)
        np.int64(12)
        >>> d.get_closest(1.1, 2.6)
        np.int64(13)
        >>> d.get_closest(-1, -1)
        Traceback (most recent call last):
        ...
        ValueError: x=-1 outside of data range=(np.int64(1), np.int64(2))
        """
        if x < self.xmin() or x > self.xmax():
            raise ValueError(f"{x=} outside of data range={self.xlims()}")
        if y < self.ymin() or y > self.ymax():
            raise ValueError(f"{y=} outside of data range={self.ylims()}")
        ix = np.argmin(np.abs(self.x - x))
        iy = np.argmin(np.abs(self.y - y))
        return self.z[ix, iy]

    def get_closest_x(self, x: float, *, xlabel: str = "__auto__") -> DataXY:
        """
        Get a slice of data y vs. z closest to `x`.

        Parameters
        ----------
        x : float
            x-value of the slice.

        xlabel : str, default = "__auto__"
            X-label of the :class:`.DataXY` output.

            If "__auto__", use :attr:`.~DataXYZ.ylabel` of current object.

        Returns
        -------
        data_slice : :class:`.DataXY`

        Examples
        --------
        .. plot:: _examples/dataxyz/get_closest.py
            :include-source:
        """
        if x < self.xmin() or x > self.xmax():
            raise ValueError(f"{x=} outside of data range={self.xlims()}")
        ix = np.argmin(np.abs(self.x - x))
        xlabel = self.ylabel if xlabel == "__auto__" else xlabel
        return DataXY(self.y.copy(), self.z[ix, :].copy(), xlabel=xlabel)

    def get_closest_y(self, y: float, *, xlabel: str = "__auto__") -> DataXY:
        """
        Get a slice of data x vs. z closest to `y`.

        Parameters
        ----------
        y : float
            y-value of the slice.

        xlabel : str, default = "__auto__"
            X-label of the :class:`.DataXY` output.

            If "__auto__", use :attr:`.~DataXYZ.xlabel` of current object.

        Returns
        -------
        data_slice : :class:`.DataXY`

        Examples
        --------
        .. plot:: _examples/dataxyz/get_closest.py
            :include-source:
        """
        if y < self.ymin() or y > self.ymax():
            raise ValueError(f"{y=} outside of data range={self.ylims()}")
        iy = np.argmin(np.abs(self.y - y))
        xlabel = self.xlabel if xlabel == "__auto__" else xlabel
        return DataXY(self.y.copy(), self.z[:, iy].copy(), xlabel=xlabel)

    def _mask(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> NDArray[np.bool_]:
        xmask = (self.x >= xmin) & (self.x < xmax)
        ymask = (self.x >= ymin) & (self.x < ymax)
        return xmask[:, None] & ymask[None, :]

    def keep(
        self,
        xmin: float = -np.inf,
        xmax: float = np.inf,
        ymin: float = -np.inf,
        ymax: float = np.inf,
        *,
        squeeze: bool = False,
        setval: float = 0.0,
    ) -> Self:
        """
        Only keep data within specified range [min, max).

        Parameters
        ----------
        xmin, ymin : float, default -numpy.inf
            The inclusive minimum value to keep.

        xmax, ymax : float, default +numpy.inf
            The exclusive maximum value to keep.

        squeeze : bool, default False
            If true, trim data range to only kept data.

        setval : float, default 0.0
            Set removed values to this.

            Has no effect if `squeeze=True`.

        Returns
        -------
        kept_data : :class:`.DataXYZ`

        See also
        --------
        keep_x
        keep_y
        remove

        Examples
        --------
        .. plot:: _examples/dataxyz/keep.py
            :include-source:
        """
        mask = self._mask(xmin, xmax, ymin, ymax)

        if squeeze:
            if not np.any(mask):
                raise ValueError("Selected region does not overlap with any data.")
            x_indices = np.any(mask, axis=1)
            y_indices = np.any(mask, axis=0)
            return type(self)(
                self.x[x_indices].copy(),
                self.y[y_indices].copy(),
                self.z[np.ix_(x_indices, y_indices)].copy(),
                **self._kwargs.copy(),
            )

        new_z = np.full_like(self.z, fill_value=setval, dtype=np.float64)
        new_z[mask] = self.z[mask].copy()
        result = self.copy()
        result.z = new_z
        return result

    def keep_x(
        self,
        xmin: float = -np.inf,
        xmax: float = np.inf,
        *,
        squeeze: bool = False,
        setval: float = 0.0,
    ) -> Self:
        """
        Only keep data within specified x-range [xmin, xmax).

        Parameters
        ----------
        xmin : float, default -numpy.inf
            The inclusive minimum x-value to keep.

        xmax : float, default +numpy.inf
            The exclusive maximum x-value to keep.

        squeeze : bool, default False
            If true, trim data range to only kept data.

        setval : float, default 0.0
            Set removed values to this.

            Has no effect if `squeeze=True`.

        Returns
        -------
        kept_data : :class:`.DataXYZ`

        See also
        --------
        keep
        keep_y
        remove

        Examples
        --------
        .. plot:: _examples/dataxyz/keep_x.py
            :include-source:
        """
        return self.keep(xmin=xmin, xmax=xmax, squeeze=squeeze, setval=setval)

    def keep_y(
        self,
        ymin: float = -np.inf,
        ymax: float = np.inf,
        *,
        squeeze: bool = False,
        setval: float = 0.0,
    ) -> Self:
        """
        Only keep data within specified y-range [ymin, ymax).

        Parameters
        ----------
        ymin : float, default -numpy.inf
            The inclusive minimum y-value to keep.

        ymax : float, default +numpy.inf
            The exclusive maximum y-value to keep.

        squeeze : bool, default False
            If true, trim data range to only kept data.

        setval : float, default 0.0
            Set removed values to this.

            Has no effect if `squeeze=True`.

        Returns
        -------
        kept_data : :class:`.DataXYZ`

        See also
        --------
        keep
        keep_x
        remove

        Examples
        --------
        .. plot:: _examples/dataxyz/keep_y.py
            :include-source:
        """
        return self.keep(ymin=ymin, ymax=ymax, squeeze=squeeze, setval=setval)

    def remove(
        self,
        xmin: float = -np.inf,
        xmax: float = np.inf,
        ymin: float = -np.inf,
        ymax: float = np.inf,
        *,
        setval: float = 0.0,
    ) -> Self:
        """
        Remove data within specified range [min, max).

        Parameters
        ----------
        xmin, ymin : float, default -numpy.inf
            The inclusive minimum value to remove.

        xmax, ymax : float, default +numpy.inf
            The exclusive maximum value to remove.

        setval : float, default 0.0
            Set removed values to this.

        Returns
        -------
        kept_data : :class:`.DataXYZ`

        See also
        --------
        remove_x
        remove_y
        keep

        Examples
        --------
        .. plot:: _examples/dataxyz/remove.py
            :include-source:
        """
        mask = self._mask(xmin, xmax, ymin, ymax)
        new_z = self.z.copy().astype(float)
        new_z[mask] = setval
        result = self.copy()
        result.z = new_z
        return result

    def remove_x(
        self, xmin: float = -np.inf, xmax: float = np.inf, *, setval: float = 0.0
    ) -> Self:
        """
        Remove data within specified x-range [xmin, xmax).

        Parameters
        ----------
        xmin : float, default -numpy.inf
            The inclusive minimum value to remove.

        xmax : float, default +numpy.inf
            The exclusive maximum value to remove.

        setval : float, default 0.0
            Set removed values to this.

        Returns
        -------
        kept_data : :class:`.DataXYZ`

        See also
        --------
        remove
        remove_y
        keep

        Examples
        --------
        .. plot:: _examples/dataxyz/remove_x.py
            :include-source:
        """
        return self.remove(xmin=xmin, xmax=xmax, setval=setval)

    def remove_y(
        self, ymin: float = -np.inf, ymax: float = np.inf, *, setval: float = 0.0
    ) -> Self:
        """
        Remove data within specified x-range [xmin, xmax).

        Parameters
        ----------
        ymin : float, default -numpy.inf
            The inclusive minimum value to remove.

        ymax : float, default +numpy.inf
            The exclusive maximum value to remove.

        setval : float, default 0.0
            Set removed values to this.

        Returns
        -------
        kept_data : :class:`.DataXYZ`

        See also
        --------
        remove
        remove_x
        keep

        Examples
        --------
        .. plot:: _examples/dataxyz/remove_y.py
            :include-source:
        """
        return self.remove(ymin=ymin, ymax=ymax, setval=setval)

    def copy(self) -> Self:
        """
        Get a copy of the :class:`!.DataXYZ` instance

        Returns
        -------
        copied_data : :class:`!.DataXYZ`.
        """
        return type(self)(
            self.x.copy(),
            self.y.copy(),
            self.z.copy(),
            **self._kwargs.copy(),
        )

    def for_pcolormesh(
        self, shading: Literal["flat", "nearest", "gouraud"] = "flat"
    ) -> tuple[NDArray[NUMBER_T], NDArray[NUMBER_T], NDArray[NUMBER_T]]:
        """
        Get data in the appropriate format for :func:`matplotlib.pyplot.pcolormesh`.

        Parameters
        ----------
        shading : "flat", "nearest", or "gouraud", default "flat"
            See documentation for the `shading` keyword of
            :func:`~matplotlib.pyplot.pcolormesh` and examples.

        Returns
        -------
        x, y, z : ndarray, ndarray, ndarray
            Appropriate  data layout for calling :func:`~matplotlib.pyplot.pcolormesh`
            with the `shading=shading` keyword.

        Examples
        --------
        .. plot:: _examples/dataxyz/for_pcolormesh.py
            :include-source:
        """
        if shading == "flat":
            x = self.x
            y = self.y
            x_edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
            y_edges = np.r_[y[0], (y[:-1] + y[1:]) / 2, y[-1]]
            return x_edges, y_edges, self.z.T
        elif shading == "nearest" or shading == "gouraud":
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
        shading: Literal["flat", "nearest", "gouraud"] = "flat",
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
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = get_topmost_figure(ax)
        norm = LogNorm() if logscale_z else None
        pcolormesh_kwargs_ = pcolormesh_kwargs.copy()
        pcolormesh_kwargs_.setdefault("norm", norm)
        pcolormesh_kwargs_.setdefault("rasterized", True)
        pcolormesh_kwargs_["shading"] = shading
        if zlim is not None:
            if zlim[0] is not None:
                pcolormesh_kwargs_.setdefault("vmin", zlim[0])
            if zlim[1] is not None:
                pcolormesh_kwargs_.setdefault("vmax", zlim[1])

        im = ax.pcolormesh(*self.for_pcolormesh(shading), **pcolormesh_kwargs_)

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

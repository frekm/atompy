from typing import Iterator, Any, Literal, Self
from os import PathLike

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DataXY:
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        _x = np.asarray(x, dtype=np.float64, copy=True)
        _y = np.asarray(y, dtype=np.float64, copy=True)
        if _x.size != _y.size:
            raise ValueError("x and y values don't match")
        self._data = np.array((_x, _y))
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._plot_kwargs: dict[str, Any] = {}

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

        idx_x : int, default 0
            The index that corresponds to the x values.

        idx_y : int, default 1
            The index that corresponds to the y values.

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
        return cls(data[idx_x], data[idx_y], title, xlabel, ylabel)

    @property
    def x(self) -> NDArray[np.float64]:
        return self._data[0]

    @x.setter
    def x(self, values: ArrayLike) -> None:
        self._data[0] = values

    @property
    def y(self) -> NDArray[np.float64]:
        return self._data[1]

    @y.setter
    def y(self, values: ArrayLike) -> None:
        self._data[1] = values

    @property
    def xy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.x, self.y

    @property
    def plot_kwargs(self) -> dict[str, Any]:
        return self._plot_kwargs

    @plot_kwargs.setter
    def plot_kwargs(self, new_dict) -> None:
        self._plot_kwargs = new_dict.copy()

    def __len__(self) -> int:
        return self._data.shape[1]

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        return iter(self._data.T)

    def asarray(self, copy: bool = False) -> NDArray[np.float64]:
        return self._data.copy() if copy else self._data

    def xmin(self) -> float:
        return float(self.x.min())

    def xmax(self) -> float:
        return float(self.x.max())

    def ymin(self) -> float:
        return float(self.y.min())

    def ymax(self) -> float:
        return float(self.y.max())

    def xlims(self) -> tuple[float, float]:
        return self.xmin(), self.xmax()

    def ylims(self) -> tuple[float, float]:
        return self.ymin(), self.ymax()

    def sum(self) -> float:
        return float(np.sum(self.y))

    def integrate(self) -> float:
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
        new_data = type(self)(
            self.x.copy(), self.y.copy(), self.title, self.xlabel, self.ylabel
        )
        new_data.plot_kwargs = self.plot_kwargs.copy()
        return new_data

    def filter_x(self, xmin, xmax) -> Self: ...
    def filter_y(self, xmin, xmax) -> Self: ...

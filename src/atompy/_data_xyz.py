from typing import Literal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray

import atompy._utils as utils


class DataXYZ:
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        **pcolormesh_kwargs,
    ) -> None:
        self._x = np.unique(np.asarray(x).flat)
        self._y = np.unique(np.asarray(y).flat)
        self._z = np.asarray(z)
        if self._z.ndim == 1:
            self._z = self._z.reshape((self._x.size, self._y.size))

        if self._x.size != self._z.shape[0]:
            raise ValueError("mismatch of x and z data")
        if self._y.size != self._z.shape[1]:
            raise ValueError("mismatch of y and z data")

        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self._z = np.asarray(z)
        self._data = np.array((x, y, z))

        self._pcolormesh_kwargs = pcolormesh_kwargs
        self._pcolormesh_kwargs.setdefault("rasterized", True)
        self._pcolormesh_kwargs.setdefault("shading", "nearest")
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._zlabel = zlabel

    @property
    def x(self) -> NDArray[np.number]:
        return self._data[0]

    @property
    def y(self) -> NDArray[np.number]:
        return self._data[1]

    @property
    def z(self) -> NDArray[np.number]:
        return self._data[2]

    @x.setter
    def x(self, vals: ArrayLike) -> None:
        self._data[0] = vals

    @y.setter
    def y(self, vals: ArrayLike) -> None:
        self._data[1] = vals

    @z.setter
    def z(self, vals: ArrayLike) -> None:
        self._data[2] = vals

    @property
    def xyz(self) -> NDArray[np.number]:
        return self._data

    def x_values(self) -> NDArray[np.number]:
        return np.unique(self.x)

    def y_values(self) -> NDArray[np.number]:
        return np.unique(self.y)

    def size_x(self) -> int:
        return len(self.x_values())

    def size_y(self) -> int:
        return len(self.y_values())

    def iteration_order(self) -> Literal["x_first", "y_first"]:
        xs = self.x_values()
        if xs[1] != self.x[1]:
            return "x_first"
        ys = self.y_values()
        if ys[1] != self.y[1]:
            return "y_first"
        raise ValueError("cannot determine iteration order of x and y data")

    def z_grid(self) -> NDArray[np.number]:
        z_ = self.z
        if self.iteration_order == "x_first":
            z_ = z_.reshape(self.size_y(), self.size_x()).T
        else:
            z_ = z_.reshape(self.size_x(), self.size_y())
        return z_

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        ax = ax or plt.gca()
        ax.pcolormesh(self._x, self._y, self._z.T, **self._pcolormesh_kwargs)
        # ax.imshow(self._z)
        return plt.gcf(), plt.gca()

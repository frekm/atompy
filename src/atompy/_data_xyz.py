import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike


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

        self._pcolormesh_kwargs = pcolormesh_kwargs
        self._pcolormesh_kwargs.setdefault("rasterized", True)
        self._pcolormesh_kwargs.setdefault("shading", "nearest")
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._zlabel = zlabel

    def plot(self) -> tuple[Figure, Axes]:
        plt.pcolormesh(self._x, self._y, self._z.T, **self._pcolormesh_kwargs)
        return plt.gcf(), plt.gca()

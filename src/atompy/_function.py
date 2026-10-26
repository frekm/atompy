from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Function:
    def __init__(
        self,
        f: Callable,
        xlims: tuple[float, float],
        *,
        steps: int = 100,
        xlog: bool = False,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **fkwargs,
    ) -> None:
        self._f = f
        self._xlims = xlims
        self._lin_or_logspace = np.logspace if xlog else np.linspace
        self._steps = steps
        self._is_stale = True
        self.update()

    def update(self) -> None:
        if self._is_stale:
            self._x = self._lin_or_logspace(*self.xlims, self.steps)
            self._y = self(self._x)
        self._is_stale = False

    def __call__(self, x: ArrayLike) -> NDArray[np.number]:
        self.update()
        return self._f(np.asarray(x))

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = value
        self._is_stale = True

    @property
    def xlog(self) -> bool:
        return self._lin_or_logspace is np.logspace

    @xlog.setter
    def xlog(self, value: bool) -> None:
        if value != self.xlog:
            self._lin_or_logspace = np.logspace if value else np.linspace
            self._is_stale = True

    @property
    def xlims(self) -> tuple[float, float]:
        return self._xlims

    @xlims.setter
    def xlims(self, new_xlims: tuple[float, float]) -> None:
        self._xlims = new_xlims
        self._is_stale = True

    def x(self) -> NDArray[np.number]:
        self.update()
        return self._x

    def y(self) -> NDArray[np.number]:
        self.update()
        return self._y

    def is_invertible(self) -> np.bool:
        monotonically_increasing = np.all(self.y()[:-1] < self.y()[1:])
        monotonically_decreasing = np.all(self.y()[:-1] > self.y()[1:])
        return monotonically_increasing or monotonically_decreasing

    def inverted(self) -> Self:
        if not self.is_invertible():
            raise TypeError()
            # TODO

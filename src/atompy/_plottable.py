from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, Self

P = ParamSpec("P")


class Plottable:
    def __init__(self, title: str, labels: tuple[str, ...], **kwargs):
        self._title = title
        self._labels = labels
        self._plot_kwargs = kwargs
        self._plot = self.plot_default

    @property
    def plot_kwargs(self) -> dict[str, Any]:
        """
        Associated keyword arguments used for :attr:`plot`
        """
        return self._plot_kwargs

    @plot_kwargs.setter
    def plot_kwargs(self, kwargs: dict[str, Any]) -> None:
        self._plot_kwargs = kwargs.copy()

    @property
    def title(self) -> str:
        """
        Associated title.
        """
        return self._title

    @title.setter
    def title(self, title: str) -> None:
        self._title = title

    @title.setter
    def title(self, title: str) -> None:
        self._labels = title

    def plot_default(self, *args, **kwargs) -> Any: ...

    def set_plot_method(
        self,
        method: Callable[Concatenate[Self, P], Any] | None,
    ) -> None:
        """
        Set a custom plot method.

        Parameters
        ----------
        method : Callable or None
            A custom plot method.

            The first input parameter must be of type the class type.
            No other restrictions exist on the call signature.

            If None, :attr:`!plot` will call :meth:`plot_default`.
        """
        if method is None:
            self._plot = self.plot_default
        else:
            self._plot = lambda *args, **kwargs: method(self, *args, **kwargs)

    @property
    def plot(self) -> Callable[..., Any]:
        """
        Plot data.

        By default, calls :meth:`plot_default`, but that can be changed
        using :meth:`set_plot_method`.
        """
        return self._plot


class Plottable2d(Plottable):
    def __init__(self, title: str, xlabel: str, ylabel: str, **kwargs) -> None:
        super().__init__(title, (xlabel, ylabel), **kwargs)
        self.xlabel = xlabel
        self.ylabel = ylabel

    @property
    def xlabel(self) -> str:
        """
        Associated x-label.
        """
        return self._labels[0]

    @xlabel.setter
    def xlabel(self, xlabel: str) -> None:
        new_labels = list(self._labels)
        new_labels[0] = xlabel
        self._labels = tuple(new_labels)

    @property
    def ylabel(self) -> str:
        """
        Associated y-label.
        """
        return self._labels[1]

    @ylabel.setter
    def ylabel(self, ylabel: str) -> None:
        new_labels = list(self._labels)
        new_labels[1] = ylabel
        self._labels = tuple(new_labels)


class Plottable3d(Plottable):
    def __init__(
        self, title: str, xlabel: str, ylabel: str, zlabel: str, **kwargs
    ) -> None:
        super().__init__(title, (xlabel, ylabel, zlabel), **kwargs)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

    @property
    def zlabel(self) -> str:
        """
        Associated z-label.
        """
        return self._labels[2]

    @zlabel.setter
    def zlabel(self, zlabel: str) -> None:
        new_labels = list(self._labels)
        new_labels[2] = zlabel
        self._labels = tuple(new_labels)

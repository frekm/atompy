import warnings
from typing import Any, Literal, TypeVar, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from numpy.typing import NDArray

from .errors import UnmatchingEdgesError

NUMBER_T = TypeVar("NUMBER_T", bound=np.number)


def get_topmost_figure(ax: Axes) -> Figure:
    """
    Get the parent figure of `ax`.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`

    Returns
    -------
    figure : :class:`matplotlib.figure.Figure`
    """
    fig = ax.get_figure()
    if fig is None:
        raise ValueError("'ax' is not assigned to any figure")
    while isinstance(fig, SubFigure):
        fig = fig.get_figure()
    return cast(Figure, fig)


def raise_unmatching_edges(
    a: NDArray[Any], b: NDArray[Any], xy: Literal["x", "y", "xy", ""] = ""
) -> None:
    if not np.allclose(a, b):
        raise UnmatchingEdgesError(xy)


def warn_unmatching_normfacs(f0: float, f1: float) -> None:
    if not np.allclose(f0, f1):
        warnings.warn("Normalization factors do not match!", stacklevel=2)


def deprecated_keyword_doing_nothing_msg(keyword: str):
    msg = f"The keyword {keyword} is depcrecated and does nothing"
    return msg


def normalize_columns_or_row_data(
    data: NDArray[np.number], layout: Literal["columns", "rows"]
) -> NDArray[np.number]:
    if layout not in ("columns", "rows"):
        raise ValueError(f"{layout=}, but it must be 'columns' or 'rows'")
    return data if layout == "rows" else data.T

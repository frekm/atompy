from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from typing import cast, Any, Literal


from .errors import UnmatchingEdgesError


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


def deprecated_keyword_doing_nothing_msg(keyword: str):
    msg = f"The keyword {keyword} is depcrecated and does nothing"
    return msg

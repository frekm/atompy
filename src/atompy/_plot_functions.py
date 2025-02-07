import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
from numpy.typing import ArrayLike
from typing import Optional, Literal, Union, Any, Sequence, overload
from . import _plotting
from ._plotting import MM_PER_INCH

FIGURE_WIDTH_PRL_1COL = 3.0 + 3.0/8.0
FIGURE_WIDTH_PRL_2COL = 2.0 * FIGURE_WIDTH_PRL_1COL
FIGURE_WIDTH_NATURE_1COL = 90.0 / MM_PER_INCH
FIGURE_WIDTH_NATURE_2COL = 180.0 / MM_PER_INCH
FIGURE_WIDTH_SCIENCE_1COL = 2.25
FIGURE_WIDTH_SCIENCE_2COL = 4.75
FIGURE_WIDTH_SCIENCE_3COL = 7.25
GOLDENRATIO = 1.618033988749


def set_style_science(figwidth: float = FIGURE_WIDTH_SCIENCE_1COL):
    # TODO
    print("set_style_science: not yet implemented")
    ...


def set_style_prl(figwidth: float = FIGURE_WIDTH_PRL_1COL):
    # TODO
    print("set_style_prl: not yet implemented")
    ...


def set_style_nature(figwidth: float = FIGURE_WIDTH_NATURE_1COL):
    # TODO
    print("set_style_nature: not yet implemented")
    ...


def _format_axes(
    ax: Axes,
    xlabel: Optional[str],
    ylabel: Optional[str],
    title: Optional[str],
    xmin: Optional[float],
    xmax: Optional[float],
    ymin: Optional[float],
    ymax: Optional[float],
    aspect_ratio: float,
    axes_width_inch: Optional[float],
    axes_height_inch: Optional[float],
):
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if axes_width_inch is None and axes_height_inch is None:
        ax.set_box_aspect(aspect_ratio)
    elif axes_width_inch is None and axes_height_inch is not None:
        _plotting.set_axes_size(axes_height_inch / aspect_ratio,
                                axes_height_inch,
                                ax)
    elif axes_width_inch is not None and axes_height_inch is None:
        _plotting.set_axes_size(axes_width_inch,
                                axes_width_inch * aspect_ratio,
                                ax)
    elif axes_width_inch is not None and axes_height_inch is not None:
        _plotting.set_axes_size(axes_width_inch, axes_height_inch)


def create_1d_plot(
    *datasets: ArrayLike,
    plot_function: Literal["plot", "step"] = "plot",
    plot_kwargs_all: dict[str, Any] = {},
    plot_kwargs_per: Optional[Sequence[dict[str, Any]]] = None,
    legend_kwargs: dict[str, Any] = {},
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    aspect_ratio: float = 1.0 / GOLDENRATIO,
    axes_width_inch: Optional[float] = 80.0 / MM_PER_INCH,
    axes_height_inch: Optional[float] = None,
    make_me_nice: bool = True,
    make_me_nice_kwargs: dict[str, Any] = {},
) -> tuple[Figure, Axes]:
    """
    Plot 1D dataset(s).

    Parameters
    ----------
    *datasets : ((x1s, y1s), (x2s, y2s), ...)
        One or many dataset(s) to be plotted.
        
        Each dataset must be composed of two arrays x and y.

    plot_function : ``"plot"`` or ``"step"``, default ``"plot"``
        Use :func:`matplotlib.pyplot.plot` or 
        :func:`matplotlib.pyplot.step` for plotting.

    plot_kwargs_all : dict, optional
        Dictionary of keyword arguments passed to the plotting function.

    plot_kwargs_per : list[dict], optional
        List of dictionaries of keyword arguments passed to the
        plot function. ``len(plot_kwargs_per)`` must be equal to the amount
        of datasets passed.

        Each dictionary corresponds to the respective datasets.

        If the same keyword is provided in ``plot_kwargs_all`` and
        ``plot_kwargs_per``, ``plot_kwargs_per`` takes priority.

    legend_kwargs : dict, optional
        Dictionary of keyword arguments passed to
        :func:`matplotlib.pyplot.legend`.

    xlabel, ylabel, colorbar_label : str, optional
        Optional labels added to the x-axis / y-axis / colorbar.

    title : str, optional
        Optional title of the figure.

    xmin, xmax, ymin, ymax : float, optional
        Optionally fix the limits of the x/y axis.

    aspect_ratio : float, default 1.0 / *golden ratio*
        Aspect ratio  (height/width) of the plot.

        Ignored, if both ``axes_width_inch`` and
        ``axes_height_inch`` are provided.

    axes_width_inch : float, optional, default 3.15 inch
        Physical width of the axes in inches.
        If not provided, it will be determined automatically.

    axes_height_inch : float, optional
        Physical height of the axes in inches.
        If not provided, it will be determined automatically.

    make_me_nice : bool, default ``True``
        Optimize white space in the figure. See docstring of
        :func:`.make_me_nice` for details.

    make_me_nice_kwargs : dict, optional
        Dictionary of keyword arguments passed to :func:`.make_me_nice`.

    Returns
    -------
    figure : :class:`matplotlib.figure.Figure`

    axes : :class:`matplotlib.axes.Axes`
    """
    fig, ax = plt.subplots(1, 1)

    if plot_kwargs_per is not None and len(plot_kwargs_per) != len(datasets):
        msg = (
            "length of 'kwargs_per' is not the same length as the number of provided datasets: "
            f"{len(plot_kwargs_per)=}, {len(datasets)=}"
        )
        raise ValueError(msg)

    if plot_kwargs_per is not None and len(plot_kwargs_per) != len(datasets):
        raise ValueError(f"Invalid shape of {plot_kwargs_per=}")

    lines_have_labels = False
    for i, dataset in enumerate(datasets):
        if plot_kwargs_per is not None:
            kwargs = plot_kwargs_all | plot_kwargs_per[i]
        else:
            kwargs = plot_kwargs_all

        if "label" in kwargs:
            lines_have_labels = True

        if plot_function == "plot":
            ax.plot(*dataset, **kwargs)
        elif plot_function == "step":
            ax.step(*dataset, **kwargs)
        else:
            msg = f"{plot_function=}, but it must be 'plot' or 'step'"
            raise ValueError(msg)

    if lines_have_labels:
        ax.legend(**legend_kwargs)

    _format_axes(ax, xlabel, ylabel, title, xmin, xmax, ymin, ymax,
                 aspect_ratio, axes_width_inch, axes_height_inch)

    if make_me_nice:
        if axes_width_inch is not None or axes_height_inch is not None:
            make_me_nice_kwargs.setdefault("fix_figwidth", False)
        _plotting.make_me_nice(**make_me_nice_kwargs)

    return fig, ax


@overload
def create_2d_plot(
    *data: ArrayLike,
    plot_kwargs: dict[str, Any] = {},
    add_colorbar: Literal[True] = True,
    colorbar_kwargs: dict[str, Any] = {},
    colorbar_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    aspect_ratio: float = 1.0,
    axes_width_inch: Optional[float] = 80.0 / MM_PER_INCH / GOLDENRATIO,
    axes_height_inch: Optional[float] = None,
    make_me_nice: bool = True,
    make_me_nice_kwargs: dict[str, Any] = {},
) -> tuple[Figure, Axes, Colorbar]: ...


@overload
def create_2d_plot(
    *data: ArrayLike,
    plot_kwargs: dict[str, Any] = {},
    add_colorbar: Literal[False] = False,
    colorbar_kwargs: dict[str, Any] = {},
    colorbar_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    aspect_ratio: float = 1.0,
    axes_width_inch: Optional[float] = 80.0 / MM_PER_INCH / GOLDENRATIO,
    axes_height_inch: Optional[float] = None,
    make_me_nice: bool = True,
    make_me_nice_kwargs: dict[str, Any] = {},
) -> tuple[Figure, Axes]: ...


def create_2d_plot(
    *data: ArrayLike,
    plot_kwargs: dict[str, Any] = {},
    add_colorbar: bool = True,
    colorbar_kwargs: dict[str, Any] = {},
    colorbar_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    aspect_ratio: float = 1.0,
    axes_width_inch: Optional[float] = 60.0 / MM_PER_INCH,
    axes_height_inch: Optional[float] = None,
    make_me_nice: bool = True,
    make_me_nice_kwargs: dict[str, Any] = {},
) -> Union[tuple[Figure, Axes],
           tuple[Figure, Axes, Colorbar]]:
    """
    Plot 2D data.

    Parameters
    ----------
    data : (x, y, z) or (image, extents)
        Data to be plotted.

        If three arrays are passed, :func:`.matplotlib.pyplot.pcolormesh`
        will be used as backend.

        If two arrays are passed, :func:`.matplotlib.pyplot.imshow`
        will be used as backend.

    plot_kwargs : dict, optional
        Dictionary of keyword arguments passed to either ``pcolormesh``
        or ``imshow``.

    add_colorbar : bool, default ``True``
        Add a colorbar to the plot. Positioning and properties are controlled
        with ``colorbar_kwargs``. Uses :func:`.add_colorbar`.

    colorbar_kwargs : dict, optional
        Dictionary of keyword arguments passed to :func:`.add_colorbar`.

    xlabel, ylabel, colorbar_label : str, optional
        Optional labels added to the x-axis / y-axis / colorbar.

    title : str, optional
        Optional title of the figure.

    xmin, xmax, ymin, ymax : float, optional
        Optionally fix the limits of the x/y axis.

    aspect_ratio : float, default 1.0
        Aspect ratio  (height/width) of the plot.

        Ignored, if both ``axes_width_inch`` and
        ``axes_height_inch`` are provided.

    axes_width_inch : float, optional, default 2.36 inch
        Physical width of the axes in inches.
        If not provided, it will be determined automatically.

    axes_height_inch : float, optional
        Physical height of the axes in inches.
        If not provided, it will be determined automatically.

    make_me_nice : bool, default ``True``
        Optimize white space in the figure. See docstring of
        :func:`.make_me_nice` for details.

    make_me_nice_kwargs : dict, optional
        Dictionary of keyword arguments passed to :func:`.make_me_nice`.

    Returns
    -------
    figure : :class:`matplotlib.figure.Figure`

    axes : :class:`matplotlib.axes.Axes`

    colorbar : :class:`matplotlib.colorbar.Colorbar`
        Only returned if ``add_colorbar == True``.
    """
    fig, ax = plt.subplots(1, 1)

    plot_kwargs.setdefault("cmap", "atom")
    if len(data) == 2:
        plot_kwargs.setdefault("aspect", "auto")
        im = plt.imshow(data[0], extent=data[1], **plot_kwargs)  # type: ignore
    elif len(data) == 3:
        plot_kwargs.setdefault("rasterized", True)
        im = plt.pcolormesh(data[0], data[1], data[2], **plot_kwargs)
    else:
        msg = f"invalid shape of data"
        raise ValueError(msg)

    _format_axes(ax, xlabel, ylabel, title, xmin, xmax, ymin, ymax,
                 aspect_ratio, axes_width_inch, axes_height_inch)

    if add_colorbar:
        cb = _plotting.add_colorbar(im, ax, **colorbar_kwargs)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)

    if make_me_nice:
        if axes_width_inch is not None or axes_height_inch is not None:
            make_me_nice_kwargs.setdefault("fix_figwidth", False)
        _plotting.make_me_nice(**make_me_nice_kwargs)

    if add_colorbar:
        return fig, ax, cb
    else:
        return fig, ax

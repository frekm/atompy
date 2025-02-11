import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
import numpy as np
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

PALETTE_OKABE_ITO = (
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
    "#000000",
)

PALETTE_OKABE_ITO_ACCENT = (
    "#D9CBBE",
    "#C3CDD6",
    "#CAB9C1",
    "#F0EDD6",
    "#044F7E",
    "#026D4E",
    "#026D4E",
)


def set_color_cycle(
    *colors: str,
    nsteps: int = 7,
) -> None:
    """
    Set the color cycle in plots.

    Modifies ``rcParams`` of matplotlib
    (see `here <https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file>`__).

    Parameters
    ----------
    *colors : str, optional
        Colors seperated by comma and given in HEX-codes.

        If now colors are provided, defaults to Okabe and Ito palette
        (see `https://jfly.uni-koeln.de/color/`_).

        Alternatively, the name of a colormap can be specified and the color
        cycle picks ``nsteps`` colors from that colormap.

        See `https://matplotlib.org/stable/users/explain/colors/colormaps.html`_
        for available colormaps.

    nsteps : int, default 7
        Define how many different colors will be set in the color cycler.

        Irrelevant if a specific colors are passed in ``colors``.

    Examples
    --------

    .. code:: python
        # Set color cycle to red, green, blue
        set_color_cycle("#FF0000", "#00FF00", "#0000FF")

        # Set color cycle to Okabe and Ito palette
        set_color_cycle()

        # Set color cycle to a continuous colormap that repeats after 8 colors
        set_color_cylce(cmap="viridis", nsteps=8)

        # Set color cycle to a diverging colormap that passes through zero
        # with the second color
        set_color_cylce("RdBu", nsteps=3)
    """
    if not colors:
        if nsteps > 7:
            msg = (
                "For Okabe and Ito palette, a maximum of 7 distinct colors can "
                f"be used, but {nsteps=}"
            )
            raise ValueError(msg)
        colors = PALETTE_OKABE_ITO[:nsteps]

    if len(colors) == 1 and colors[0] in plt.colormaps():
        cmap_ = plt.get_cmap(colors[0])
        colors = tuple(
            [mcolors.to_hex(cmap_(i / (nsteps-1))) for i in range(nsteps)])

    cycler_str = "cycler('color', ["
    for color in colors:
        if color[0] == "#":
            color = color[1:]
        cycler_str += f"'{color}', "
    cycler_str += "])"
    plt.rcParams["axes.prop_cycle"] = cycler_str


def set_theme_latex_backend(
    font: Literal["FiraSans"]
) -> None:
    if font == "FiraSans":
        plt.rcParams["backend"] = "pgf"
        plt.rcParams["pgf.texsystem"] = "lualatex"
        plt.rcParams["pgf.rcfonts"] = False
        plt.rcParams["pgf.preamble"] = r"\usepackage[mathrm=sym]{unicode-math}\setmathfont{Fira Math}[Scale=MatchUppercase,Numbers=Tabular]\setsansfont{Fira Sans}[Scale=MatchUppercase,Numbers=Lining]\usepackage{picture,xcolor}\usepackage{nicefrac}"


def set_ticks_tight() -> None:
    """
    Reduce padding between ticks, ticklabels, and axis labels.
    """
    plt.rcParams["xtick.major.pad"] = 1.0
    plt.rcParams["xtick.minor.pad"] = 0.9
    plt.rcParams["ytick.major.pad"] = 1.0
    plt.rcParams["ytick.minor.pad"] = 0.9
    plt.rcParams["axes.labelpad"] = 2.0
    plt.rcParams["axes.titlepad"] = 3.0


def _set_theme_atompy(
    spines: str = "",
    use_latex: bool = False,
    fontsize: float = 10.0,
):
    spines_sort = "".join(sorted(spines))
    valid_spines = (
        "b", "l", "r", "t", "bl", "br", "bt", "blr", "blt", "brt", "blrt", ""
    )
    if spines_sort not in valid_spines:
        msg = (
            "invalid spines identifier. spines must only contain b, l, r, or t "
            "a maximum of one time"
        )
        raise ValueError(msg)

    plt.style.use("default")

    set_color_cycle()

    if use_latex:
        plt.rcParams["font.size"] = fontsize
        set_theme_latex_backend(font="FiraSans")
    else:
        plt.rcParams["font.size"] = fontsize * 0.9

    plt.rcParams["figure.figsize"] = 80.0 / MM_PER_INCH, 60.0 / MM_PER_INCH
    plt.rcParams["figure.dpi"] = 300

    plt.rcParams["axes.spines.left"] = True if "l" in spines else False
    plt.rcParams["axes.spines.bottom"] = True if "b" in spines else False
    plt.rcParams["axes.spines.top"] = True if "t" in spines else False
    plt.rcParams["axes.spines.right"] = True if "r" in spines else False

    plt.rcParams["xtick.major.size"] = 3.5# 0.0 if "l" in spines else 3.5
    plt.rcParams["ytick.major.size"] = 3.5#0.0 if "b" in spines else 3.5

    plt.rcParams["xtick.major.pad"] = 1.8
    plt.rcParams["xtick.minor.pad"] = 1.5
    plt.rcParams["ytick.major.pad"] = 1.8
    plt.rcParams["ytick.minor.pad"] = 1.5
    plt.rcParams["axes.labelpad"] = 2.0
    plt.rcParams["axes.titlepad"] = 3.0

    plt.rcParams["axes.grid"] = True

    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = plt.rcParams["axes.linewidth"]
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.major.width"] = plt.rcParams["axes.linewidth"]
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["lines.linewidth"] = 2.0

    plt.rcParams["axes.titlelocation"] = "left"

    plt.rcParams["grid.color"] = "#E2E2E2"
    plt.rcParams["grid.alpha"] = 0.7 
    plt.rcParams["axes.edgecolor"] = "#AFAFAF"
    plt.rcParams["xtick.color"] = plt.rcParams["grid.color"]
    plt.rcParams["ytick.color"] = plt.rcParams["grid.color"]
    plt.rcParams["xtick.labelcolor"] = "k"
    plt.rcParams["ytick.labelcolor"] = "k"

    plt.rcParams["grid.linewidth"] = plt.rcParams["axes.linewidth"]

    plt.rcParams["legend.frameon"] = False

    plt.rcParams["image.cmap"] = "atom"
    plt.rcParams["image.aspect"] = "auto"


def set_theme_science(
    figwidth_inch: float = FIGURE_WIDTH_SCIENCE_1COL,
    reset_rcParams: bool = False
) -> None:
    """
    Adjust ``rcParams`` to fit requirements for Science figures.

    See also `https://www.science.org/content/page/instructions-preparing-initial-manuscript#preparation-of-figures`_
    and `https://www.science.org/do/10.5555/page.2385607/full/author_figure_prep_guide_2022-1738682509963.pdf`_.

    Parameters
    ----------
    figwidth_inch : float, default 2.25 inch
        Figure width. Default is 1-column width. See also
        :data:`.FIGURE_WIDTH_SCIENCE_1COL`, 
        :data:`.FIGURE_WIDTH_SCIENCE_2COL`, and
        :data:`.FIGURE_WIDTH_SCIENCE_3COL`.

    reset_rcParams : bool, default ``False``.
        Reset rcParams to default values
        (see `here <https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file>`__).
    """
    if reset_rcParams:
        plt.style.use("default")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 7.0
    plt.rcParams["font.sans-serif"] = "Helvetica, Arial, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Avant Garde, sans-serif"

    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.5

    plt.rcParams["figure.figsize"] = figwidth_inch, 3./4. * figwidth_inch
    plt.rcParams["figure.dpi"] = 300

    set_color_cycle()
    set_ticks_tight()


def set_theme_prl(
    figwidth_inch: float = FIGURE_WIDTH_PRL_1COL,
    reset_rcParams: bool = False
) -> None:
    """
    Adjust ``rcParams`` to fit requirements for Physical Review figures.

    See also `https://journals.aps.org/authors/style-basics#figures`_.

    Parameters
    ----------
    figwidth_inch : float, default 2.25 inch
        Figure width. Default is 1-column width. See also
        :data:`.FIGURE_WIDTH_PRL_1COL` and
        :data:`.FIGURE_WIDTH_PRL_2COL`.

    reset_rcParams : bool, default ``False``.
        Reset rcParams to default values
        (see `here <https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file>`__).
    """
    if reset_rcParams:
        plt.style.use("default")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["font.serif"] = "STIXGeneral, DejaVu Serif, Bitstream Vera Serif, Computer Modern Roman, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.5

    plt.rcParams["figure.figsize"] = figwidth_inch, 3./4. * figwidth_inch
    plt.rcParams["figure.dpi"] = 300

    set_color_cycle()
    set_ticks_tight()


def set_theme_nature(
    figwidth_inch: float = FIGURE_WIDTH_NATURE_1COL,
    reset_rcParams: bool = False
) -> None:
    """
    Adjust ``rcParams`` to fit requirements for Nature figures.

    See also `https://www.nature.com/nature/for-authors/formatting-guide`_
    and `https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/`_.

    Parameters
    ----------
    figwidth_inch : float, default 2.25 inch
        Figure width. Default is 1-column width. See also
        :data:`.FIGURE_WIDTH_NATURE_1COL` and
        :data:`.FIGURE_WIDTH_NATURE_2COL`.

    reset_rcParams : bool, default ``False``.
        Reset rcParams to default values
        (see `here <https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file>`__).
    """
    if reset_rcParams:
        plt.style.use("default")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 7.0
    plt.rcParams["font.sans-serif"] = "Helvetica, Arial, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Avant Garde, sans-serif"

    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.5

    plt.rcParams["figure.figsize"] = figwidth_inch, 3./4. * figwidth_inch
    plt.rcParams["figure.dpi"] = 300

    set_color_cycle()
    set_ticks_tight()


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
    labels: Optional[Union[str, Sequence[str]]] = None,
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

    labels : array_like, optional
        Labels of each dataset. Length must match the amounts of datasets
        passed.

        If labels are also passed in the ``plot_kwargs_all/per`` dictionaries,
        those will take precedence.

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

    Examples
    --------

    .. plot:: _examples/create_1D_plot.py
        :include-source:
    """
    fig, ax = plt.subplots(1, 1)

    labels_provided_directly = False
    if labels is not None:
        labels_provided_directly = True
        if isinstance(labels, str):
            labels_ = np.array([labels])
        else:
            labels_ = np.asarray(labels)
        if len(labels_) != len(datasets):
            msg = (
                "length of 'kwargs_per' is not the same length as the number of provided datasets: "
                f"{len(labels_)=}, {len(datasets)=}"
            )
            raise ValueError(msg)

    if plot_kwargs_per is not None and len(plot_kwargs_per) != len(datasets):
        msg = (
            "length of 'kwargs_per' is not the same length as the number of provided datasets: "
            f"{len(plot_kwargs_per)=}, {len(datasets)=}"
        )
        raise ValueError(msg)

    if plot_kwargs_per is not None and len(plot_kwargs_per) != len(datasets):
        raise ValueError(f"Invalid shape of {plot_kwargs_per=}")

    lines_have_labels = False

    if plot_kwargs_per is None:
        plot_kwargs_per = []
        for i in range(len(datasets)):
            plot_kwargs_per.append({})

    for i, dataset in enumerate(datasets):
        if labels_provided_directly:
            plot_kwargs_per[i].setdefault("label", labels_[i])
        kwargs = plot_kwargs_all | plot_kwargs_per[i]

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

    Examples
    --------

    Using :func:`matplotlib.pyplot.pcolormesh` backend:

    .. plot:: _examples/create_2D_plot_pcolormesh.py
        :include-source:

    Using :func:`matplotlib.pyplot.imshow` backend:

    .. plot:: _examples/create_2D_plot_imshow.py
        :include-source:

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

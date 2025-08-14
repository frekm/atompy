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
import cycler

FIGURE_WIDTH_PRL_1COL = 3.0 + 3.0 / 8.0
FIGURE_WIDTH_PRL_2COL = 2.0 * FIGURE_WIDTH_PRL_1COL
FIGURE_WIDTH_NATURE_1COL = 90.0 / MM_PER_INCH
FIGURE_WIDTH_NATURE_2COL = 180.0 / MM_PER_INCH
FIGURE_WIDTH_SCIENCE_1COL = 2.25
FIGURE_WIDTH_SCIENCE_2COL = 4.75
FIGURE_WIDTH_SCIENCE_3COL = 7.25
GOLDENRATIO = 1.618033988749

PALETTE_OKABE_ITO = (
    "#56b4e9",
    "#e69f00",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
    "#000000",
)

PALETTE_OKABE_ITO_MUTE = (
    "#D9CBBE",
    "#C3CDD6",
    "#CAB9C1",
    "#F0EDD6",
)

PALETTE_OKABE_ITO_ACCENT = (
    "#044F7E",
    "#954000",
    "#026D4E",
)

PALETTE_COLORBREWER_DARK2 = (
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
)

PALETTE_COLORBREWER_MUTE = (
    "#7FC97F",
    "#BEAED4",
    "#FDC086",
    "#FFFF99",
)
PALETTE_COLORBREWER_ACCENT = (
    "#386CB0",
    "#F0027F",
    "#BF5B17",
)


def set_color_cycle(
    *colors: str,
    nsteps: int = 7,
    fig: Optional[Figure] = None,
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
        (see `here <https://jfly.uni-koeln.de/color/>`__ for a motivation).

        Alternatively, the name of a colormap can be specified and the color
        cycle picks ``nsteps`` colors from that colormap.

        See `matplolib's colormaps documenation <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`__
        for available colormaps.

    nsteps : int, default 7
        Define how many different colors will be set in the color cycler.

        Irrelevant if a specific colors are passed in ``colors``.

    fig : :class:`matplotlib.figure.Figure`, optional
        Optionally, provide a figure. The color cycle of all axes of that figure
        will be updated.

        If ``None``, check if a figure already exists. If so, update the color
        cycle of all axes of the last active figure.

    Notes
    -----
    For some color palettes included in `atompy`,
    see :ref:`Color Palettes <constants palettes>`.

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
    # format colors appropriately
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
        colors = tuple([mcolors.to_hex(cmap_(i / (nsteps - 1))) for i in range(nsteps)])

    # no figure passed, but a figure exists
    if fig is None and plt.get_fignums():
        fig = plt.gcf()

    # some axes exist already, update the cycler for them
    if fig is not None and fig.get_axes():
        axs = fig.get_axes()
        color_cycler = cycler.cycler(color=colors)
        for ax in axs:
            ax.set_prop_cycle(color_cycler)

    # update rcParams so future axes will use this one
    cycler_str = "cycler('color', ["
    for color in colors:
        if color[0] == "#":
            color = color[1:]
        cycler_str += f"'{color}', "
    cycler_str += "])"
    plt.rcParams["axes.prop_cycle"] = cycler_str


def set_latex_backend(font: Literal["FiraSans", "Times", "ScholaX"]) -> None:
    """
    Enable a latex backend for rendering figures.

    Must be called before figure creation.

    Parameters
    ----------
    font : {``"FiraSans"``, ``"Times"``, ``"ScholaX"``}
        Choose a font. ``"FiraSans"`` uses ``lualatex`` (which is very slow)
        as backend, the other fonts use ``pdflatex`` (not as slow).
    """
    plt.rcParams["pgf.rcfonts"] = False
    plt.rcParams["backend"] = "pgf"

    if font == "FiraSans":
        plt.rcParams["pgf.texsystem"] = "lualatex"
        plt.rcParams["pgf.preamble"] = (
            r"\usepackage[mathrm=sym]{unicode-math}\setmathfont{Fira Math}[Scale=MatchUppercase,Numbers=Tabular]\setsansfont{Fira Sans}[Scale=MatchUppercase,Numbers=Tabular]\usepackage{picture,xcolor}\usepackage{nicefrac}"
        )
    elif font == "Times":
        plt.rcParams["pgf.texsystem"] = "pdflatex"
        plt.rcParams["pgf.preamble"] = (
            r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}\usepackage{picture,xcolor}\usepackage{nicefrac}"
        )
    elif font == "ScholaX":
        plt.rcParams["pgf.texsystem"] = "pdflatex"
        plt.rcParams["pgf.preamble"] = (
            r"\usepackage{scholax}\usepackage{amsmath,amsthm}\usepackage[scaled=1.075,ncf,vvarbb]{newtxmath}\usepackage{picture,xcolor}\usepackage{nicefrac}"
        )


def _set_ticks_tight() -> None:
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
    spines_color: Union[Literal["grid"], str] = "#AFAFAF",
    use_latex: bool = True,
    fontsize: float = 10.0,
    use_serif: bool = True,
):
    """
    Sets a theme me likes.

    .. attention::

        The default behavior of this function may change without further notice.
        Do not assume that the behavior will stay constant over multiple
        versions of ``atompy``.

        If the behavior changes, however, it will be listed in the
        `changelog <https://github.com/frekm/atompy/blob/main/changelog.md>`__.

    Parameters
    ----------
    spines : str, default ""
        Any combination of ``"l"``, ``"b"``, ``"t"``, ``"r"``.
        E.g., ``"lb"`` would enable the left and bottom spines of the axes.
        ``""`` disables all spines.

    use_latex : bool, default ``True``
        Use a latex backend for rendering. May be slow.

    fontsize : float, default 10 pts

    use_serif : bool, default ``True``
        Use a font-family with serifs. If ``False``, use a sans-serif font
        family.

    Examples
    --------

    .. plot:: _examples/set_theme_atompy.py
        :include-source:

    """
    spines_sort = "".join(sorted(spines))
    valid_spines = (
        "b",
        "l",
        "r",
        "t",
        "bl",
        "br",
        "bt",
        "blr",
        "blt",
        "brt",
        "blrt",
        "",
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
        if use_serif:
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["font.family"] = "serif"
            set_latex_backend(font="ScholaX")
        else:
            plt.rcParams["font.size"] = fontsize
            set_latex_backend(font="FiraSans")
    else:
        if use_serif:
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = (
                "STIXGeneral, DejaVu Serif, Bitstream Vera Serif, Computer Modern Roman, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif"
            )
            plt.rcParams["mathtext.fontset"] = "stix"
        else:
            plt.rcParams["font.size"] = fontsize * 0.9

    plt.rcParams["figure.figsize"] = 80.0 / MM_PER_INCH, 60.0 / MM_PER_INCH
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["savefig.format"] = "pdf"

    plt.rcParams["axes.spines.left"] = True if "l" in spines else False
    plt.rcParams["axes.spines.bottom"] = True if "b" in spines else False
    plt.rcParams["axes.spines.top"] = True if "t" in spines else False
    plt.rcParams["axes.spines.right"] = True if "r" in spines else False

    plt.rcParams["xtick.major.pad"] = 1.8
    plt.rcParams["xtick.minor.pad"] = 1.5
    plt.rcParams["ytick.major.pad"] = 1.8
    plt.rcParams["ytick.minor.pad"] = 1.5
    plt.rcParams["axes.labelpad"] = 2.0
    plt.rcParams["axes.titlepad"] = 5.0

    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "both"
    plt.rcParams["axes.grid.which"] = "major"

    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.linewidth"] = plt.rcParams["axes.linewidth"]

    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["lines.markersize"] = 3.0
    plt.rcParams["errorbar.capsize"] = plt.rcParams["lines.markersize"]

    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5
    plt.rcParams["xtick.minor.size"] = 2.0
    plt.rcParams["ytick.minor.size"] = 2.0
    plt.rcParams["xtick.major.width"] = plt.rcParams["axes.linewidth"]
    plt.rcParams["ytick.major.width"] = plt.rcParams["axes.linewidth"]
    plt.rcParams["xtick.minor.width"] = 0.4
    plt.rcParams["ytick.minor.width"] = 0.4

    plt.rcParams["axes.titlelocation"] = "left"

    c_grid = "#E2E2E2"
    c_spines = c_grid if spines_color == "grid" else spines_color

    plt.rcParams["grid.color"] = c_grid
    plt.rcParams["grid.alpha"] = 1.0
    plt.rcParams["axes.edgecolor"] = c_spines
    plt.rcParams["xtick.color"] = c_spines if "b" in spines else c_grid
    plt.rcParams["ytick.color"] = c_spines if "l" in spines else c_grid
    plt.rcParams["xtick.labelcolor"] = "k"
    plt.rcParams["ytick.labelcolor"] = "k"

    plt.rcParams["grid.linewidth"] = plt.rcParams["axes.linewidth"]

    plt.rcParams["legend.frameon"] = False

    plt.rcParams["image.cmap"] = "atom"
    plt.rcParams["image.aspect"] = "auto"


def _create_plot_format_axes(
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
        _plotting.set_axes_size(axes_height_inch / aspect_ratio, axes_height_inch, ax)
    elif axes_width_inch is not None and axes_height_inch is None:
        _plotting.set_axes_size(axes_width_inch, axes_width_inch * aspect_ratio, ax)
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

    _create_plot_format_axes(
        ax,
        xlabel,
        ylabel,
        title,
        xmin,
        xmax,
        ymin,
        ymax,
        aspect_ratio,
        axes_width_inch,
        axes_height_inch,
    )

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
) -> Union[tuple[Figure, Axes], tuple[Figure, Axes, Colorbar]]:
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

    plot_kwargs_ = plot_kwargs.copy()
    plot_kwargs_.setdefault("cmap", "atom")
    if len(data) == 2:
        plot_kwargs_.setdefault("aspect", "auto")
        im = plt.imshow(data[0], extent=data[1], **plot_kwargs_)  # type: ignore
    elif len(data) == 3:
        plot_kwargs_.setdefault("rasterized", True)
        im = plt.pcolormesh(data[0], data[1], data[2], **plot_kwargs_)
    else:
        msg = f"invalid shape of data"
        raise ValueError(msg)

    _create_plot_format_axes(
        ax,
        xlabel,
        ylabel,
        title,
        xmin,
        xmax,
        ymin,
        ymax,
        aspect_ratio,
        axes_width_inch,
        axes_height_inch,
    )

    if add_colorbar:
        cb = _plotting.add_colorbar(im, ax, **colorbar_kwargs)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)

    if make_me_nice:
        if axes_width_inch is not None or axes_height_inch is not None:
            make_me_nice_kwargs_ = make_me_nice_kwargs.copy()
            make_me_nice_kwargs_.setdefault("fix_figwidth", False)
        _plotting.make_me_nice(**make_me_nice_kwargs_)

    if add_colorbar:
        return fig, ax, cb
    else:
        return fig, ax

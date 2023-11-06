from __future__ import annotations
import time
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike, NDArray
import uproot
from dataclasses import dataclass
from typing import (Sequence, Union, Any, TypeVar, overload,
                    Literal, Generic, Optional, NamedTuple)
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolor
import matplotlib.cm as mplcm
import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import matplotlib.text as mpltxt
import matplotlib.colorbar as mplcb

PTS_PER_INCH = 72.0
"""float: 72 pts = 1 inch"""

MM_PER_INCH = 25.4
"""float: 25.4 mm = 1 inch"""

T = TypeVar('T')
DType = TypeVar("DType")

RED = "#AE1117"
TEAL = "#008081"
BLUE = "#2768F5"
GREEN = "#007F00"
GREY = "#404040"
ORANGE = "#FD8D3C"
PINK = "#D4B9DA"
YELLOW = "#FCE205"
LEMON = "#EFFD5F"
CORN = "#E4CD05"
PURPLE = "#CA8DFD"
DARK_PURPLE = "#9300FF"
FOREST_GREEN = "#0B6623"
BRIGHT_GREEN = "#3BB143"

###############################################################################
###############################################################################
###############################################################################
# Plotting
###############################################################################
###############################################################################
###############################################################################


class _Colors(NamedTuple):
    red: Literal["#ae1117"]
    blue: Literal["#2768f5"]
    orange: Literal["#fd8d3c"]
    pink: Literal["#d4b9da"]
    green: Literal["#007f00"]
    teal: Literal["#008081"]
    grey: Literal["#404040"]
    yellow: Literal["#fce205"]
    lemon: Literal["#effd5f"]
    corn: Literal["#e4cd05"]
    purple: Literal["#ca8dfd"]
    dark_purple: Literal["#9300ff"]
    forest_green: Literal["#0b6623"]
    bright_green: Literal["#3bb143"]


colors = _Colors(
    RED,
    BLUE,
    ORANGE,
    PINK,
    GREEN,
    TEAL,
    GREY,
    YELLOW,
    LEMON,
    CORN,
    PURPLE,
    DARK_PURPLE,
    FOREST_GREEN,
    BRIGHT_GREEN
)

_cm_whitered_dict = {
    'red': ((0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0)),
    'green': ((0.0, 1.0, 1.0),
              (1.0, 128. / 255., 128. / 255.)),
    'blue': ((0.0, 1.0, 1.0),
             (1.0, 129. / 255., 129. / 255.))}
cm_whitered = mplcolor.LinearSegmentedColormap('whitered', _cm_whitered_dict)

_cm_lmf2root_dict = {
    'red': [[0.0, 0.0, 0.5],
            [0.3, 0.0, 0.0],
            [0.7, 1.0, 1.0],
            [1.0, 1.0, 1.0]],
    'green': [[0.0, 0.0, 1.0],
              [0.3, 0.0, 0.0],
              [0.7, 0.0, 0.0],
              [1.0, 1.0, 1.0]],
    'blue': [[0.0, 0.0, 1.0],
             [0.3, 1.0, 1.0],
             [0.7, 0.0, 0.0],
             [1.0, 0.0, 0.0]]}
cm_lmf2root = mplcolor.LinearSegmentedColormap(
    'lmf2root', _cm_lmf2root_dict)
mpl.colormaps.register(cm_lmf2root, force=True)

_cm_lmf2root_from_white_dict = {
    'red': ((0.0, 0.0, 1.0),
            (0.065, 0.5, 0.5),
            (0.3, 0.0, 0.0),
            (0.7, 1.0, 1.0),
            (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 1.0),
              (0.065, 1.0, 1.0),
              (0.3, 0.5, 0.5),
              (0.7, 0.0, 0.0),
              (1.0, 1.0, 1.0)),
    'blue': ((0.0, 0.0, 1.0),
             (0.065, 1.0, 1.0),
             (0.3, 1.0, 1.0),
             (0.7, 0.0, 0.0),
             (1.0, 0.0, 0.0))}
cm_lmf2root_from_white = mplcolor.LinearSegmentedColormap(
    'lmf2root_from_white', _cm_lmf2root_from_white_dict)
mpl.colormaps.register(cm_lmf2root_from_white, force=True)

_font_scalings = {
    'xx-small': 0.579,
    'x-small': 0.694,
    'small': 0.833,
    'medium': 1.0,
    'large': 1.200,
    'x-large': 1.440,
    'xx-large': 1.728,
    'larger': 1.2,
    'smaller': 0.833}


@dataclass
class FigureMargins:
    left: NDArray[np.float64]
    right: NDArray[np.float64]
    top: NDArray[np.float64]
    bottom: NDArray[np.float64]

    def __iter__(self):
        return [self.left, self.right, self.top, self.bottom].__iter__()

    def __getitem__(self, index: int) -> NDArray[np.float64]:
        return [self.left, self.right, self.top, self.bottom][index]


@dataclass
class FigureMarginsFloat:
    left: float
    right: float
    top: float
    bottom: float

    def __iter__(self):
        return [self.left, self.right, self.top, self.bottom].__iter__()

    def __getitem__(self, index: int) -> float:
        return [self.left, self.right, self.top, self.bottom][index]

    def __setitem__(self, index: int, value: float) -> None:
        if index == 0:
            self.left = value
        elif index == 1:
            self.right = value
        elif index == 2:
            self.top = value
        elif index == 3:
            self.bottom = value
        else:
            raise IndexError


@dataclass
class FigureLayout:
    fig_width: float
    fig_height: float
    nrows: int
    ncols: int
    axes_x0s: NDArray[np.float64]
    axes_y0s: NDArray[np.float64]
    axes_x1s: NDArray[np.float64]
    axes_y1s: NDArray[np.float64]
    axes_widths: NDArray[np.float64]
    axes_heights: NDArray[np.float64]
    ratios: NDArray[np.float64]
    xpads: NDArray[np.float64]
    ypads: NDArray[np.float64]
    margins: FigureMargins


@dataclass
class Colorbar:
    colorbar: mplcb.Colorbar
    parent_axes: mplax.Axes
    location: str
    pad_inch: float
    width_inch: float

    @property
    def ax(self) -> mplax.Axes:
        return self.colorbar.ax


@dataclass
class ColorbarLarge:
    colorbar: mplcb.Colorbar
    parent_axes: tuple[mplax.Axes, mplax.Axes]
    location: str
    pad_inch: float
    width_inch: float

    @property
    def ax(self) -> mplax.Axes:
        return self.colorbar.ax


@dataclass
class _Regions:
    left: list
    right: list
    top: list
    bottom: list

    def __iter__(self):
        return [self.left, self.right, self.top, self.bottom].__iter__()


def flatten(
    input: Union[Sequence[T],
                 Sequence[Sequence[T]]]
) -> list[T]:
    """
    Flatten a Sequence of Sequences

    Parameters
    ----------
    input : Sequence[T] or Sequence of Seqence[T]

    Returns
    -------
    output : list[T]
        The flattened Sequence as a list

    Examples
    --------

    >>> flatten([[1, 2, 3], [4, 5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    output: list[T] = []
    for input_ in input:
        if isinstance(input_, str):
            output.append(input_)
            continue
        try:
            iter(input_)
        except TypeError:
            output.append(input_)
        else:
            output += flatten(input_)
    return output


def return_as_list(
    input: Union[T,
                 Sequence[T],
                 Sequence[Sequence[T]]],
    desired_length: int = 1
) -> list[T]:
    """
    Check if input is given as a Sequence. If not, return it as a list
    of length *desired_length*, otherwise return original list
    """
    if isinstance(input, str):
        return [input] * desired_length
    try:
        iter(input)
    except TypeError:
        return [input] * desired_length
    return flatten(input)


def get_column(n: int, lst: Sequence[Sequence[T]]) -> list[T]:
    """
    Get column *n* of a list of lists

    Parameters
    ----------
    n : int
        the column to get

    lst : list of lists
        each sublist should have the same length

    Returns
    -------
    list
        A list of the n-th element of each sublist

    Examples
    --------

    >>> get_column(0, [[1, 2, 3], [4, 5, 6]])
    [1, 4]
    >>> get_column(-1, [[1, 2, 3], [4, 5, 6]])
    [3, 6]
    """
    return [sublst[n] for sublst in lst]


def reshape(
    input: Sequence[T],
    rowscols: Sequence[int]
) -> list[list[T]]:
    """
    Reshape a Sequence to have rowscols[0] rows and rowscols[1] columns

    Parameters
    ----------
    input : Sequence

    rowscols : (number of rows, number of columns)

    Returns
    -------
    output : list of lists

    Examples
    --------

    >>> reshape([1, 2, 3, 4, 5, 6], (2, 3))
    [[1, 2, 3], [4, 5, 6]]
    """
    if rowscols[0] * rowscols[1] != len(input):
        raise ValueError(
            "reshaping with this *shape* won't work"
        )

    out: list[list[T]] = []
    for irow in range(0, len(input), rowscols[1]):
        out.append([elem for elem in input[irow:irow + rowscols[1]]])
    return out


def transpose(
    input: Sequence[Sequence[T]]
) -> Sequence[Sequence[T]]:
    """
    Transpose a 2D Sequence

    Parameters
    ----------
    input : Sequence[Sequence[T]]

    Returns
    -------
    output : list[list[T]]
        Transposed input
    """
    return list(map(list, zip(*input)))


def create_colormap(
    steps: Sequence[float],
    reds: Sequence[float],
    greens: Sequence[float],
    blues: Sequence[float],
    lut_size: int = 256,
    cmap_name: str = "MyCmap",
    register_cmap: bool = False
) -> mplcolor.LinearSegmentedColormap:
    """
    Create a colormap from a specified color Sequence

    Parameters
    ----------
    steps : Sequence floats
        the steps of the colorbar ranging from :code:`0.0` to :code:`1.0`

    reds/greens/blues : Sequence floats
        The corresponding value of the colors ranging from :code:`0.0` to
        :code:`1.0`

    n_colors : int, default :code:`256`
        number of different colors in the colormap

    cmap_name : str, default :code:`"MyCmap"`
        The name of the colormap instance

    Returns
    -------
    :code:`matplotlib.colors.LinearSegmentedColormap`

    Notes
    -----
    The colormap can be registered using

    >>> import matplotlib
    >>> cmap_name = "MyCmap"
    >>> cmap = create_colormap(*args, cmap_name=cmap_name)
    >>> matplotlib.colormaps.register(cmap)

    It can then be set as default via rcParams

    >>> matplotlib.rcParams["image.cmap"] = cmap_name

    Examples
    --------
    .. code-block:: python

        from atompy import create_colormap
        import matplotlib.pyplot as plt
        import numpy as np

        # create colormap starting white going through to red
        cmap = create_colormap([0.0, 1.0],
                               [1.0, 1.0],
                               [1.0, 0.0],
                               [1.0, 0.0])

        # create colormap starting blue going through white to red
        cmap = create_colormap([0.0, 0.5, 1.0],
                               [0.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0],
                               [1.0, 1.0, 0.0])

        # plottable with something like this
        image = np.arange(9).reshape((3, 3))
        plt.imshow(image, cmap=cmap)

    """
    if not all(len(steps) == len(l) for l in [reds, greens, blues]):
        raise ValueError(
            "lengths of 'steps', 'reds', 'greens', 'blues' need to be equal"
        )
    reds_ = [(0.0, 0.0, reds[0])]
    greens_ = [(0.0, 0.0, greens[0])]
    blues_ = [(0.0, 0.0, blues[0])]
    for step, r, g, b in zip(steps, reds, greens, blues):
        reds_.append((step, r, r))
        greens_.append((step, g, g))
        blues_.append((step, b, b))

    cm_dict = {"red": reds_, "green": greens_, "blue": blues_}
    cmap = mplcolor.LinearSegmentedColormap(
        cmap_name, cm_dict, N=lut_size)
    if register_cmap:
        mpl.colormaps.register(cmap)
    return cmap


def create_colormap_from_hex(
    steps: Sequence[float],
    colors: Sequence[str],
    lut_size: int = 256,
    cmap_name: str = "MyCmap"
) -> mplcolor.LinearSegmentedColormap:
    """
    Create a colormap from a color Sequence

    Parameters
    ----------
    steps : Sequence floats
        the steps of the colorbar ranging from :code:`0.0` to :code:`1.0`

    colors : Sequence str
        The corresponding colors given as hex codes, e.g., :code:`"#FF00FF"`

    n_colors : int, default :code:`256`
        number of different colors in the colormap

    cmap_name : str, default :code:`"MyCmap"`
        The name of the colormap instance

    Returns
    -------
    :code:`matplotlib.colors.LinearSegmentedColormap`

    Notes
    -----
    The colormap can be registered using

    >>> import matplotlib
    >>> cmap_name = "MyCmap"
    >>> cmap = create_colormap(*args, cmap_name=cmap_name)
    >>> matplotlib.colormaps.register(cmap)

    It can then be set as default via rcParams

    >>> matplotlib.rcParams["image.cmap"] = cmap_name

    Examples
    --------
    .. code-block:: python

        from atompy import create_colormap_from_hex
        import matplotlib.pyplot as plt
        import numpy as np

        # create colormap starting white going through to red
        cmap = create_colormap_from_hex([0.0, 1.0], ["#FFFFFF", "#FF0000"])

        # create colormap starting blue going through white to red
        cmap = create_colormap_from_hex([0.0, 0.5, 1.0],
                                        ["#0000FF", "#FFFFFF", "#FF0000"])

        # plottable with something like this
        image = np.arange(9).reshape((3, 3))
        plt.imshow(image, cmap=cmap)
    """
    reds, greens, blues = [], [], []
    for color in colors:
        reds.append(int(color[1:3], 16) / 255)
        greens.append(int(color[3:5], 16) / 255)
        blues.append(int(color[5:], 16) / 255)

    return create_colormap(steps, reds, greens, blues, lut_size=lut_size,
                           cmap_name=cmap_name)


def textwithbox(
    axes: mplax.Axes,
    x: float,
    y: float,
    text: str,
    pad: float = 1.0,
    boxbackground: Optional[str] = "white",
    boxedgecolor: str = "black",
    boxedgewidth: float = 0.5,
    **text_kwargs
) -> mpltxt.Text:
    """
    Plot text with matplotlib surrounded by a box. Only works with a
    latex backend

    Parameters
    ----------
    ax : `matplotlib.pyplot.axes`
        the axes

    x : float
        x-position

    y : float
        y-position

    text : str
        The text to be surrounded by the box

    pad : float, default: :code:`1.0` (in pts)
        padding between boxedge and text

    boxbackground : :code:`None`, :code:`False`, or str, \
default: :code:`"white"`
        background of box

        - :code:`None` or :code:`False`: No background color
        - str: latex xcolor named color

    boxedgecolor : str, optional, default: :code:`"black"`
        edge color using named color from latex package *xcolor*
        only used if boxbackground != None

    boxedgewidth : float, default :code:`0.5` (in pts)
        edgelinewidth of the box

    Returns
    -------
    `matplotlib.text.Text <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`_
        The created :code`matplotib.text.Text` instance

    Other Parameters
    ----------------
    **kwargs : `matpotlib.text.Text <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`_ \
properties
        Other miscellaneous text parameters
    """
    sep = r"\setlength{\fboxsep}{%lfpt}" % pad
    rule = r"\setlength{\fboxrule}{%lfpt}" % boxedgewidth
    if boxbackground is not None:
        text = r"%s\fcolorbox{%s}{%s}{%s}" % (sep + rule, boxedgecolor,
                                              boxbackground, text)
    else:
        text = r"%s\fbox{%s}" % (sep + rule, text)
    return axes.text(x, y, text, **text_kwargs)


class AliasError(Exception):
    def __init__(self,
                 keyword_arg: str,
                 alias: str):
        self.keyword_arg = keyword_arg
        self.alias = alias

    def __str__(self):
        return (f"Both '{self.keyword_arg}' and '{self.alias}' have been "
                "provided, but they are aliases")


def _set_lw_fs_lh(
    linewidth: Optional[float],
    fontsize: Optional[Union[float, str]],
    legend_handlelength: Optional[float],
    **aliases
) -> tuple[float, float, float]:
    """ Process parameters for dashed/dotted/... """
    # check if aliases are doubled
    if "lw" in aliases and linewidth is not None:
        raise AliasError("linewidth", "lw")
    if "lh" in aliases and legend_handlelength is not None:
        raise AliasError("legend_handlelength", "lh")

    lw = linewidth if linewidth else \
        aliases.get("lw", plt.rcParams["lines.linewidth"])
    lh = legend_handlelength if legend_handlelength else \
        aliases.get("lh", plt.rcParams["legend.handlelength"])
    fontsize_ = (fontsize if fontsize is not None
                 else plt.rcParams["legend.fontsize"])
    if isinstance(fontsize_, str):
        if fontsize_ in _font_scalings:
            fontsize_ = _font_scalings[fontsize_] * plt.rcParams["font.size"]
        else:
            raise ValueError("Invalid specifier for fontsize")

    return lw, fontsize_, lh


def dotted(
    linewidth: Optional[float] = None,
    fontsize: Optional[Union[float,
                             Literal["xx-small", "x-small", "small", "medium",
                                     "large", "x-large", "xx-large", "larger",
                                     "smaller"]
                             ]] = None,
    legend_handlelength: Optional[float] = None,
    **aliases
) -> tuple[float, tuple[float, float]]:
    """
    Return a ls tuple to create a dotted line that fits perfectly into a
    legend. For that to work properly you may need to provide the linewidth of
    the graph and the fontsize of the legend.

    Parameters
    ----------
    linewidth (or lw) : float, optional, default: rcParams["lines.linewidth"]

    fontsize : float or str, Optional, default: rcParams["legend.fontsize"]
        The fontsize used in the legend

        - float: fontsize in pts
        - str: :code:`"xx-small"`, :code:`"x-small"`, :code:`"small"`,
          :code:`"medium"`, :code:`"large"`, :code:`"x-large"`, 
          :code:`"xx-large"`, :code:`"larger"`, or :code:`"smaller"`

    legend_handlelength (or lh) : float, default \
:code:`rcParams["legend.handlelength"]`
        Length of the legend handles (the dotted line, in this case) in font
        units

    Returns
    -------
    tuple : (float, (float, float))
        tuple to be used as linetype in plotting
    """
    lw_, fs_, lh_ = _set_lw_fs_lh(
        linewidth, fontsize, legend_handlelength, **aliases)

    total_points = fs_ * lh_ / lw_
    n_dots = math.ceil(total_points / 2.0)
    spacewidth = (total_points - n_dots) / (n_dots - 1)

    return 0.0, (1.0, spacewidth)


def dash_dotted(
    ratio: float = 3.0,
    n_dashes: int = 3,
    linewidth: Optional[float] = None,
    fontsize: Optional[Union[float,
                             Literal["xx-small", "x-small", "small", "medium",
                                     "large", "x-large", "xx-large", "larger",
                                     "smaller"]
                             ]] = None,
    legend_handlelength: Optional[float] = None,
    **aliases
) -> tuple[float, tuple[float, float, float, float]]:
    """
    Return a ls tuple to create a dash-dotted line that fits perfectly into a
    legend. For that to work properly you may need to provide the linewidth of
    the graph and the fontsize of the legend.

    Parameters
    ----------
    ratio : float, default: 3.0
        Ratio between dash-length and gap-length

    n_dashes : int, default: 3
        Number of dashes drawn

    linewidth (or lw): float, optional, default: `rcParams["lines.linewidth"]`

    fontsize : float or str, Optional, default: `rcParams["legend.fontsize"]`
        The fontsize used in the legend

        - float: fontsize in pts
        - str: :code:`"xx-small"`, :code:`"x-small"`, :code:`"small"`,
          :code:`"medium"`, :code:`"large"`, :code:`"x-large"`, 
          :code:`"xx-large"`, :code:`"larger"`, or :code:`"smaller"`

    legend_handlelength (or 'lh') : float, default \
:code:`rcParams["legend.handlelength"]`
        Length of the legend handles (the dotted line, in this case) in font
        units

    Returns
    -------
    tuple : (float, (float, float, float, float))
        tuple to be used as linetype in plotting
    """
    lw_, fs_, lh_ = _set_lw_fs_lh(
        linewidth, fontsize, legend_handlelength, **aliases)

    total_points = fs_ * lh_ / lw_
    spacewidth = (total_points - n_dashes) / \
                 (2.0 * n_dashes - 1 + n_dashes * ratio)
    dashwidth = ratio * spacewidth

    return 0.0, (dashwidth, spacewidth, 1.0, spacewidth)


def dashed(
    ratio: float = 1.5,
    n_dashes: int = 4,
    linewidth: Optional[float] = None,
    fontsize: Optional[Union[float,
                             Literal["xx-small", "x-small", "small", "medium",
                                     "large", "x-large", "xx-large", "larger",
                                     "smaller"]
                             ]] = None,
    legend_handlelength: Optional[float] = None,
    **aliases
) -> tuple[float, tuple[float, float]]:
    """
    Return a ls tuple to create a dashed line that fits perfectly into a
    legend. For that to work properly you may need to provide the linewidth of
    the graph and the fontsize of the legend.

    Parameters
    ----------
    ratio : float, default: 1.5
        Ratio between dash-length and gap-length

    n_dashes : int, default: 4
        Number of dashes drawn

    linewidth (or lw): float, optional, default: rcParams["lines.linewidth"]

    fontsize : float or str, Optional, default: rcParams["legend.fontsize"]
        The fontsize used in the legend

        - float: fontsize in pts
        - str: :code:`"xx-small"`, :code:`"x-small"`, :code:`"small"`,
          :code:`"medium"`, :code:`"large"`, :code:`"x-large"`, 
          :code:`"xx-large"`, :code:`"larger"`, or :code:`"smaller"`

    legend_handlelength (or lh): float, default \
:code:`rcParams["legend.handlelength"]`
        Length of the legend handles (the dotted line, in this case) in font
        units

    Returns
    -------
    (float, (float, float, float, float))
        tuple to be used as linetype in plotting
    """
    lw_, fs_, lh_ = _set_lw_fs_lh(
        linewidth, fontsize, legend_handlelength, **aliases)

    total_points = fs_ * lh_ / lw_

    n_gaps = n_dashes - 1
    spacewidth = total_points / (n_gaps + n_dashes * ratio)
    dashwidth = ratio * spacewidth

    return 0.0, (dashwidth, spacewidth)


def emarker(**kwargs) -> dict:
    """
    Use to change the default format of markers with errorbars

    Parameters
    ----------
    **kwargs : `matplotlib.lines.Line2D` properties
        These keyword parameters overwrite the default parameters of this
        fuction

    Returns
    -------
    dict
        A dictionary containing default parameters
    """
    rtn_dict = dict(kwargs)

    if not ("color" in kwargs or "c" in kwargs):
        rtn_dict["color"] = "k"

    if not ("markeredgecolor" in kwargs or "mec" in kwargs):
        try:
            color = rtn_dict["color"]
        except KeyError:
            color = rtn_dict["c"]
        rtn_dict["markeredgecolor"] = color

    rtn_dict.setdefault("elinewidth", 0.5)
    rtn_dict.setdefault("capthick", rtn_dict["elinewidth"])
    rtn_dict.setdefault("capsize", 1.5)
    rtn_dict.setdefault("marker", "o")

    if not ("markersize" in kwargs or "ms" in kwargs):
        rtn_dict["markersize"] = 2.0

    if not ("linestyle" in kwargs or "ls" in kwargs):
        rtn_dict["linestyle"] = None

    if not ("linewidth" in kwargs or "lw" in kwargs):
        rtn_dict["linewidth"] = 0.0

    return rtn_dict


def get_equal_tick_distance(
    limits: Sequence[float],
    n: int = 3
) -> NDArray[np.float_]:
    """
    Calculate ticks for *n_ticks* number of ticks from *limits[0]* to
    *limits[1]*

    Parameters
    ----------
    limits : (float, float)
        the minimum and maximum

    n : int, default: 3
        Number of equally spaced ticks

    Returns
    -------
    `numpy.ndarray`
        The values where the ticks need to be placed
    """
    spacing = (limits[1] - limits[0]) / (n - 1)
    return np.array([limits[0] + i * spacing for i in range(n)])


def equalize_xtick_distance(
        axes: mplax.Axes,
        n: int = 4,
        limits: Optional[Sequence[float]] = None
) -> NDArray[np.float_]:
    """
    Draw *n_ticks* number of ticks from 

    Parameters
    ----------
    ax : :code:`matplotlib.axes.Axes`
        The axes to change the xticks from

    n : int, default :code:`4`
        Number of equally spaced ticks

    limits : (float, float), optional
        the limits inbetween to equalize.

    Returns
    -------
    :code:`numpy.ndarray`
        The values where the ticks are placed
    """
    limits = limits or axes.get_xlim()
    ticks = get_equal_tick_distance(limits, n)
    axes.set_xticks(ticks)
    return ticks


def equalize_ytick_distance(
        axes: mplax.Axes,
        n: int = 4,
        limits: Optional[Sequence[float]] = None
) -> NDArray[np.float_]:
    """
    Draw *n_ticks* number of ticks from *limits[0]* to *limits[1]*

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The axes to change the yticks from

    n : int, default: :code:`4`
        Number of equally spaced ticks

    limits : (float, float), optional
        the limits inbetween to equalize

    Returns
    -------
    `numpy.ndarray`
        The values where the ticks are placed
    """
    limits = limits or axes.get_ylim()
    ticks = get_equal_tick_distance(limits, n)
    axes.set_yticks(ticks)
    return ticks


def get_figure_layout(
    figure: mplfig.Figure,
    axes: list[list[mplax.Axes]],
    unit: Literal["mm", "inch", "pts"]
) -> FigureLayout:
    """
    Get the layout of the *figure* with *axes*

    Paramters
    ---------
    figure : `matplotlib.figure.Figure`

    axes : list[list[`matplotlib.axes.Axes`]]

    unit : "mm", "inch", "pts"

    Returns
    -------
    `FigureLayout`
        Named tuple (fig_width, fig_height, nrows, ncols, axes_width,
        axes_height", xpad, ypad, margins)
    """
    if unit not in ["mm", "inch", "pts", "relative"]:
        raise ValueError(
            f"{unit=} needs to be either 'mm', 'inch' or 'pts' instead"
        )
    if unit == "mm":
        convert_factor = MM_PER_INCH
    elif unit == "pts":
        convert_factor = PTS_PER_INCH
    else:
        convert_factor = 1.0

    fw = 1.0 if unit == "relative" \
        else figure.get_size_inches()[0] * convert_factor
    fh = 1.0 if unit == "relative" \
        else figure.get_size_inches()[1] * convert_factor
    nrows, ncols = len(axes), len(axes[0])
    aw = np.empty((nrows, ncols))
    ah = np.empty((nrows, ncols))
    x0s = np.empty((nrows, ncols))
    y0s = np.empty((nrows, ncols))
    x1s = np.empty((nrows, ncols))
    y1s = np.empty((nrows, ncols))
    ratio = np.empty((nrows, ncols))
    for irow in range(nrows):
        for icol in range(ncols):
            x0s[irow, icol] = (axes[irow][icol].get_position().x0 * fw)
            y0s[irow, icol] = (axes[irow][icol].get_position().y0 * fh)
            x1s[irow, icol] = (axes[irow][icol].get_position().x1 * fw)
            y1s[irow, icol] = (axes[irow][icol].get_position().y1 * fh)
            aw[irow, icol] = (axes[irow][icol].get_position().width * fw)
            ah[irow, icol] = (axes[irow][icol].get_position().height * fh)

            ratio[irow, icol] = ah[irow, icol] / aw[irow, icol]

    xpad = np.empty((nrows, ncols - 1))
    ypad = np.empty((nrows - 1, ncols))
    for irow in range(nrows):
        for icol in range(ncols - 1):
            xpad[irow, icol] = ((axes[irow][icol + 1].get_position().x0
                                 - axes[irow][icol].get_position().x1)
                                * fw)
    for irow in range(nrows - 1):
        for icol in range(ncols):
            ypad[irow, icol] = ((axes[irow][icol].get_position().y0
                                 - axes[irow + 1][icol].get_position().y1)
                                * fh)

    margins = FigureMargins(
        left=np.array([ax.get_position().x0 * fw
                       for ax in get_column(0, axes)]),
        right=np.array([(1.0 - ax.get_position().x1) * fw
                        for ax in get_column(-1, axes)]),
        top=np.array([(1.0 - ax.get_position().y1) * fh
                      for ax in axes[0]]),
        bottom=np.array([ax.get_position().y0 * fh
                         for ax in axes[-1]]))

    return FigureLayout(fig_width=fw, fig_height=fh,
                        nrows=nrows, ncols=ncols,
                        axes_x0s=x0s, axes_y0s=y0s,
                        axes_x1s=x1s, axes_y1s=y1s,
                        axes_widths=aw, axes_heights=ah, ratios=ratio,
                        xpads=xpad, ypads=ypad,
                        margins=margins)


def convert_figure_layout_to_relative(layout: FigureLayout) -> FigureLayout:
    """ Convert a figure layout to relative units (everything from 0 to 1) """
    margins = FigureMargins(
        layout.margins.left / layout.fig_width,
        layout.margins.right / layout.fig_width,
        layout.margins.top / layout.fig_height,
        layout.margins.bottom / layout.fig_height)
    return FigureLayout(
        1.0, 1.0,
        layout.nrows, layout.ncols,
        layout.axes_x0s / layout.fig_width,
        layout.axes_y0s / layout.fig_height,
        layout.axes_x1s / layout.fig_width,
        layout.axes_y1s / layout.fig_height,
        layout.axes_widths / layout.fig_width,
        layout.axes_heights / layout.fig_height,
        layout.ratios,
        layout.xpads / layout.fig_width, layout.ypads / layout.fig_height,
        margins)


def _determine_aw(
    grid: tuple[int, int],
    fig_width: float,
    xpad: float,
    margins: FigureMarginsFloat
) -> float:
    total_xpad = (grid[1] - 1) * xpad
    space_for_axes = fig_width - (margins.left + margins.right + total_xpad)
    return space_for_axes / grid[1]


def _determine_xpad(
    grid: tuple[int, int],
    fig_width: float,
    axes_width: float,
    margins: FigureMarginsFloat
) -> float:
    leftover = fig_width - (
        grid[1] * axes_width + margins.left + margins.right)
    return leftover / (grid[1] - 1)


def _determine_fw(
    grid: tuple[int, int],
    xpad: float,
    axes_width: float,
    margins: FigureMarginsFloat
) -> float:
    return margins[0] + margins[1] + (grid[1] - 1) * xpad \
        + grid[1] * axes_width


def _determine_lr_margins(
    grid: tuple[int, int],
    xpad: float,
    fig_width: float,
    axes_width: float,
    margins: FigureMarginsFloat
) -> FigureMarginsFloat:
    leftover = fig_width - (grid[1] - 1) * xpad + grid[1] * axes_width
    return FigureMarginsFloat(leftover / 2.0, leftover / 2.0,
                              margins.top, margins.bottom)


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    axes_width: Optional[float] = None,
    margins: ArrayLike = (35.0, 15.0, 5.0, 35.0),
    xpad: Optional[float] = 45.0,
    ypad: Optional[float] = 35.0,
    fig_width: Optional[float] = None,
    ratio: float = 1.618,
    projections: Optional[Union[str, list[Union[str, None]]]] = None,
    axes_width_unit: str = "mm",
    margins_unit: str = "pts",
    pads_unit: str = "pts",
    fig_width_unit: str = "mm",
    axes_zorder: Optional[float] = 10000.0
) -> tuple[mplfig.Figure,
           list[list[mplax.Axes]]]:
    """
    Create a :code:`matplotlib.figure.Figure` with *nrows* times *ncols*
    subaxes and fixed margins, ratios, and pads

    Parameters
    ----------

    nrows : int, default :code:`1`
        Number of rows of axes in the figure

    ncols : int, default :code:`1`
        Number of columns of axes in the figure

    axes_width : float or :code:`None`, default :code:`45`
        Width of all axes. Units are specified by *axes_width_unit*.
        If `None`, determine width corresponding to fig_width, xpad and margins

    margins : (float, float, float, float), default :code`(35, 15, 5, 35)`
        Left, right, top, bottom margins. Units are specified by *margins_unit*

    xpad : float or :code:`None`, default :code:`35.0`
        Horizontal space between axes. Units are specified by *pads_unit*.
        If `None`, determine width corresponding to fig_width, axes_width
        and margins

    ypad : float or :code:`None`, default :code:`25`
        Vertical space between axes. Units are specified by *pads_unit*
        If `None`, use same as xpad

    fig_width : float or :code:`None`, default :code:`None`
        Width of the figure. Units are specified by *fig_width_unit*.
        If `None`, determine width corresponding to axes_width, xpad and
        margins

    ratio : float, default :code:`1.618`
        Axes width / axes height

    projection : :code:`"aitoff"`, :code:`"hammer"`, :code:`"lambert"`, \
:code:`"mollweide"`, :code:`"polar"`, :code:`"rectilinear"`, str,\
or list thereof, optional
        The projection type of the `matplotlib.axes.Axes`. *str* is the name of
        a custom projection, see `~matplotlib.projections`. The default
        `None` results in a 'rectilinear' projection.

    axes_width_unit : :code:`"inch"`, :code:`"pts"` or :code:`"mm"`, \
default :code:`"mm"`
        Unit of *axes_width*

    margins_unit : :code:`"inch"`, :code:`"pts"` or :code:`"mm"`, \
default :code:`"pts"`
        Unit of *margins*

    pads_unit : :code:`"inch"`, :code:`"pts"` or :code:`"mm"`, \
default :code:`"pts"`
        Units of *xpad* and *ypad*

    fig_width_unit : :code:`"inch"`, :code:`"pts"` or :code:`"mm"`, \
default :code:`"mm"`
        Unit of *fig_width*

    axis_zorder : float, optional, default 10000.0 (because why not)
        Set zorder of the axes spines to this value

    Returns
    -------
    :code:`matplotlib.figure.Figure`:
        The figure

    list[list[:code:`matplotlib.axes.Axes`]]
        A two-dimensional list, where the first index refers to columns, the
        second index to rows
    """
    XPAD_DEF = 45.0 / PTS_PER_INCH
    AW_DEF = 45.0 / MM_PER_INCH
    # check and convert every layout parameters to inches if they are not None
    valid_units = ["mm", "pts", "inch"]
    for unit in [fig_width_unit, axes_width_unit, pads_unit, margins_unit]:
        if unit not in valid_units:
            raise ValueError(
                f"Invalid unit specifier. "
                f"Valid options are 'mm', 'pts', 'inch'"
            )

    if fig_width_unit == "pts" and fig_width is not None:
        fig_width = fig_width / PTS_PER_INCH
    elif fig_width_unit == "mm" and fig_width is not None:
        fig_width = fig_width / MM_PER_INCH

    if axes_width_unit == "pts" and axes_width is not None:
        axes_width = axes_width / PTS_PER_INCH
    elif axes_width_unit == "mm" and axes_width is not None:
        axes_width = axes_width / MM_PER_INCH

    if pads_unit == "pts":
        xpad = xpad / PTS_PER_INCH if xpad else xpad
        ypad = ypad / PTS_PER_INCH if ypad else ypad
    elif pads_unit == "mm":
        xpad = xpad / MM_PER_INCH if xpad else xpad
        ypad = ypad / MM_PER_INCH if ypad else ypad

    try:
        iter(margins)
    except TypeError:
        margins_ = FigureMarginsFloat(*[margins] * 4)
    else:
        margins_ = FigureMarginsFloat(*margins)

    if margins_unit == "pts":
        margins_ = FigureMarginsFloat(*[m / PTS_PER_INCH for m in margins_])
    if margins_unit == "mm":
        margins_ = FigureMarginsFloat(*[m / MM_PER_INCH for m in margins_])

    grid = (nrows, ncols)

    # figure out all layout parameters

    if grid[1] == 1:  # xpad not utilized since only one axes
        if xpad is None:
            xpad = XPAD_DEF

        if fig_width is not None and axes_width is not None:
            raise ValueError(
                "Both fw and aw are provided and there is only one "
                "column. I don't know what to adjust"
            )

        elif fig_width is not None and axes_width is None:
            axes_width = _determine_aw(grid, fig_width, xpad, margins_)

        elif fig_width is None and axes_width is None:
            axes_width = AW_DEF
            fig_width = _determine_fw(grid, xpad, axes_width, margins_)

        elif fig_width is None and axes_width is not None:
            fig_width = _determine_fw(grid, xpad, axes_width, margins_)

    else:
        if fig_width is not None \
                and axes_width is not None and xpad is not None:
            margins_ = _determine_lr_margins(
                grid, xpad, fig_width, axes_width, margins_)

        elif fig_width is not None and axes_width is not None and xpad is None:
            xpad = _determine_xpad(grid, fig_width, axes_width, margins_)

        elif fig_width is not None and axes_width is None and xpad is not None:
            axes_width = _determine_aw(grid, fig_width, xpad, margins_)

        elif fig_width is not None and axes_width is None and xpad is None:
            axes_width = AW_DEF
            axes_width = _determine_xpad(grid, fig_width, axes_width, margins_)

        elif fig_width is None and axes_width is not None and xpad is not None:
            fig_width = _determine_fw(grid, xpad, axes_width, margins_)

        elif fig_width is None and axes_width is not None and xpad is None:
            xpad = XPAD_DEF
            fig_width = _determine_fw(grid, xpad, axes_width, margins_)

        elif fig_width is None and axes_width is None and xpad is not None:
            axes_width = AW_DEF
            fig_width = _determine_fw(grid, xpad, axes_width, margins_)

        elif fig_width is None and axes_width is None and xpad is None:
            axes_width = AW_DEF
            xpad = XPAD_DEF
            axes_width = _determine_fw(grid, xpad, axes_width, margins_)

    # if any layout parameters are still None (except ypad), something went
    # wrong
    if xpad is None or fig_width is None or axes_width is None:
        raise RuntimeError("something went wrong with layout algorithm")

    if ypad is None:
        ypad = xpad

    layout_width = ((grid[1] - 1) * xpad + grid[1] * axes_width + margins_.left
                    + margins_.right)
    if (layout_width >= fig_width + 0.0005 or fig_width <= 0
            or axes_width <= 0 or xpad < 0):
        print("\n\nLayout too big for fw\n")

    axes_height = axes_width / ratio
    fig_height = (margins_.top + margins_.bottom + (grid[0] - 1) * ypad
                  + grid[0] * axes_height)

    if isinstance(projections, list):
        if len(projections) != grid[0] * grid[1]:
            raise ValueError(
                f"{len(projections)=} must match number "
                f"of axes={grid[0]*grid[1]}"
            )
    else:
        projections = [projections] * grid[0] * grid[1]

    # layout is now set
    # move on to actually create the figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height))

    axes_height_rel = axes_height / fig_height  # relative coordinates
    axes_width_rel = axes_width / fig_width
    xpad_rel = xpad / fig_width
    ypad_rel = ypad / fig_height
    margins_rel = FigureMarginsFloat(
        margins_.left / fig_width, margins_.right / fig_width,
        margins_.top / fig_height, margins_.bottom / fig_height)

    origins = np.empty((grid[0], grid[1], 2))
    y0 = 1.0 - margins_rel.top - axes_height_rel
    for irow in range(grid[0]):
        x0 = margins_rel.left
        for icol in range(grid[1]):
            origins[irow, icol, 0] = x0
            origins[irow, icol, 1] = y0
            x0 += axes_width_rel + xpad_rel
        y0 -= ypad_rel + axes_height_rel

    axes: list[list[mplax.Axes]] = []
    for irow in range(grid[0]):
        axes.append([])
        for icol in range(grid[1]):
            axes[-1].append(
                fig.add_axes([origins[irow, icol, 0],
                              origins[irow, icol, 1],
                              axes_width_rel, axes_height_rel],
                             projection=projections[irow * grid[1] + icol]))
            if axes_zorder is not None:
                for _, spine in axes[-1][-1].spines.items():
                    spine.set_zorder(axes_zorder)

    return fig, axes


def _update_colorbar_position(
    colorbar: Union[Colorbar, ColorbarLarge],
    fig_width_inches: float,
    fig_height_inches: float
) -> None:
    """ Update *colorbar* position to fit *colorbar.parent_axes* """
    pos_cb = colorbar.colorbar.ax.get_position()

    if isinstance(colorbar, Colorbar):
        pos_ax = colorbar.parent_axes.get_position()
        if colorbar.location == "right":
            colorbar.colorbar.ax.set_position([
                pos_ax.x1 + colorbar.pad_inch / fig_width_inches,
                pos_ax.y0,
                pos_cb.width,
                pos_ax.height])
        else:  # location == "top"
            colorbar.colorbar.ax.set_position([
                pos_ax.x0,
                pos_ax.y1 + colorbar.pad_inch / fig_height_inches,
                pos_ax.width,
                pos_cb.height])

    else:  # colorbar == ColorbarLarge
        bbox_ax_0 = colorbar.parent_axes[0].get_position()
        bbox_ax_1 = colorbar.parent_axes[1].get_position()

        if colorbar.location == "right":
            colorbar.colorbar.ax.set_position([
                bbox_ax_1.x1 + colorbar.pad_inch / fig_width_inches,
                bbox_ax_1.y0,
                pos_cb.width,
                bbox_ax_0.y1 - bbox_ax_1.y0])
        else:  # location == "top"
            colorbar.colorbar.ax.set_position([
                bbox_ax_0.x0,
                bbox_ax_0.y1 + colorbar.pad_inch / fig_height_inches,
                bbox_ax_1.x1 - bbox_ax_0.x0,
                pos_cb.height])


def _get_offsets(
    figure: mplfig.Figure,
    edge_axes: _Regions,
    renderer,
) -> FigureMarginsFloat:
    tight_bboxes = _Regions(
        [ax.get_tightbbox(renderer) for ax in edge_axes.left],
        [ax.get_tightbbox(renderer) for ax in edge_axes.right],
        [ax.get_tightbbox(renderer) for ax in edge_axes.top],
        [ax.get_tightbbox(renderer) for ax in edge_axes.bottom])

    fw_inch, fh_inch = figure.get_size_inches()
    fw_px, fh_px = fw_inch * figure.get_dpi(), fh_inch * figure.get_dpi()

    all_offsets = FigureMargins(
        np.array([b.x0 / fw_px * fw_inch for b in tight_bboxes.left]),
        np.array([(fw_px - b.x1) / fw_px * fw_inch for b in tight_bboxes.right]),
        np.array([(fh_px - b.y1) / fh_px * fh_inch for b in tight_bboxes.top]),
        np.array([b.y0 / fh_px * fh_inch for b in tight_bboxes.bottom]))

    relevant_offsets = FigureMarginsFloat(*[np.min(o) for o in all_offsets])

    # calculate current margins, in case their largest value is smaller
    # than the axes linewidth, adjust the corresponding offsets
    current_margins = FigureMargins(
        np.array([a.get_position().x0 * fw_inch
                  for a in edge_axes.left]),
        np.array([(1.0 - a.get_position().x1) * fw_inch
                  for a in edge_axes.right]),
        np.array([(1.0 - a.get_position().y1) * fh_inch
                  for a in edge_axes.top]),
        np.array([a.get_position().y0 * fh_inch
                  for a in edge_axes.bottom]))

    all_margins = FigureMargins(
        current_margins.left - all_offsets.left,
        current_margins.right - all_offsets.right,
        current_margins.top - all_offsets.top,
        current_margins.bottom - all_offsets.bottom)

    margins = FigureMarginsFloat(*[np.min(m) for m in all_margins])

    lw = plt.rcParams["axes.linewidth"] / PTS_PER_INCH
    for i in range(4):
        if margins[i] < lw / 2.0:
            relevant_offsets[i] -= lw / 2.0

    return relevant_offsets


def _find_axes_x0y0(
    margins_ofs_inch: FigureMarginsFloat,
    layout_inch: FigureLayout,
    pad_inch: FigureMarginsFloat
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x0s = np.empty(layout_inch.axes_x0s.shape)
    y0s = np.empty(layout_inch.axes_y0s.shape)
    for irow in range(layout_inch.nrows):
        for icol in range(layout_inch.ncols):
            aw_until_now = np.sum(layout_inch.axes_widths[irow, :icol])
            ah_until_now = np.sum(layout_inch.axes_heights[:irow + 1, icol])

            ypads_until_now = np.sum(layout_inch.ypads[:irow, icol])
            xpads_until_now = np.sum(layout_inch.xpads[irow, :icol])

            x0s[irow, icol] = (
                layout_inch.margins.left[irow] - margins_ofs_inch.left
                + pad_inch.left
                + xpads_until_now + aw_until_now)
            y0s[irow, icol] = (
                layout_inch.fig_height
                - layout_inch.margins.top[icol]
                + margins_ofs_inch.top
                - pad_inch.top
                - ypads_until_now
                - ah_until_now)
    return x0s, y0s


def _change_margins(
    margins_ofs_inch: FigureMarginsFloat,
    axes: list[list[mplax.Axes]],
    figure: mplfig.Figure,
    fixed_figwidth: bool,
    pad_inch: FigureMarginsFloat,
    colorbars: list[Union[Colorbar, ColorbarLarge]]
) -> None:
    layout_inch = get_figure_layout(figure, axes, "inch")

    xpads_sum = np.sum(layout_inch.xpads, axis=1)
    ypads_sum = np.sum(layout_inch.ypads, axis=0)

    if not fixed_figwidth:
        axes_widths_sum = np.sum(layout_inch.axes_widths, axis=1)
        axes_heights_sum = np.sum(layout_inch.axes_heights, axis=0)

        layout_inch.fig_width = np.max(
            layout_inch.margins.left - margins_ofs_inch.left
            + layout_inch.margins.right - margins_ofs_inch.right
            + pad_inch.left + pad_inch.right
            + xpads_sum + axes_widths_sum)
        layout_inch.fig_height = np.max(
            layout_inch.margins.top - margins_ofs_inch.top
            + layout_inch.margins.bottom - margins_ofs_inch.bottom
            + pad_inch.top + pad_inch.bottom
            + ypads_sum + axes_heights_sum)

        layout_inch.axes_x0s, layout_inch.axes_y0s = _find_axes_x0y0(
            margins_ofs_inch, layout_inch, pad_inch)

    else:  # "fig_width"
        # scale everything such that everything fits new_fw
        current_effective_fig_width = (layout_inch.fig_width
                                       - margins_ofs_inch.left
                                       - margins_ofs_inch.right)

        scale_x = layout_inch.fig_width / current_effective_fig_width

        layout_inch.axes_widths *= scale_x
        layout_inch.axes_heights = layout_inch.axes_widths * layout_inch.ratios
        layout_inch.xpads *= scale_x

        layout_inch.fig_height = np.max(
            layout_inch.margins.top - margins_ofs_inch.top
            + layout_inch.margins.bottom - margins_ofs_inch.bottom
            + np.sum(layout_inch.axes_heights, axis=0)
            + np.sum(layout_inch.ypads, axis=0))

        layout_inch.axes_x0s, layout_inch.axes_y0s = _find_axes_x0y0(
            margins_ofs_inch, layout_inch, pad_inch)

    layout_rel = convert_figure_layout_to_relative(layout_inch)

    figure.set_size_inches(layout_inch.fig_width, layout_inch.fig_height)

    for irow in range(layout_rel.nrows):
        for icol in range(layout_rel.ncols):
            axes[irow][icol].set_position([
                layout_rel.axes_x0s[irow, icol],
                layout_rel.axes_y0s[irow, icol],
                layout_rel.axes_widths[irow, icol],
                layout_rel.axes_heights[irow, icol]])

    for cb in colorbars:
        _update_colorbar_position(
            cb, layout_inch.fig_width, layout_inch.fig_height)


def make_margins_tight(
    axes: list[list[mplax.Axes]],
    figure: Optional[mplfig.Figure] = None,
    fixed_figwidth: bool = False,
    colorbars: Union[Colorbar,
                     list[Colorbar],
                     list[list[Colorbar]],
                     ColorbarLarge,
                     None] = None,
    pad: Union[float, Sequence[float]] = 0.0,
    nruns: int = 1,
    relevant_axes: Optional[Sequence[mplax.Axes]] = None,
    log: bool = False
) -> FigureMargins:
    """
    Change figure margins such that all elements of the axes fit neatly.

    Parameters
    ----------
    axes : list[list[`~matplotlib.axes.Axes`]
        A 2D matrix of axes of the figure, where the first index specifies
        the row, the second the column

    figure: `matplotlib.figure.Figure`, optional
        if not provided, get current figure with plt.gcf()

    fix_figwidth: bool, default `False`
        Specify what width to keep constant
        - `True`: Change "axes_width" and horizontal padding between
        axes. May result in buggy behaviour if the aspect ratio of the axes
        is not constant. In that case, a fixed "axes_width" will not have
        problems
        - `False`: Change the width of the figure to accomodate for the new
        margins. Should also work if axes have different aspect ratios

    colorbars: `atompy.plotting.Colorbar` or `list[atompy.plotting.Colorbar]`
    or `list[list[atompy.plotting.Colorbar]` or `atompy.plotting.ColorbarLarge`
        Also consider and update colorbars of the figure.

    pad: float or (float, float, float, float), default 0pts
        Add padding in pts around the figure. If passed as a Sequecne, order
        is (left, right, top, bottom)

    nruns: int, default 1
        If the width of the axes changes dramatically, the routine may need
        to run multiple times to accommodate possibly updated ticklabels

    relevant_axes: (plt.Axes, plt.Axes, plt.Axes, plt.Axes), optional
        If the algorithm fails to figure out the correct margins, it may help
        to tell it which axes should be used to determine which margin, i.e.
        the passed Sequence of Axes is used to determine the
        (left, right, top, bottom)-margins.

    log: bool, default `False`
        Print margins that were determined

    Returns
    -------
    margins: `FigureMargins`
        The (left, right, top, bottom) margins of each axes located at the
        edges of the figure
    """

    if nruns <= 0 or not isinstance(nruns, int):
        raise ValueError(
            f"{nruns=}, but it needs to be of type int and larger than 0"
        )

    try:
        iter(pad)
    except TypeError:
        pad = FigureMarginsFloat(*[pad / PTS_PER_INCH] * 4)
    else:
        if not isinstance(pad, FigureMarginsFloat):
            pad = FigureMarginsFloat(*[p / PTS_PER_INCH for p in pad])

    figure = figure or plt.gcf()

    renderer = figure.canvas.get_renderer()

    colorbars_ = [] if colorbars is None else return_as_list(colorbars)

    if relevant_axes is not None:
        edge_axes = _Regions(*[[ax] for ax in relevant_axes])
    else:
        edge_axes = _Regions(
            get_column(0, axes).copy(),
            get_column(-1, axes).copy(),
            axes[0].copy(),
            axes[-1].copy())
        # append colorbars to relevant axes if they exist
        for cb in colorbars_:
            if isinstance(cb, Colorbar):
                if cb.parent_axes in edge_axes.left and cb.location == "top":
                    edge_axes.left.append(cb.colorbar.ax)
                if cb.parent_axes in edge_axes.right:
                    edge_axes.right.append(cb.colorbar.ax)
                if cb.parent_axes in edge_axes.top:
                    edge_axes.top.append(cb.colorbar.ax)
                if (cb.parent_axes in edge_axes.bottom
                        and cb.location == "right"):
                    edge_axes.bottom.append(cb.colorbar.ax)
            elif isinstance(cb, ColorbarLarge):
                if (cb.parent_axes[0] in edge_axes.left
                        and cb.location == "top"):
                    edge_axes.left.append(cb.colorbar.ax)
                if cb.parent_axes[1] in edge_axes.right:
                    edge_axes.right.append(cb.colorbar.ax)
                if cb.parent_axes[0] in edge_axes.top:
                    edge_axes.top.append(cb.colorbar.ax)
                if (cb.parent_axes[1] in edge_axes.bottom
                        and cb.location == "right"):
                    edge_axes.bottom.append(cb.colorbar.ax)

    for _ in range(nruns):
        margins = _get_offsets(figure, edge_axes, renderer)
        _change_margins(
            margins, axes, figure, fixed_figwidth, pad, colorbars_)

    new_layout = get_figure_layout(figure, axes, "pts")
    if log:
        print("I have changed the margins to the follwoing:")
        for l, m in zip("left right top bottom".split(), new_layout.margins):
            with np.printoptions(precision=3):
                print(f"{l}:\t{m}")

    return new_layout.margins


def _format_colorbar(
    cb: mplcb.Colorbar,
    where: str,
    label: str,
    **kwargs
) -> None:
    """ Do some formatting for colorbars """
    if where == "right":
        kwargs.setdefault("rotation", 270.0)
        if not ("va" in kwargs or "verticalalignment" in kwargs):
            kwargs["va"] = "baseline"

    elif where == "top":
        kwargs.setdefault("rotation", 0.0)
        if not ("va" in kwargs or "verticalalignment" in kwargs):
            kwargs["va"] = "bottom"
        cb.ax.xaxis.set_label_position("top")
        cb.ax.xaxis.set_ticks_position("top")

    cb.set_label(label, **kwargs)


@overload
def add_colorbar(
    axes: mplax.Axes,
    image: mplcm.ScalarMappable,
    figure: Optional[mplfig.Figure] = None,
    width_pts: float = 4.8,
    pad_pts: float = 3.0,
    where: Literal["right", "top"] = "right",
    rasterized: bool = True,
    label: str = "",
    **text_kwargs
) -> Colorbar: ...


@overload
def add_colorbar(
    axes: list[mplax.Axes],
    image: Union[mplcm.ScalarMappable,
                 list[mplcm.ScalarMappable]],
    figure: Optional[mplfig.Figure] = None,
    width_pts: float = 4.8,
    pad_pts: float = 3.0,
    where: Union[Literal["right", "top"],
                 Sequence[Literal["right", "top"]]] = "right",
    rasterized: bool = True,
    label: Union[str, Sequence[str]] = "",
    **text_kwargs
) -> list[Colorbar]: ...


@overload
def add_colorbar(
    axes: list[list[mplax.Axes]],
    image: Union[mplcm.ScalarMappable,
                 list[list[mplcm.ScalarMappable]]],
    figure: Optional[mplfig.Figure] = None,
    width_pts: float = 4.8,
    pad_pts: float = 3.0,
    where: Union[Literal["right", "top"],
                 Sequence[Sequence[Literal["right", "top"]]]] = "right",
    rasterized: bool = True,
    label: Union[str, Sequence[Sequence[str]]] = "",
    **text_kwargs
) -> list[list[Colorbar]]: ...


def add_colorbar(
    axes: Union[mplax.Axes,
                list[mplax.Axes],
                list[list[mplax.Axes]]],
    image: Union[mplcm.ScalarMappable,
                 list[mplcm.ScalarMappable],
                 list[list[mplcm.ScalarMappable]]],
    figure: Optional[mplfig.Figure] = None,
    width_pts: float = 4.8,
    pad_pts: float = 3.0,
    where: Union[Literal["right", "top"],
                 Sequence[Literal["right", "top"]],
                 Sequence[Sequence[Literal["right", "top"]]]] = "right",
    rasterized: bool = True,
    label: Union[str, Sequence[str], Sequence[Sequence[str]]] = "",
    **text_kwargs
) -> Union[Colorbar, list[Colorbar], list[list[Colorbar]]]:
    """ Add a `~matplotlib.pyplot.colorbar` to a `~matplotlib.pyplot.axes`

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes` or list or list of lists thereof
        The axes to append the colorbar to. If a list is given, append
        colorbar to all those

    figure: `matplotlib.figure.Figure`, optional
        if not provided, get current figure with plt.gcf()

    image : Mappable or list or list of lists thereof
        The images corresponding to *axes*. (returned from,
        e.g., `~matplotlib.pyplot.imshow`) If no list is given while *axes*
        is a list, use same image for all colorbars.

    width_pts : float, default: 4.8
        width of the colorbar in pts
        - float: overwrites ratio
        - `None`: ignore, use *ratio* instead

    pad_pts : float, default: 3pts
        padding between colorbar and axes in pts

    where : {'right', 'top'}, or list[str], default: 'right'
        Position of the colorbar
        - single string: global setting
        - sequence of string: different position for each *axes* (as long as
          a list of *axes* was given)

    rasterized : bool, default `True`
        Rasterize the colorbar image (replace it by a raster image in
        vectorized graphics)

    label : str or list[str], optional
        Label for the color
        - str: A label
        - `None`: draw no label

    **text_kwargs : `matplotlib.text.Text` properties
        used for the label

    Returns
    -------
    `atompy.Colorbar` or list or list of lists thereof
        The colorbar(s). The `matplotlib.colorbar` object is stored in
        `atompy.Colorbar.colorbar`.

    Notes
    -----
    A colorbar without a previously drawn artist (e.g., by imshow) can
    be created on the fly, e.g.,
    >>> add_colorbar(fig, ax, matplotlib.cm.ScalarMappable())

    """
    ############################
    # process input parameters #
    ############################
    original_shape = None
    if isinstance(axes, list) and isinstance(axes[0], list):
        original_shape = (len(axes), len(axes[0]))
        axes = flatten(axes)
    else:
        axes = return_as_list(axes, 1)

    figure = figure or plt.gcf()

    image = return_as_list(image, len(axes))
    where_ = return_as_list(where, len(axes))
    label = return_as_list(label, len(axes))

    if len(image) != len(axes):
        raise ValueError(
            f"{len(image)=} != {len(axes)=}, but that should not be"
        )
    if len(where_) != len(axes):
        raise ValueError(
            f"{len(where_)=} != {len(axes)=}, but that should not be"
        )
    if len(label) != len(axes):
        raise ValueError(
            f"{len(label)=} != {len(axes)=}, but that should not be"
        )

    for w in where_:
        if w not in ["top", "right"]:
            raise ValueError(
                f"where is '{w}', but needs to be either 'top' or 'right'"
            )

    #######################################
    # loop through axes and add colorbars #
    #######################################
    return_colorbars: list[Colorbar] = []
    for i in range(len(axes)):
        figsize = figure.get_size_inches()
        bbox_ax = axes[i].get_position()

        if where_[i] == "right":
            orientation = "vertical"
            x0 = bbox_ax.x0 + bbox_ax.width \
                + pad_pts / PTS_PER_INCH / figsize[0]
            y0 = bbox_ax.y0
            width = width_pts / PTS_PER_INCH / figsize[0]
            height = bbox_ax.height
        else:
            orientation = "horizontal"
            x0 = bbox_ax.x0
            y0 = bbox_ax.y0 + bbox_ax.height \
                + pad_pts / PTS_PER_INCH / figsize[1]
            width = bbox_ax.width
            height = width_pts / PTS_PER_INCH / figsize[1]

        cax = figure.add_axes([x0, y0, width, height])
        cb = plt.colorbar(image[i], cax=cax, orientation=orientation)
        cb.solids.set_rasterized(rasterized)

        _format_colorbar(cb, where_[i], label[i], **text_kwargs)

        return_colorbars.append(
            Colorbar(cb, axes[i], where_[i],
                     pad_pts / PTS_PER_INCH, width_pts / PTS_PER_INCH))

    if original_shape:
        return reshape(return_colorbars, original_shape)
    else:
        return (return_colorbars if len(return_colorbars) > 1
                else return_colorbars[0])


def add_colorbar_large(
    axes: Sequence[mplax.Axes],
    image: mplcm.ScalarMappable,
    figure: Optional[mplfig.Figure] = None,
    width_pts: float = 4.8,
    pad_pts: float = 3.0,
    where: Literal["right", "top"] = "right",
    rasterized: bool = True,
    label: str = "",
    **text_kwargs
) -> ColorbarLarge:
    """
    Add a colorbar spanning multiple axes

    Parameters
    ----------
    axes : (axes, axes)
        The left-/top-most and right-/bottom-most axes that the colorbar
        should span

    figure: `matplotlib.figure.Figure`, optional
        if not provided, get current figure with plt.gcf()

    image : `AxesImage`
        The images corresponding to axes.
        (returned from, e.g., `~matplotlib.pyplot.imshow`)

    width_pts : `None` or float, default: `None`
        width of the colorbar in pts
        - float: overwrites ratio
        - `None`: ignore, use *ratio* instead

    pad_pts : float, default: 3pts
        padding between colorbar and axes in pts

    where : {'right', 'top'}
        Position of the colorbar

    rasterized : bool, default `True`
        Rasterize the colorbar image (replace it by a raster image in
        vectorized graphics)

    label : str, default ""
        Label for the color
        - str: A label
        - `None`: draw no label

    **text_kwargs : `matplotlib.text.Text` properties
        used for the label

    Returns
    -------
    `atompy.ColorbarLarge`
        The colorbar. The parent_axes asigned to it will be the
        top-most/left-most axes, i.e., parameter `axes[0]`

    Notes
    -----
    A colorbar without a previously drawn artist (e.g., by imshow) can
    be created on the fly, e.g.,
    >>> add_colorbar_large(fig, ax, matplotlib.cm.ScalarMappable())
    """
    figure = figure or plt.gcf()
    figsize = figure.get_size_inches()
    if len(axes) != 2:
        raise ValueError(
            f"*axes* needs to be of length 2, but is of length {len(axes)}"
        )
    bbox_ax_0 = axes[0].get_position()
    bbox_ax_1 = axes[1].get_position()

    if where == "right":
        orientation = "vertical"
        x0 = bbox_ax_1.x1 + pad_pts / PTS_PER_INCH / figsize[0]
        y0 = bbox_ax_1.y0
        width = width_pts / PTS_PER_INCH / figsize[0]
        height = bbox_ax_0.y1 - bbox_ax_1.y0
    else:
        orientation = "horizontal"
        x0 = bbox_ax_0.x0
        y0 = bbox_ax_0.y1 + pad_pts / PTS_PER_INCH / figsize[1]
        width = bbox_ax_1.x1 - bbox_ax_0.x0
        height = width_pts / PTS_PER_INCH / figsize[1]

    cax = figure.add_axes([x0, y0, width, height])
    cb = plt.colorbar(image, cax=cax, orientation=orientation)
    cb.solids.set_rasterized(rasterized)

    _format_colorbar(cb, where, label, **text_kwargs)

    return ColorbarLarge(cb, tuple(axes), where, pad_pts / PTS_PER_INCH, width)


@overload
def square_polar_frame(
    axes: mplax.Axes,
    figure: Optional[mplfig.Figure] = None,
    n_gridlines: int = 12,
    mark_zero: bool = True,
    **plot_kwargs
) -> mplax.Axes: ...


@overload
def square_polar_frame(
    axes: list[mplax.Axes],
    figure: Optional[mplfig.Figure] = None,
    n_gridlines: int = 12,
    mark_zero: bool = True,
    **plot_kwargs
) -> list[mplax.Axes]: ...


@overload
def square_polar_frame(
    axes: list[list[mplax.Axes]],
    figure: Optional[mplfig.Figure] = None,
    n_gridlines: int = 12,
    mark_zero: bool = True,
    **plot_kwargs
) -> list[list[mplax.Axes]]: ...


def square_polar_frame(
    axes: Union[mplax.Axes,
                list[mplax.Axes],
                list[list[mplax.Axes]]],
    figure: Optional[mplfig.Figure] = None,
    n_gridlines: int = 12,
    mark_zero: bool = True,
    **plot_kwargs
) -> Union[mplax.Axes,
           list[mplax.Axes],
           list[list[mplax.Axes]]]:
    """
    Draw a square frame around a polar plot and hide the other stuff

    Parameters
    ----------
    axes: list of list of `matplotlib.axes.Axes`

    figure: `matplotlib.figure.Figure`, optional
        if not provided, get current figure with plt.gcf()

    n_gridlines: int
        Draw this many (outspreading) gridlines

    mark_zero: bool,
        Draw a crosshair at 0, 0

    **plot_kwargs
        `matplotlib.pyplot.plot` keyword arguments

    Returns
    -------
    `matplotlib.axes.Axes` or list or list of list thereof
        The (new) axes
    """
    figure = figure or plt.gcf()

    original_shape = None
    if isinstance(axes, list) and isinstance(axes[0], list):
        original_shape = (len(axes), len(axes[0]))
        axes = flatten(axes)
    else:
        axes = return_as_list(axes, 1)

    ax_frame: list[mplax.Axes] = []
    for ax in axes:
        pos = ax.get_position()
        zorder = ax.get_zorder()
        ax.axis("off")

        ax_frame.append(figure.add_axes(pos))
        ax_frame[-1].set_zorder(zorder - 1)
        ax_frame[-1].set_xlim(-1, 1)
        ax_frame[-1].set_ylim(-1, 1)
        ax_frame[-1].set_xticks([])
        ax_frame[-1].set_yticks([])

        if not ("linewidth" in plot_kwargs or "lw" in plot_kwargs):
            plot_kwargs["linewidth"] = plt.rcParams["axes.linewidth"]
        if not ("color" in plot_kwargs or "c" in plot_kwargs):
            plot_kwargs["color"] = plt.rcParams["axes.edgecolor"]
        plot_kwargs.setdefault("zorder", zorder)

        if n_gridlines > 0:
            if 360 % n_gridlines:
                print(f"WARNING: {n_gridlines=}, which 360 is not dividable "
                      "by. Should that be the case?")
            angles = [x * 2.0 * np.pi / n_gridlines
                      for x in range(n_gridlines)]

            for angle in angles:
                a, b = 0.95, 1.5
                ax_frame[-1].plot([a * np.cos(angle), b * np.cos(angle)],
                                  [a * np.sin(angle), b * np.sin(angle)],
                                  **plot_kwargs)

        if mark_zero:
            ax_frame[-1].axvline(0, **plot_kwargs)
            ax_frame[-1].axhline(0, **plot_kwargs)

    if original_shape:
        return reshape(ax_frame, original_shape)
    else:
        return ax_frame if len(ax_frame) > 1 else ax_frame[0]


def change_ratio(
    new_ratio: float,
    axes: Union[mplax.Axes,
                list[mplax.Axes],
                list[list[mplax.Axes]]],
    figure: Optional[mplfig.Figure] = None,
    adjust: Literal["height", "width"] = "height",
    anchor: Literal["center", "left", "right", "upper", "lower",
                    "upper left", "upper right", "upper center",
                    "center left", "center right", "center center",
                    "lower left", "lower right", "lower center"] = "center",
    colorbars: Optional[Union[Colorbar,
                              list[Colorbar],
                              list[list[Colorbar]]]] = None
) -> None:
    """
    Change the ratio of *axes* to *new_ratio*

    Parameters
    ----------
    new_ratio: float
        The new ratio width/height

    axes: `matplotlib.axes.Axes` or list thereof

    figure: `matplotlib.figure.Figure`, optional
        If not provided, get current figure using plt.gcf()

    adjust: "height", "width"

    anchor: "center", "left", "right", "upper", "lower",
        "upper left", "upper right", "upper center",
        "center left", "center right", "center center",
        "lower left", "lower right", "lower center"

    colorbar: `atompy.Colorbar` or list thereof, optional
        If colorbars exist, they need to be passed to also be updated. This
        can be passed as a list, even if *axes* is a single axes. (It'll
        pick the correct one)

    Notes
    -----
    Should not be called before `atompy.make_margins_tight()` is called,
    since that relies on all axes-ratios to be equal.
    If you plan to add a colorbar that spans multiple axes, do this
    afterwards, too.
    """
    figure = figure or plt.gcf()

    valid_adjusts = ["height", "width"]
    if adjust not in valid_adjusts:
        raise ValueError(
            f"{adjust=}, but must be in {valid_adjusts}"
        )
    valid_anchors = [
        "center", "left", "right", "upper", "lower",
        "upper left", "upper right", "upper center",
        "center left", "center right", "center center",
        "lower left", "lower right", "lower center"]
    if anchor not in valid_anchors:
        raise ValueError(
            f"{anchor=}, but must be in {valid_anchors}"
        )
    axes = return_as_list(axes, desired_length=1)
    if colorbars:
        colorbars = return_as_list(colorbars, desired_length=1)
    else:
        colorbars = []

    @ dataclass
    class Position:
        x0: float
        y0: float
        width: float
        height: float

    fig_ratio = figure.get_size_inches()[0] / figure.get_size_inches()[1]
    for ax in axes:
        old_pos = ax.get_position()
        new_pos = Position(old_pos.x0, old_pos.y0,
                           old_pos.width, old_pos.height)

        # adjust ratio ...
        if adjust == "height":
            new_pos.height = old_pos.width / new_ratio * fig_ratio
        else:
            new_pos.width = old_pos.height * new_ratio / fig_ratio

        # then adjust x0/y0 position
        if anchor == "center":
            anchor = "center center"
        if anchor == "left":
            anchor = "center left"
        if anchor == "right":
            anchor = "center right"
        if anchor == "upper":
            anchor = "upper center"
        if anchor == "lower":
            anchor = "lower center"

        anchor_split = anchor.split()

        if anchor_split[0] == "lower":
            pass
        elif anchor_split[0] == "upper":
            new_pos.y0 = old_pos.y0 + (old_pos.height - new_pos.height)
        elif anchor_split[0] == "center":
            new_pos.y0 = old_pos.y0 + (old_pos.height - new_pos.height) / 2.0

        if anchor_split[1] == "left":
            pass
        elif anchor_split[1] == "right":
            new_pos.x0 = old_pos.x0 + (old_pos.width - new_pos.width)
        elif anchor_split[1] == "center":
            new_pos.x0 = old_pos.x0 + (old_pos.width - new_pos.width) / 2.0

        ax.set_position([new_pos.x0, new_pos.y0,
                         new_pos.width, new_pos.height])
        for cb in colorbars:
            _update_colorbar_position(cb, *figure.get_size_inches())


def add_abc(
    axes: list[list[mplax.Axes]],
    figure: Optional[mplfig.Figure] = None,
    xoffset: Union[Optional[float],
                   Sequence[Sequence[Optional[float]]]] = None,
    yoffset: Union[Optional[float],
                   Sequence[Sequence[Optional[float]]]] = 0.0,
    anchor: Literal["upper left", "upper right",
                    "lower left", "lower right"] = "upper left",
    prefix: str = "",
    suffix: str = "",
    start_at: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                      "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
                      "w", "x", "y", "z"] = "a",
    rowsfirst: bool = True,
    sansserif: bool = False,
    smallcaps: bool = False,
    uppercase: bool = False,
    display_warning: bool = True,
    **text_kwargs
) -> list[list[mpltxt.Text]]:
    r"""
    Add labeling to all *axes* of *figure*

    Parameters
    ----------
    axes: list[list[`matplotlib.axes.Axes`]

    figure: `matplotlib.figure.Figure`
        If not provided, get current figure using plt.gcf()

    xoffset: float | None | list[list[float | None]], default None
        Horizontal shift from *anchor* in pts. Positive moves right.
        If provided as a 2D list, entries correspond to each *axes*,
        otherwise global.
        If xoffset is None, use the width of ylabel of each axes

    yoffset: float | None | list[list[float | None]], default 0pts.
        Vertical shift from *anchor* in pts. Positive moves up. If provided
        as a 2D list, entries correspond to each *axes*, otherwise global.
        If yoffset is None, use the extent of all axes elements into the upper
        margins instead.

    anchor : {'upper/lower left/right'}
        Where to align the labels from

    prefix : str, optional
        Print this before the labels a, b, c, ...

    suffix : str, optional
        Print this after the labels a, b, c, ...

    start_at : {'a', 'b', 'c', ..., 'z'}, default 'a'
        Specify with which letter to start

    rowsfirst : bool, default `True`

    sf : bool, default `False`
        Print `\sffamily` before label (works only with latex backend)

    sc : bool, default `False`
        Print \scfamily before label (works only with latex backend)

    uppercase : bool, default `False`
        Use uppercase instead of lowercase letters

    **text_kwargs : `matplotlib.text.Text` properties

    Returns
    -------
    list[list[`matplotlib.text.Text`]]
        The text instances of the labels
    """
    if display_warning:
        print("WARNING: Don't call 'add_abc()' before changes to the "
              "figure layout since that'll end up with funny business")

    figure = figure or plt.gcf()

    valid_anchors = ["upper left", "upper right", "lower left", "lower right"]
    if anchor not in valid_anchors:
        raise ValueError(
            f"{anchor=}, but it needs to be in {valid_anchors}"
        )

    layout_inch = get_figure_layout(figure, axes, "inch")

    try:
        iter(xoffset)
    except TypeError:
        xoffset = [[xoffset] * layout_inch.ncols] * layout_inch.nrows
    try:
        iter(yoffset)
    except TypeError:
        yoffset = [[yoffset] * layout_inch.ncols] * layout_inch.nrows

    if rowsfirst:
        axes_flat = flatten(axes)
        xoffsets_flat: list[Union[float, None]] = flatten(xoffset)
        yoffsets_flat: list[Union[float, None]] = flatten(yoffset)
    else:
        axes_transposed: list[list[mplax.Axes]] = list(map(list, zip(*axes)))
        xoffset_transposed = list(map(list, zip(*xoffset)))
        yoffset_transposed = list(map(list, zip(*yoffset)))
        axes_flat = flatten(axes_transposed)
        xoffsets_flat: list[Union[float, None]] = flatten(xoffset_transposed)
        yoffsets_flat: list[Union[float, None]] = flatten(yoffset_transposed)

    abc = "abcdefghijklmnopqrstuvwxyz"
    abc_ofs = abc.index(start_at)
    if uppercase:
        abc = abc.upper()

    renderer = figure.canvas.get_renderer()
    prefix_sf = r"\sffamily" if sansserif else ""
    prefix_sc = r"\scshape" if smallcaps else ""
    text_kwargs["transform"] = figure.transFigure
    text_kwargs["clip_on"] = False

    @ dataclass
    class Offset:
        dx: float
        dy: float

    output_text: list[mpltxt.Text] = []
    for i, (ax, xofs, yofs) in enumerate(zip(axes_flat,
                                             xoffsets_flat,
                                             yoffsets_flat)):
        ofs_rel = Offset(0.0, 0.0)
        bbox = ax.get_position()
        if xofs is None:
            tbbox = ax.get_tightbbox(renderer)
            ofs_rel.dx = tbbox.x0 / figure.get_dpi() / layout_inch.fig_width - bbox.x0
        else:
            ofs_rel.dx = xofs / PTS_PER_INCH / layout_inch.fig_width

        if yofs is None:
            tbbox = ax.get_tightbbox(renderer)
            ofs_rel.dy = tbbox.y1 / layout_inch.fig_height / figure.get_dpi() - bbox.y1
        else:
            ofs_rel.dy = yofs / PTS_PER_INCH / layout_inch.fig_height

        if anchor.split()[0] == "upper":
            pos_abc_y = bbox.y1 + ofs_rel.dy
        else:
            pos_abc_y = bbox.y0 + ofs_rel.dy
        if anchor.split()[1] == "left":
            pos_abc_x = bbox.x0 + ofs_rel.dx
        else:
            pos_abc_x = bbox.x1 + ofs_rel.dx

        label = f"{prefix_sf}{prefix_sc}{prefix}{abc[i+abc_ofs]}{suffix}"

        output_text.append(ax.text(pos_abc_x, pos_abc_y, label, **text_kwargs))

    return reshape(output_text, (layout_inch.nrows, layout_inch.ncols))


def abcify_axes(
    axes: Optional[Union[mplax.Axes,
                         list[mplax.Axes],
                         list[list[mplax.Axes]]]] = None,
) -> dict[str, mplax.Axes]:
    """
    Return a dictionary with keys a, b, b, ... of the input axes. If a 2D
    list of axes is provided, it cycles through rows first. (If you want to
    cycle through columns first, pass a transposed 2D list).

    Paramters
    ---------
    axes: `matplotlib.axes.Axes` or list or list of lists thereof

    Returns
    -------
    dict[str, matplotlib.axes.Axes]
        A dictionary where keys are a, b, c, ... and values are the input axes
    """
    if isinstance(axes, str):
        raise ValueError(
            "*axes* cannot be a string"
        )

    abc = "abcdefghijklmnopqrstufvxyz"

    if isinstance(axes, mplax.Axes):
        return {abc[0]: axes}

    try:
        iter(axes)
    except TypeError:
        raise ValueError(
            "Invalid input for *axes*"
        )

    axes = flatten(axes)
    output = {}
    for i, a in enumerate(axes):
        output[abc[i]] = a
    return output


if __name__ == "__main__":
    import atompy as ap
    fig, ax = subplots(2, 1, margins=50, ypad=40)

    label = "abcdef"
    for i, a in enumerate(ap.flatten(ax)):
        a.text(0.5, 0.5, label[i], transform=a.transAxes)

    data = np.arange(9).reshape(3, 3)
    im = ax[0][0].imshow(data, aspect="auto")

    # cb = ap.add_colorbar(fig, ax[0][0], im)

    # change_ratio(1, [ax[0][0], ax[1][0]], adjust="height", anchor="center")
    # change_ratio(1, ax[1][1], adjust="width")
    # change_ratio(1, ax[2][0], adjust="width")
    # change_ratio(1, get_column(0, ax), adjust="width")
    # change_ratio(1, ax[0][0], adjust="width")
    make_margins_tight(ax, fixed_figwidth=True,
                       nruns=3, relevant_axes=(ax[1][0], ax[1][0], ax[0][0], ax[1][0]))
    change_ratio(1, ax[0][0], adjust="width")

    fig.savefig("test_tight.pdf")


if __name__ == "__main__":
    print("Achwas")

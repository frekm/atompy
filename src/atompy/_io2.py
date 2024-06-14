import uproot
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Literal, overload
from . import _histogram
from . import _miscellaneous as _misc


class UnderdeterminedBinsizeError(Exception):
    def __init__(self, fname: str, axis: Literal["x", "y"]):
        self._fname = fname
        self._axis = axis

    def __str__(self) -> str:
        return (
            f"Distance between {self._axis}-points in '{self._fname}' "
            f"is not constant and no lower or upper {self._axis}-"
            "limits are provided. Provide at least one limit so I can "
            "determine the binsizes."
        )


def save_1d_as_txt(
    histogram: NDArray[np.float_],
    edges: NDArray[np.float_],
    fname: str,
    **savetxt_kwargs
) -> None:
    """
    Save a 1d histogram to a file.

    Saves the centers of the bin, not the edges.

    Parameters
    ----------
    histogram : ndarray, shape(n,)
        The histogram values.

    edges : ndarray, shape(n+1,)`
        Edges of histogram.

    **savetxt_kwargs
        :func:`numpy.savetxt` keyword arguments. Useful to, e.g., set a header
        with the ``header`` keyword.

    Examples
    --------
    .. code-block:: python

        samples = np.random.default_rng().normal(size=1_000)
        h, edges = np.histogram(samples, 50)
        ap.save_1d_as_txt(h, edges, "filename.txt")
    """
    bincenters = edges[:-1] + 0.5 * np.diff(edges)
    output = np.zeros((len(bincenters), 2))
    output[:, 0] = bincenters
    output[:, 1] = histogram
    savetxt_kwargs.setdefault("header", "x\tvalues")
    np.savetxt(fname, output, **savetxt_kwargs)


def save_2d_as_txt(
    H: NDArray[np.float_],
    xedges: NDArray[np.float_],
    yedges: NDArray[np.float_],
    fname: str,
    **savetxt_kwargs
) -> None:
    """
    Save a 2d histogram to a file.

    The first column in the file will be y, the second x, the third z.
    (this is chosen as such because the the standard hist2ascii-macro of the
    Atomic Physics group has this format)

    Parameters
    ----------
    H : ndarray shape(nx,ny)
        A bi-dimensional histogram of samples x and y.

    xedges : ndarray, shape(nx+1,)
        Edges along x.

    yedges : ndarray, shape(ny+1,)
        Edges along y.

    fname : str
        Filename, including filetype.

    **savetxt_kwargs
        :func:`numpy.savetxt` keyword arguments. Useful to, e.g., set a header
        with the ``header`` keyword.
    """
    xbinsizes = np.diff(xedges, 1)
    ybinsizes = np.diff(yedges, 1)
    xbincenters = xedges[:-1] + xbinsizes / 2.0
    ybincenters = yedges[:-1] + ybinsizes / 2.0
    nx = xbincenters.shape[0]
    ny = ybincenters.shape[0]

    out = np.zeros((nx * ny, 3))
    for ix, x in enumerate(xbincenters):
        for iy, y in enumerate(ybincenters):
            out[ix + iy * nx, 0] = y
            out[ix + iy * nx, 1] = x
            out[ix + iy * nx, 2] = H[ix, iy]
    savetxt_kwargs.setdefault("delimiter", "\t")
    savetxt_kwargs.setdefault("header", "y\tx\tvalues")
    np.savetxt(fname, out, **savetxt_kwargs)


def _work_out_bin_edges(
    centers: NDArray,
    lower: Optional[float],
    upper: Optional[float],
    fname: str,
    which: Literal["x", "y"]
) -> NDArray:
    """
    Work out bin edges from bin centers.

    If the bins don't have constant size, at least one limit has to be
    provided, from which the edges can be determined

    Parameters
    ----------
    centers : ndarray, shape(n)
        centers of the bins

    lower, uppper : float, optional
        Lower and upper limits of the bins.

        At least one limit must be provided if bins don't have a constant 
        size. If both lower and upper limits are provided, the lower one
        will be prioritized.

    Returns
    -------
    edges : ndarray, shape(n+1)
        Edges of the bins.
    """
    # if bins don't have a constant size, determine xbinedges differently
    edges = np.empty(centers.size + 1)
    binsize = centers[1] - centers[0]
    if np.any(np.abs(np.diff(centers) - binsize) > binsize * 0.001):
        if lower is not None:
            # take lower edge and work out binsize forward
            edges[0] = lower
            for i in range(len(centers)):
                edges[i+1] = 2.0 * centers[i] - edges[i]

        elif upper is not None:
            # take upper edge and work out binsize backward
            edges[-1] = upper
            for i in reversed(range(len(centers))):
                edges[i] = 2.0 * centers[i] - edges[i+1]
        else:
            # cannot determine binsize, throw exception
            raise UnderdeterminedBinsizeError(fname, which)
    else:  # bins have equal size
        edges[:-1] = centers - 0.5 * binsize
        edges[-1] = centers[-1] + 0.5 * binsize

    return edges


@overload
def load_1d_from_txt(
    fname: str,
    output_format: Literal["ndarray"] = "ndarray",
    transform: bool = True,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    **loadtxt_kwargs
) -> NDArray[np.float64]: ...


@overload
def load_1d_from_txt(
    fname: str,
    output_format: Literal["Hist1d"] = "ndarray",  # type: ignore
    transform: bool = True,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    **loadtxt_kwargs
) -> _histogram.Hist1d: ...


def load_1d_from_txt(
        fname: str,
        output_format: Literal["ndarray", "Hist1d"] = "ndarray",
        transform: bool = True,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        **loadtxt_kwargs
) -> Union[NDArray[np.float64], _histogram.Hist1d]:
    """
    Load a text file.

    This is a wrapper function for :func:`numpy.loadtxt` that changes the
    output according to `output_format`.

    Parameters
    ----------
    fname : str
        Filename

    output_format : {``"ndarray"``, ``Hist1d``}, default ``"ndarray"``
        Change output format. See Returns.

    transform : bool, default True
        Transform the loaded NumPy ndarray.

    xmin : float, optional
        Specify the lower x-limit of the data in `fname`. Only necessary if
        the x-values in `fname` are not equally spaced. Alternatively, specify
        `xmax`.

    xmax : float, optional
        Specify the upper x-limit of the data in `fname`. Only necessary if
        the x-values in `fname` are not equally spaced. Alternatively, specify
        `xmin`. If `xmin` *and* `xmax` are specified, `xmax` is not used.

    **loadtxt_kwargs
        Other :func:`numpy.loadtxt` keyword arguments. Useful if, e.g., you want
        to skip a certain number of lines in the text file.

    Returns
    -------
    output : ndarray or :class:`.Hist1d`
        Depends on `output_format`.

        - ``output_format == "ndarray"``: Return a NumPy ndarray. ``output[0]``
          refers to the bin-centers of the loaded histogram, ``output[1]`` to
          the corresponding values.
        - ``output_format == "Hist1d"``: Return a :class:`.Hist1d`.
          ``output.edges`` (``output.histogram``) is analogous to 
          ``output[0]`` (``output[1]``) of the ``"ndarray"`` output.

    Examples
    --------
    .. code-block:: python

        x, y = ap.load_1d_from_txt("filename.txt", output_format="ndarray")
        plt.plot(x, y)

        hist = ap.load_1d_from_txt("filename.txt", output_format="Hist1d")
        plt.plot(hist.centers, hist.histogram)


    """
    valid_output_formats = ("ndarray", "Hist1d")
    if output_format not in valid_output_formats:
        errmsg = (
            f"{output_format=}, but it must be one of {valid_output_formats}"
        )
        raise ValueError(errmsg)

    output = np.loadtxt(fname, **loadtxt_kwargs)
    if transform:
        output = output.T

    if output_format == "ndarray":
        return output  # type: ignore

    if output_format == "Hist1d":
        xedges = _work_out_bin_edges(output[0], xmin, xmax, fname, "x")
        return _histogram.Hist1d(output[1], xedges)


@overload
def load_1d_from_root(
    fname: str,
    hname: str,
    output_format: Literal["Hist1d"] = "ndarray"  # type: ignore
) -> _histogram.Hist1d: ...


@overload
def load_1d_from_root(
    fname: str,
    hname: str,
    output_format: Literal["ndarray"] = "ndarray"
) -> NDArray[np.float64]: ...


def load_1d_from_root(
    fname: str,
    hname: str,
    output_format: Literal["ndarray", "Hist1d"] = "ndarray"
) -> Union[NDArray[np.float64], _histogram.Hist1d]:
    """
    Import a 1d histogram from a `ROOT <https://root.cern.ch/>`_ file.

    Parameters
    ----------
    fname : str
        The filename of the ROOT file, e.g., ``important_data.root``

    hname : str
        The name of the histogram within the ROOT file,
        e.g., ``path/to/histogram1d``.

    output_format : {``"ndarray"``, ``"Hist1d"``}, default ``"ndarray"``
        Change output format. See Returns.

    Returns
    -------
    output : ndarray or :class:`.Hist1d`
        Depends on `output_format`.

        - ``output_format == "ndarray"``: Return a NumPy ndarray. ``output[0]``
          refers to the bin-centers of the loaded histogram, ``output[1]`` to
          the corresponding values.
        - ``output_format == "Hist1d"``: Return a :class:`.Hist1d`.
          ``output.centers`` (``output.histogram``) is analogous to 
          ``output[0]`` (``output[1]``) of the ``"ndarray"`` output.

    Examples
    --------
    .. code-block:: python

        x, y = ap.load_root_hist1d("rootfile.root", "path/to/hist1d", output_format="ndarray")
        plt.plot(x, y)

        hist = ap.load_root_hist1d("rootfile.root", "path/to/hist1d", output_format="Hist1d")
        plt.plot(hist.centers, hist.histogram)

    """
    valid_output_formats = ("ndarray", "Hist1d")
    if output_format not in valid_output_formats:
        errmsg = (
            f"{output_format=}, but it must be one of {valid_output_formats}"
        )
        raise ValueError(errmsg)

    with uproot.open(fname) as file:  # type: ignore
        histogram, edges = file[hname].to_numpy()  # type: ignore

    if output_format == "ndarray":
        output = np.empty((2, histogram.shape[0]))
        output[0] = edges[:-1] + 0.5 * np.diff(edges)
        output[1] = histogram
        return output

    if output_format == "Hist1d":
        return _histogram.Hist1d(histogram, edges)


@overload
def load_2d_from_txt(
    fname: str,
    output_format: Literal["pcolormesh"] = "pcolormesh",
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    **loadtxt_kwargs
) -> _misc.PcolormeshData: ...


@overload
def load_2d_from_txt(
    fname: str,
    output_format: Literal["imshow"] = "pcolormesh",  # type: ignore
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    **loadtxt_kwargs
) -> _misc.ImshowData: ...


@overload
def load_2d_from_txt(
    fname: str,
    output_format: Literal["Hist2d"] = "pcolormesh",  # type: ignore
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    **loadtxt_kwargs
) -> _histogram.Hist2d: ...


def load_2d_from_txt(
    fname: str,
    output_format: Literal["imshow",
                           "pcolormesh", "Hist2d"] = "pcolormesh",
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    **loadtxt_kwargs
) -> Union[_misc.PcolormeshData, _misc.ImshowData, _histogram.Hist2d]:
    """
    Load 2D data stored in a text file.

    Three columns in the file should specify the x, y, and corresponding z
    values of the data. E.g.,

    ::

        y0 x0 z00
        y0 x1 z01
        y0 x2 z02
        ...
        y0 xN z0N
        y1 x0 z10
        ...
        yM xN zMN

    You can specify which value is permuting first (in the example above ``x``)
    with the `permuting` keyword. The assignment of the columns (here, y, x, z)
    is specified by the `xyz_indices` keyword.

    Parameters
    ----------
    fname : str
        Filename, including filetype.

    output_format : {``"imshow"``, ``"pcolormesh"``, ``"Hist2d"`` }, default ``"pcolormesh"``
        Change output format. See `Returns`.

    xyz_indices : (int, int, int), default (1, 0, 2)
        Specify which column in the file corresponds to x, y, z.

        The default corresponds to the output format of the default
        ROOT macro of the Atomic Physics group that exports 2D data to a
        text file (``hist2ascii``).

    permuting : {``"x"``, ``"y"``}, default ``"x"``
        Specify if the x- or y-column permutes through the values first.

    xmin, ymin : float, optional
        Specify the lower x (y) limit of the data in `fname`. Only necessary if
        the x (y) values in `fname` are not equally spaced. Alternatively,
        specify `xmax` (`ymax`.

    xmax, ymax : float, optional
        Specify the upper x (y) limit of the data in `fname`. Only necessary if
        the x (y) values in `fname` are not equally spaced. Alternatively,
        specify `xmin` (`ymin`). If `xmin` (`ymin`) *and* `xmax` (`ymin`) are
        specified, `xmax` (`ymax`) is not used.

    **loadtxt_kwargs
        Other :func:`numpy.loadtxt` keyword arguments. Useful if, e.g., you want
        to skip a certain number of lines in the text file.

    Returns
    -------
    output : :class:`.PcolormeshData`, :class:`.ImshowData` or :class:`.Hist2d` 
        Depends on `output_format`.

        - ``output_format == "pcolormesh"``: Return :class:`.PcolormeshData`.
        - ``output_format == "imshow"``: Return :class:`.Imshow`.
        - ``output_format == "Hist2d"``: Return a :class:`.Hist2d`.

    Examples
    --------
    Load data such that it can be plotted using
    :meth:`matplotlib.pyplot.pcolormesh`.

    .. code-block:: python

        data = ap.load_2d_from_txt("data.txt", output_format="pcolormesh")
        plt.pcolormesh(data.x, data.y, data.z)

    Load data such that it can be plotted using
    :meth:`matplotlib.pyplot.imshow`.

    .. code-block:: python

        data = ap.load_2d_from_txt("data.txt", output_format="imshow")
        plt.imshow(data.image, extent=data.extent)

    Load data as a :class:`.Hist2d"`.

    .. code-block:: python

        hist = ap.load_2d_from_txt("data.txt", output_format="Hist2d")
        data = hist.column_normalized_to_sum.for_pcolormesh
        plt.pcolormesh(data.x, data.y, data.z)


    """
    valid_output_formats = ["imshow", "pcolormesh", "Hist2d"]
    if output_format not in valid_output_formats:
        errmsg = (
            f"{output_format=}, but it must be one of {valid_output_formats}"
        )
        raise ValueError(errmsg)

    idx_x, idx_y, idx_z = xyz_indices

    data = np.loadtxt(fname, **loadtxt_kwargs)

    x = np.unique(data[:, idx_x])
    y = np.unique(data[:, idx_y])

    xedges = _work_out_bin_edges(x, xmin, xmax, fname, "x")
    yedges = _work_out_bin_edges(y, ymin, ymax, fname, "y")

    if permuting == "x":
        z = data[:, idx_z].reshape(y.size, x.size).T
    elif permuting == "y":
        z = data[:, idx_z].reshape(x.size, y.size)
    else:
        msg = f"'{permuting=}', but should be 'x' or 'y'"
        raise ValueError(msg)

    output = _histogram.Hist2d(z, xedges, yedges)

    if output_format == "Hist2d":
        return output
    if output_format == "imshow":
        return output.for_imshow
    if output_format == "pcolormesh":
        return output.for_pcolormesh


@overload
def load_2d_from_root(
    fname: str,
    hname: str,
    output_format: Literal["pcolormesh"] = "pcolormesh"
) -> _misc.PcolormeshData: ...


@overload
def load_2d_from_root(
    fname: str,
    hname: str,
    output_format: Literal["imshow"] = "pcolormesh"  # type: ignore
) -> _misc.ImshowData: ...


@overload
def load_2d_from_root(
    fname: str,
    hname: str,
    output_format: Literal["Hist2d"] = "pcolormesh"  # type: ignore
) -> _histogram.Hist2d: ...


def load_2d_from_root(
    fname: str,
    hname: str,
    output_format: Literal["pcolormesh", "imshow", "Hist2d"] = "pcolormesh"
) -> Union[_misc.PcolormeshData, _misc.ImshowData, _histogram.Hist2d]:
    """
    Load 2D data stored in a `ROOT <https://root.cern.ch/>`_ file.

    Parameters
    ----------
    fname : str
        Filename, e.g., ``"my_root_file.root"``.

    hname : str
        Histogram name in the root file, e.g.,
        ``"path/to/histogram"``.

    output_format : {``"imshow"``, ``"pcolormesh"``, ``"Hist2d"`` }, default ``"pcolormesh"``
        Change output format. See `Returns`.

    Returns
    -------
    output : :class:`.PcolormeshData`, :class:`.ImshowData` or :class:`.Hist2d` 
        Depends on `output_format`.

        - ``output_format == "pcolormesh"``: Return :class:`.PcolormeshData`.
        - ``output_format == "imshow"``: Return :class:`.Imshow`.
        - ``output_format == "Hist2d"``: Return a :class:`.Hist2d`.

    Examples
    --------
    Load data such that it can be plotted using
    :meth:`matplotlib.pyplot.pcolormesh`.

    .. code-block:: python

        data = ap.load_2d_from_txt("rootfile.root", "path/to/hist",
                                   output_format="pcolormesh")
        plt.pcolormesh(data.x, data.y, data.z)

    Load data such that it can be plotted using
    :meth:`matplotlib.pyplot.imshow`.

    .. code-block:: python

        data = ap.load_2d_from_txt("rootfile.root", "path/to/hist",
                                   output_format="imshow")
        plt.imshow(data.image, extent=data.extent)

    Load data as a :class:`.Hist2d"`.

    .. code-block:: python

        data = ap.load_2d_from_txt("rootfile.root", "path/to/hist",
                                   output_format="Hist2d")
        data = hist.column_normalized_to_sum.for_pcolormesh
        plt.pcolormesh(data.x, data.y, data.z)


    """
    valid_output_formats = ["imshow", "pcolormesh", "Hist2d"]
    if output_format not in valid_output_formats:
        errmsg = (
            f"{output_format=}, but it must be one of {valid_output_formats}"
        )
        raise ValueError(errmsg)

    with uproot.open(fname) as file:  # type: ignore
        output = _histogram.Hist2d(*file[hname].to_numpy())  # type: ignore

    if output_format == "Hist2d":
        return output
    if output_format == "imshow":
        return output.for_imshow
    if output_format == "pcolormesh":
        return output.for_pcolormesh

if __name__ == "__main__":
    print("Hi")

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
import uproot
from typing import Literal, Sequence, Union, overload, Optional
import matplotlib.pyplot as plt
from . import _histogram as aph
from . import _miscellaneous as apm


class NonconstantBinsizeError(Exception):
    def __init__(
        self,
        fname: str,
        which: Literal["x", "y", ""]
    ) -> None:
        self.fname = fname
        self.which = which

    def __str__(self):
        return (
            f"{self.which}binsizes from {self.fname} are not constant and "
            f"no {self.which}-limits are provided. Provide either "
            f"{self.which}lim[0] or {self.which}lim[1]"
        )


def save_ascii_hist1d(
    histogram: npt.NDArray[np.float_],
    edges: npt.NDArray[np.float_],
    fname: str,
    **savetxt_kwargs
) -> None:
    """
    Save a 1d histogram created by `numpy.histogram` to a file. Saves the
    centers of the bin, not the edges

    Parameters
    ----------
    histogram : `np.ndarray`, `shape(n,)`
        The histogram of samples

    edges : `np.ndarray`, `shape(n+1,)`
        Edges of histogram

    Other Parameters
    ----------------

    **savetxt_kwargs
        `numpy.savetxt <https://numpy.org/doc/stable/reference/generated/
        numpy.savetxt.html>`_ keyword argument

    Examples
    --------
    ::

        import numpy as np
        import atompy as ap

        rng = np.random.default_rng()
        xsamples = rng.normal(100_000)
        histogram = np.histogram(xsamples, 50)
        ap.save_ascii_hist1d(*histogram, "filename.txt")
    """
    bincenters = edges[:-1] + 0.5 * np.diff(edges)

    output = np.zeros((len(bincenters), 2))
    output[:, 0] = bincenters
    output[:, 1] = histogram

    savetxt_kwargs.setdefault("fmt", "%.5lf")
    np.savetxt(fname, output, **savetxt_kwargs)


def save_ascii_hist2d(
    H: NDArray[np.float_],
    xedges: NDArray[np.float_],
    yedges: NDArray[np.float_],
    fname: str,
    **savetxt_kwargs
) -> None:
    """
    Save a 2d histogram created by `numpy.histogram2d` to a file. The first
    column in the file will be y, the second x, the third z. (this is chosen
    as such because the the standard hist2ascii-macro of our group outputs this
    order)

    Parameters
    ----------
    H : `ndarray, shape(nx,ny)`
        The bi-dimensional histogram of samples x and y 

    xedges : `ndarray, shape(nx+1,)`
        Edges along x

    yedges : `ndarray, shape(ny+1,)`
        Edges along y

    fname : str
        Filename, including filetype

    Other Parameters
    ----------------

    **savetxt_kwargs
        `numpy.savetxt <https://numpy.org/doc/stable/reference/generated/
        numpy.savetxt.html>`_ keyword argument
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
    savetxt_kwargs.setdefault("fmt", "%.5lf")
    savetxt_kwargs.setdefault("delimiter", "\t")
    np.savetxt(fname, out, **savetxt_kwargs)


@overload
def load_ascii_hist1d(
    fnames: str,
    **loadtxt_kwargs
) -> aph.Hist1d: ...


@overload
def load_ascii_hist1d(
    fnames: Sequence[str],
    **loadtxt_kwargs
) -> tuple[aph.Hist1d, ...]: ...


def load_ascii_hist1d(
    fnames: Union[str, Sequence[str]],
    **loadtxt_kwargs
) -> Union[aph.Hist1d,
           tuple[aph.Hist1d, ...]]:
    """
    Load :class:`.Hist1d` from a file. If you don't want a histogram but simply
    (bincenters, histogram_values), use :func:`.load_ascii_data1d` instead.

    Parameters
    ----------
    fnames : str or Sequence[str]
        Filename(s). Should contain two columns, where the first represents the
        centers of the xbins, the second the values of the histogram.

    **loadtxt_kwargs
        Keyword arguments for `np.loadtxt <https://numpy.org/doc/stable/
        reference/generated/numpy.loadtxt.html>`_

    Returns
    -------
    histogram : :class:`.Hist1d` or tuple[`Hist1d`, ...]
        A atompy Hist1d instance. If *fnames* is a Sequence,
        *histogram* is a tuple of Hist1d instances.

    Examples
    --------
    ::

        # hist.histogram are the histogram values, hist.edges its edges
        hist = load_ascii_hist1d("file.dat")

        # Load multiple histograms at once
        hists = load_ascii_hist1d(["file1.dat", "file2.dat"])
    """
    if isinstance(fnames, str):
        fnames = [fnames]
    output = []
    for fname in fnames:
        data = np.loadtxt(fname, **loadtxt_kwargs)

        binsize = data[1, 0] - data[0, 0]

        if not np.all(np.abs(np.diff(data[:, 0]) - binsize) < 1e-2):
            raise NonconstantBinsizeError(fname, "")

        edges = np.empty(data.shape[0] + 1)
        edges[: -1] = data[:, 0] - 0.5 * binsize
        edges[-1] = data[-1, 0] + 0.5 * binsize

        output.append(aph.Hist1d(data[:, 1], edges))

    return tuple(output) if len(fnames) > 1 else output[0]


def __work_out_bin_edges(
    centers: NDArray,
    limits: Sequence[Optional[float]]
) -> NDArray:
    """
    Work out bin edges from bin centers.

    If the bins don't have constant size, at least one limit has to be
    provided, from which the edges can be determined

    Parameters
    ----------
    centers : np.ndarray, shape(n)
        centers of the bins

    limits : (float, float), optional
        Lower and upper limits of the bins

        At least on limit must be provided if bins don't have a constant 
        size. If both lower and upper limits are provided, the lower one
        will be prioritized

    Returns
    -------
    edges : np.ndarray, shape(n+1)
        Edges of the bins
    """
    # if bins don't have a constant size, determine xbinedges differently
    edges = np.empty(centers.size + 1)
    binsize = centers[1] - centers[0]
    if not np.all(np.diff(centers) - binsize < binsize * 0.001):
        if limits[0] is not None:
            # take lower edge and work out binsize forward
            edges[0] = limits[0]
            for i in range(len(centers)):
                edges[i+1] = 2.0 * centers[i] - edges[i]

        elif limits[1] is not None:
            # take upper edge and work out binsize backward
            edges[-1] = limits[1]
            for i in reversed(range(len(centers))):
                edges[i] = 2.0 * centers[i] - edges[i+1]
        else:
            # cannot determine binsize, throw exception
            raise ValueError
    else:  # bins have equal size
        edges[:-1] = centers - 0.5 * binsize
        edges[-1] = centers[-1] + 0.5 * binsize

    return edges


@overload
def load_ascii_hist2d(
    fnames: str,
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xlim: Sequence[Optional[float]] = (None, None),
    ylim: Sequence[Optional[float]] = (None, None),
    **loadtxt_kwargs
) -> aph.Hist2d: ...


@overload
def load_ascii_hist2d(
    fnames: Sequence[str],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xlim: Sequence[Optional[float]] = (None, None),
    ylim: Sequence[Optional[float]] = (None, None),
    **loadtxt_kwargs
) -> tuple[aph.Hist2d, ...]: ...


def load_ascii_hist2d(
    fnames: Union[str, Sequence[str]],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xlim: Sequence[Optional[float]] = (None, None),
    ylim: Sequence[Optional[float]] = (None, None),
    **loadtxt_kwargs
) -> Union[aph.Hist2d,
           tuple[aph.Hist2d, ...]]:
    """
    Load :class:`.Hist2d` from a file.

    Parameters
    ----------
    fnames : str or Sequence[str]
        Filename(s). Should contain three columns defining the centers of
        the x and y bins, and the histogram values. Which column is which
        is defined by *xyz_indices*.

    xyz_indices : (int, int, int), default (1, 0, 2)
        Specify columns of x, y, and z, starting at 0

    permuting : "x" or "y", default "x"
        Order of permutation of x and y in ascii file
        - "x": first permute through x-values before changing y-values
        - "y": first permute through y-values before changing x-values

    xlim, ylim : ({None, float}, {None, float}), default (None, None)
        Lower and/or upper limits of x/y range (NOT the center of the bin,
        but the lower/upper edge of the histogram)

        If the binsize of the histogram is not constant, provide at least one
        of them, otherwise a NonconstantBinsizeError exception is thrown.

    **loadtxt_kwargs
        Keyword arguments for `np.loadtxt <https://numpy.org/doc/stable/
        reference/generated/numpy.loadtxt.html>`_

    Returns
    -------
    histogram2d : :class:`.Hist2d` or tuple[`Hist2d`, ...]
        A Hist2d instance of the histogram data. If *fnames* was
        a Sequence, *histogram2d* will be a tuple of Hist2d instances.
    """
    if isinstance(fnames, str):
        fnames = [fnames]

    idx_x, idx_y, idx_z = xyz_indices

    output = []
    for fname in fnames:
        data = np.loadtxt(fname, **loadtxt_kwargs)

        x = np.unique(data[:, idx_x])
        y = np.unique(data[:, idx_y])

        try:
            xedges = __work_out_bin_edges(x, xlim)
        except ValueError:
            raise NonconstantBinsizeError(fname, "x")
        try:
            yedges = __work_out_bin_edges(y, ylim)
        except ValueError:
            raise NonconstantBinsizeError(fname, "y")

        if permuting == "x":
            z = data[:, idx_z].reshape(y.size, x.size).T
        elif permuting == "y":
            z = data[:, idx_z].reshape(x.size, y.size)
        else:
            msg = f"'{permuting=}', but should be 'x' or 'y'"
            raise ValueError(msg)

        output.append(aph.Hist2d(z, xedges, yedges))

    return tuple(output) if len(fnames) > 1 else output[0]


@overload
def load_ascii_data1d(
    fnames: str,
    **loadtxt_kwargs
) -> tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64]]: ...


@overload
def load_ascii_data1d(
    fnames: Sequence[str],
    **loadtxt_kwargs
) -> tuple[tuple[npt.NDArray[np.float64], ...],
           tuple[npt.NDArray[np.float64], ...]]: ...


def load_ascii_data1d(
    fnames: Union[str, Sequence[str]],
    **loadtxt_kwargs
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           tuple[tuple[npt.NDArray[np.float64], ...],
                 tuple[npt.NDArray[np.float64], ...]]]:
    """
    Import 1d data from an ascii file with two columns (x, y). For any other
    layout, directly use `np.loadtxt <https://numpy.org/doc/stable/reference/
    generated/numpy.loadtxt.html>`_ instead.

    Parameters
    ----------
    fnames : str or Sequence[str]
        Filename(s)

    Other Parameters
    ----------------

    **loadtxt_kwargs
        optional `np.loadtxt <https://numpy.org/doc/stable/reference/
        generated/numpy.loadtxt.html>`_ keyword arguments

    Returns
    -------
    x : `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy. \
    ndarray.html>`_ or tuple[`np.ndarray`, ...]
        Data of the first column of the file.
        If *fnames* is a Sequence, *x* is a tuple of numpy arrays.

    x : `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy. \
    ndarray.html>`_ or tuple[`np.ndarray`, ...]
        Data of the second column of the file
        If *fnames* is a Sequence, *y* is a tuple of numpy arrays.

    """
    if isinstance(fnames, str):
        x, y = np.loadtxt(fnames, **loadtxt_kwargs).T
        return x, y
    else:
        outx, outy = [], []
        for fname in fnames:
            x, y = np.loadtxt(fname, **loadtxt_kwargs).T
            outx.append(x)
            outy.append(y)
        return tuple(outx), tuple(outy)


@overload
def import_ascii_for_imshow(
    fnames: str,
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xlim: Sequence[Optional[float]] = (None, None),
    ylim: Sequence[Optional[float]] = (None, None),
    origin: Literal["auto"] = "auto",
    **kwargs
) -> apm.ImshowData: ...


@overload
def import_ascii_for_imshow(
    fnames: Sequence[str],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xlim: Sequence[Optional[float]] = (None, None),
    ylim: Sequence[Optional[float]] = (None, None),
    origin: Literal["auto"] = "auto",
    **kwargs
) -> tuple[apm.ImshowData, ...]: ...


def import_ascii_for_imshow(
    fnames: Union[str, Sequence[str]],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    xlim: Sequence[Optional[float]] = (None, None),
    ylim: Sequence[Optional[float]] = (None, None),
    origin: Literal["auto"] = "auto",
    **kwargs
) -> Union[apm.ImshowData,
           tuple[apm.ImshowData, ...]]:
    """
    Import 2d-histogram data and return an image and the extents of the image.
    If you want to work with the histogram, consider using
    :func:`.load_ascii_hist2d` instead.

    Parameters
    ----------
    fnames : str or Sequence[str]
        The filename(s) of the data. If a list is passed, return a list
        of images and extents

    xyz_indices : (int, int, int), default (1, 0, 2)
        Specify columns of x, y, and z, starting at 0

    permuting : `"x"` or `"y"`
        Order of permutation of x and y in ascii file

        - `"x"`: first permutate through x-values before changing y-values
        - `"y"`: first permutate through y-values before changing x-values

    **kwargs :
        `np.loadtxt <https://numpy.org/doc/stable/reference/generated/
        numpy.loadtxt.html>`_ keyword arguments

    Returns
    -------
    image : `np.ndarray` or tuple[`np.ndarray`, ...]
        A 2d-array of pixel values.
        If *fnames* is a Sequence, *image* is a tuple of arrays

    extent : `np.ndarray`, shape((4,)), or tuple[`np.ndarray`, ...]
        The extent of the image, e.g., [-1, 1, -2, 2]
        If *fnames* is a Sequence, *extent* is a tuple of arrays

    Other Parameters
    ----------------
    origin : ``"auto"``
        Origin of the image plotted by `plt.imshow
        <https://matplotlib.org/stable/api/
        _as_gen/matplotlib.pyplot.imshow.html>`_

        Here for backward compatibility. Leave at ``"auto"``.

    Examples
    --------
    ::

        # 1 file
        data, extent = import_ascii_for_imshow("filename.txt")
        plt.imshow(data, extent=extent)

        # 2 files
        filenames = ["filename1.txt", "filename2.txt"]
        data, extents = import_ascii_for_imshow(filenames)
        plt.imshow(data[0], extent=extent[0])
        plt.imshow(data[1], extent=extent[1])
    """
    if origin == "lower" or origin == "upper":
        msg = (
            f"{origin=} is no longer supported. If you want to force "
            f"{origin=}, set rcParams['image.origin'] to '{origin}'"
        )
        raise ValueError(msg)
    elif origin != "auto":
        msg = (f"{origin=}, but it must be 'auto'")
        raise ValueError(msg)

    hist2d = load_ascii_hist2d(
        fnames, xyz_indices, permuting, xlim, ylim, **kwargs)

    if isinstance(hist2d, aph.Hist2d):
        return hist2d.for_imshow
    else:
        return tuple([x.for_imshow for x in hist2d])


@overload
def load_root_data1d(
    root_filename: str,
    histogram_names: str
) -> tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64]]: ...


@overload
def load_root_data1d(
    root_filename: str,
    histogram_names: Sequence[str]
) -> tuple[tuple[npt.NDArray[np.float64], ...],
           tuple[npt.NDArray[np.float64], ...]]: ...


def load_root_data1d(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]]
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           tuple[tuple[npt.NDArray[np.float64], ...],
                 tuple[npt.NDArray[np.float64], ...]]]:
    """
    Import 1d histogram(s) from a root file.

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names : str or Sequence thereof
        The name of the histogram(s) within the root file,
        e.g., 'path/to/histogram1d' or
        ['path/to/histogram1d_1', 'path/to/histogram1d_2']

    Returns
    -------
    x : `np.ndarray` or tuple[`np.ndarray`, ...]
        The x-data.
        If *histogram_names* was a Sequence, *x* is a tuple of arrays

    y : `np.ndarray` or tuple[`np.ndarray`, ...]
        The y-data.
        If *histogram_names* was a Sequence, *y* is a tuple of arrays


    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        import atompy as ap

        # single histogram
        x, y = ap.load_root_hist1d("rootfile.root",
                                   "path/to/hist1d")
        plt.plot(x, y)

        # multiple histograms
        xs, ys = ap.load_root_hist1d("rootfile.root",
                                     ["path/to/hist1d_0",
                                      "path/to/hist1d_1"])
        for x, y in zip(xs, ys):
            plt.plot(x, y)
    """
    if isinstance(histogram_names, str):
        histogram_names = [histogram_names]
    with uproot.open(root_filename) as file:  # type: ignore
        # np.empty((len(histogram_names), 2), dtype=object)
        output_x, output_y = [], []
        for i, h in enumerate(histogram_names):
            y, x = file[h].to_numpy()  # type: ignore
            x = x[:-1] + 0.5 * np.diff(x)
            output_x.append(x)
            output_y.append(y)
    if len(histogram_names) > 1:
        return tuple(output_x), tuple(output_y)
    else:
        return output_x[0], output_y[0]


@overload
def import_root_for_imshow(
    root_filename: str,
    histogram_names: str,
    origin: Literal["auto", "upper", "lower"] = "auto"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def import_root_for_imshow(
    root_filename: str,
    histogram_names: Sequence[str],
    origin: Literal["auto", "upper", "lower"] = "auto"
) -> tuple[tuple[npt.NDArray[np.float64], ...],
           tuple[npt.NDArray[np.float64], ...]]: ...


def import_root_for_imshow(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]],
    origin: Literal["auto", "upper", "lower"] = "auto"
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           tuple[tuple[npt.NDArray[np.float64], ...],
                 tuple[npt.NDArray[np.float64], ...]]]:
    """
    Import a 2d histogram from a root file to be plottable by
    `plt.imshow <https://matplotlib.org/stable/api/_as_gen/
    matplotlib.pyplot.imshow.html>`_.

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names : str or Sequence[str]
        The name of the histogram within the root file,
        e.g., 'path/to/histogram2d'.
        If a list of strings is passed, get multiple 2d histograms from the
        root file

    Returns
    -------
    image : `np.ndarray` or tuple[`np.ndarray`, ...]
        A 2d-array of pixel values.
        If *fnames* is a Sequence, *image* is a tuple of arrays

    extent : `np.ndarray`, shape((4,)) or tuple[`np.ndarray`, ...]
        The extent of the image, e.g., [-1, 1, -2, 2]
        If *fnames* is a Sequence, *extent* is a tuple of arrays

    Examples
    --------
    ::

        import matplotlib.pyplot as plt

        # Import one histogram
        data, extent = import_root_for_imshow("rootfile.root", "path/to/hist")
        plt.imshow(data, extent=extent)

        # Import multiple histograms
        data, extents = import_root_for_imshow(
            "rootfile.root", ["path/to/histo1", "path/to/histo2"])
        for date, extent in zip(data, extents):
            plt.imshow(date, extent=extent),

    """
    if isinstance(histogram_names, str):
        histogram_names = [histogram_names]

    if origin == "auto":
        origin = plt.rcParams["image.origin"]
    elif origin != "upper" and origin != "lower":
        raise ValueError(
            f"{origin=}, but it needs to be 'upper', 'lower', or 'auto'"
        )

    with uproot.open(root_filename) as file:  # type: ignore
        output_data: list[npt.NDArray] = []
        output_extent: list[npt.NDArray] = []
        for h in histogram_names:
            image, xedges, yedges = file[h].to_numpy()  # type: ignore
            if origin == "upper":
                image = np.flip(image.T, axis=0)
            else:
                image = image.T
            extent = np.array((np.min(xedges), np.max(xedges),
                               np.min(yedges), np.max(yedges)))

            output_data.append(image)
            output_extent.append(extent)

        if len(histogram_names) > 1:
            return tuple(output_data), tuple(output_extent)
        else:
            return output_data[0], output_extent[0]


@overload
def load_root_hist1d(
    root_filename: str,
    histogram_names: str
) -> aph.Hist1d: ...


@overload
def load_root_hist1d(
    root_filename: str,
    histogram_names: Sequence[str]
) -> tuple[aph.Hist1d, ...]: ...


def load_root_hist1d(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]]
) -> Union[aph.Hist1d,
           tuple[aph.Hist1d, ...]]:
    """
    Import :class:`.Hist1d` from a root file.
    If you want (bincenters, histogram_values), use :func:`.load_root_data1d`
    instead.

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names : str or Sequence thereof
        The name of the histogram(s) within the root file,
        e.g., 'path/to/histogram1d' or
        ['path/to/histogram1d_1', 'path/to/histogram1d_2']

    Returns
    -------
    histogram : :class:`.Hist1d` or tuple[`Hist1d`, ...]
        A atompy Hist1d instance. If *histogram_names* is a Sequence,
        *histogram* is a tuple of Hist1d instances.

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        import atompy as ap

        # single histogram
        hist = ap.load_root_hist1d("rootfile.root", "path/to/hist1d")
        plt.step(hist.edges[1:], hist.histogram)

        # multiple histograms
        hists = ap.load_root_hist1d(
            "rootfile.root", ["path/to/hist1d_0", "path/to/hist1d_1"])
        for hist in hists:
            plt.step(hist.edge[1:], hist.histogram)
    """
    if isinstance(histogram_names, str):
        histogram_names = [histogram_names]
    output = []
    with uproot.open(root_filename) as file:  # type: ignore
        for h in histogram_names:
            output.append(aph.Hist1d(*file[h].to_numpy()))  # type: ignore
    return tuple(output) if len(histogram_names) > 1 else output[0]


@overload
def load_root_hist2d(
    root_filename: str,
    histogram_names: str,
) -> aph.Hist2d: ...


@overload
def load_root_hist2d(
    root_filename: str,
    histogram_names: Sequence[str],
) -> tuple[aph.Hist2d, ...]: ...


def load_root_hist2d(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]],
) -> Union[aph.Hist2d,
           tuple[aph.Hist2d, ...]]:
    """
    Import a 2d histogram from a root file equivalent to `numpy.histogram2d`

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names : str or Sequence[str]
        The name of the histogram within the root file,
        e.g., 'path/to/histogram2d'
        If a list of strings is passed, get multiple 2d histograms from the
        root file.

    Returns
    -------
    histogram2d : :class:`.Hist2d` or tuple[`Hist2d`, ...]
        A Hist2d instance of the histogram data. If *histogram_names* was
        a Sequence, *histogram2d* will be a tuple of Hist2d instances.

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        import atompy as ap

        # Import one histogram and plot it
        hist = load_root_hist2d("rootfile.root", "histogram_name")
        plt.pcolormesh(*hist.for_pcolormesh)

        # Import multiple histograms and plot them
        hists = load_root_hist2d(
            "rootfile.root", ["histogram_name_1", "histogram_name_2"])
        for hist in hists:
            plt.pcolormesh(*hist.for_pcolormesh)

    """
    if isinstance(histogram_names, str):
        histogram_names = [histogram_names]
    output = []
    with uproot.open(root_filename) as file:  # type: ignore
        for h in histogram_names:
            output.append(aph.Hist2d(*file[h].to_numpy()))  # type: ignore
    return tuple(output) if len(histogram_names) > 1 else output[0]


@overload
def import_ascii_for_pcolormesh(
    fnames: str,
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    **kwargs
) -> tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64],
           npt.NDArray[np.float64]]: ...


@overload
def import_ascii_for_pcolormesh(
    fnames: Sequence[str],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    **kwargs
) -> tuple[tuple[npt.NDArray[np.float64], ...],
           tuple[npt.NDArray[np.float64], ...],
           tuple[npt.NDArray[np.float64], ...]]: ...


def import_ascii_for_pcolormesh(
    fnames: Union[str, Sequence[str]],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    **kwargs
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           tuple[tuple[npt.NDArray[np.float64], ...],
                 tuple[npt.NDArray[np.float64], ...],
                 tuple[npt.NDArray[np.float64], ...]]]:
    """
    Import 2d-histogram data and return three numpy arrays X, Y, C which are
    formatted such that they can be plotted using `plt.pcolormesh <https://
    matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html>`_

    Parameters
    ----------
    fnames : str or Sequence[str]
        The filename(s) of the data. If a list is passed, return a list
        of images and extents

    xyz_indiceds : (int, int, int), default (1, 0, 2)
        Specify columns of x, y, and z, starting at 0

    permuting : `"x"` or `"y"`
        Order of permutation of x and y in ascii file

        - `"x"`: first permutate through x-values before changing y-values
        - `"y"`: first permutate through y-values before changing x-values

    **kwargs :
        `np.loadtxt <https://numpy.org/doc/stable/reference/generated/
        numpy.loadtxt.html>`_ keyword arguments

    Returns
    -------
    X : `np.ndarray` or tuple[`np.ndarray`, ...]

    Y : `np.ndarray` or tuple[`np.ndarray`, ...]

    C : `np.ndarray` or tuple[`np.ndarray`, ...]


    Examples
    --------
    ::

        import matplotlib.pyplots as plt
        import atompy as ap

        # 1 file
        X, Y, C = ap.import_ascii_for_pcolormesh("filename.txt")
        plt.pcolormesh(X, Y, C)

        # 2 files
        X, Y, C = ap.import_ascii_for_pcolormesh(
            ["filename1.txt", "filename2.txt"]
        plt.pcolormesh(X[0], Y[0], C[0])
        plt.pcolormesh(X[1], Y[1], C[1])
    """
    histos = load_ascii_hist2d(fnames, xyz_indices, permuting, **kwargs)
    if isinstance(histos, aph.Hist2d):
        return histos.for_pcolormesh
    output_x, output_y, output_c = [], [], []
    for histo in histos:
        x, y, c = histo.for_pcolormesh
        output_x.append(x)
        output_y.append(y)
        output_c.append(c)
    return tuple(output_x), tuple(output_y), tuple(output_c)


@overload
def import_root_for_pcolormesh(
    root_filename: str,
    histogram_names: str
) -> tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64],
           npt.NDArray[np.float64]]: ...


@overload
def import_root_for_pcolormesh(
    root_filename: str,
    histogram_names: Sequence[str]
) -> tuple[tuple[npt.NDArray[np.float64], ...],
           tuple[npt.NDArray[np.float64], ...],
           tuple[npt.NDArray[np.float64], ...]]: ...


def import_root_for_pcolormesh(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]]
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           tuple[tuple[npt.NDArray[np.float64], ...],
                 tuple[npt.NDArray[np.float64], ...],
                 tuple[npt.NDArray[np.float64], ...]]]:
    """
    Import a 2d histogram from a root file to be plottable by
    `plt.pcolormesh <https://matplotlib.org/stable/api/_as_gen/
    matplotlib.pyplot.pcolormesh.html>`_.

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names : str or Sequence[str]
        The name of the histogram within the root file,
        e.g., 'path/to/histogram2d'.
        If a list of strings is passed, get multiple 2d histograms from the
        root file

    Returns
    -------
    X : `np.ndarray` or tuple[`np.ndarray`, ...]

    Y : `np.ndarray` or tuple[`np.ndarray`, ...]

    C : `np.ndarray` or tuple[`np.ndarray`, ...]

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        import atompy as ap

        # Import one histogram
        x, y, c = ap.import_root_for_pcolormesh(
            "rootfile.root", "path/to/hist")
        plt.pcolormesh(x, y, c)

        # Import multiple histograms
        xs, ys, cs = import_root_for_pcolormesh(
            "rootfile.root", ["path/to/histo1", "path/to/histo2"])
        for x, y, c in zip(xs, ys, cs):
            plt.pcolormesh(x, y, c)

    """
    histos = load_root_hist2d(root_filename, histogram_names)
    if isinstance(histos, aph.Hist2d):
        return histos.for_pcolormesh
    output_x, output_y, output_c = [], [], []
    for histo in histos:
        x, y, c = histo.for_pcolormesh
        output_x.append(x)
        output_y.append(y)
        output_c.append(c)
    return tuple(output_x), tuple(output_y), tuple(output_c)

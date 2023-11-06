import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
import uproot
from typing import Generic, TypeVar, Literal, Sequence, Union, overload
import matplotlib.pyplot as plt

DType = TypeVar("DType")


class Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:
        return super().__getitem__(key)


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
            f"{self.which}binsizes from {self.fname} are not constant. "
            f"Therefore, I cannot accurately reconstruct {self.which}edges. "
            "Use a different type of data import instead "
            "(e.g., numpy.loadtxt)"
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

    **kwargs : `numpy.savetxt` keyword args

    Examples
    --------
    ::

        import numpy as np
        import atompy.histogram as ah
        xsamples = np.random.normal(100_000)
        histogram = np.histogram(xsamples, 50)
        ah.save_ascii_hist1d(*histogram, "filename.txt")
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

    **savetxt_kwargs : `numpy.savetxt` keyword args
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
) -> tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64]]: ...


@overload
def load_ascii_hist1d(
    fnames: Sequence[str],
    **loadtxt_kwargs
) -> Array[npt.NDArray[np.float64]]: ...


def load_ascii_hist1d(
    fnames: Union[str, Sequence[str]],
    **loadtxt_kwargs
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           Array[npt.NDArray[np.float64]]]:
    """
    Load a 1d histogram from a file and return it analogous to the output
    of `numpy.histogram`

    Parameters
    ----------
    fnames: str or Sequence[str]
        Filename(s). Should contain two columns, where the first represents the
        centers of the xbins, the second the values of the histogram.

    **loadtxt_kwargs
        Keyword arguments for `numpy.loadtxt`

    Returns
    -------
    * *filenames* is str

        histogram: `np.ndarray, shape(n,)`

        edges: `np.ndarray, shape(n+1,)`

    * *filenames* is Sequence[str]

        `np.ndarray` of tuples of the above
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

        output.append((data[:, 1], edges))

    output = np.array(output, dtype=object)
    return output if len(output) > 1 else output[0]


@overload
def load_ascii_hist2d(
    fnames: str,
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    **loadtxt_kwargs
) -> tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64],
           npt.NDArray[np.float64]]: ...


@overload
def load_ascii_hist2d(
    fnames: Sequence[str],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    **loadtxt_kwargs
) -> Array[npt.NDArray[np.float64]]: ...


def load_ascii_hist2d(
    fnames: Union[str, Sequence[str]],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    **loadtxt_kwargs
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           Array[npt.NDArray[np.float64]]]:
    """
    Load a 2d histogram from a file and return it corresponding to the output
    of `numpy.histogram2d`

    Parameters
    ----------
    fnames: str or Sequence[str]
        Filename(s). Should contain three columns, where the first represents
        the centers of the xbins, the second the centers of the ybins and
        the third values of the histogram.

    xyz_indiceds: (int, int, int), default (1, 0, 2)
        Specify columns of x, y, and z, starting at 0

    permuting: "x" or "y", default "x"
        Order of permutation of x and y in ascii file
        - "x": first permute through x-values before changing y-values
        - "y": first permute through y-values before changing x-values

    **loadtxt_kwargs
        Keyword arguments for `numpy.loadtxt`

    Returns
    -------
    * *fnames* is a single string

        H: `np.ndarray, shape(nx, ny)`

        xedges: `np.ndarray, shape(nx+1,)`

        yedges: `np.ndarray, shape(ny+1,)`

    * *fnames* is a sequence

        list of tuples of the above
    """
    if permuting not in ["x", "y"]:
        raise ValueError(
            f"{permuting=}, but must be 'x' or 'y'"
        )

    if isinstance(fnames, str):
        fnames = [fnames]

    idx_x, idx_y, idx_z = xyz_indices

    output = []
    for fname in fnames:
        data = np.loadtxt(fname, **loadtxt_kwargs)

        x = np.unique(data[:, idx_x])
        y = np.unique(data[:, idx_y])

        xbinsize = x[1] - x[0]
        if not np.all(np.diff(x) - xbinsize < 1e-5):
            raise NonconstantBinsizeError(fname, "x")
        ybinsize = y[1] - y[0]
        if not np.all(np.diff(y) - ybinsize < 1e-5):
            raise NonconstantBinsizeError(fname, "y")

        xedges = np.empty(x.size + 1)
        xedges[:-1] = x - 0.5 * xbinsize
        xedges[-1] = x[-1] + 0.5 * xbinsize
        yedges = np.empty(y.size + 1)
        yedges[:-1] = y + 0.5 * ybinsize
        yedges[-1] = y[-1] + 0.5 * ybinsize

        if permuting == "x":
            z = data[:, idx_z].reshape(y.size, x.size)
        else:
            z = data[:, idx_z].reshape(x.size, y.size).T

        output.append((z, xedges, yedges))

    output = np.array(output, dtype=object)
    return output if len(output) > 1 else output[0]


@overload
def load_ascii_data1d(
    fnames: str,
    **loadtxt_kwargs
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def load_ascii_data1d(
    fnames: Sequence[str],
    **loadtxt_kwargs
) -> Array[npt.NDArray[np.float64]]: ...


def load_ascii_data1d(
    fnames: Union[str, Sequence[str]],
    **loadtxt_kwargs
) -> Union[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
           Array[npt.NDArray[np.float64]]]:
    """
    Import 1d data from an ascii file with two columns (x, y). For any other
    layout, directly use `np.loadtxt` instead.

    Parameters
    ----------
    fnames : str or Sequence[str]
        Filename(s)

    **loadtxt_kwargs
        optional `np.loadtxt` keyword arguments

    Returns
    -------
    * *fnames* is str

        x : `np.ndarray`
            Data of the first column of the file

        y : `np.ndarray`
            Data of the second column of the file

    * *fnames* is Sequence[str]

        `np.ndarray` of tuples of the above
    """
    if isinstance(fnames, str):
        x, y = np.loadtxt(fnames, **loadtxt_kwargs).T
        return x, y

    output = np.empty((len(fnames), 2), dtype=object)
    for i, fname in enumerate(fnames):
        x, y = np.loadtxt(fname, **loadtxt_kwargs).T
        output[i, 0] = x
        output[i, 1] = y
    return output


@overload
def load_ascii_data2d(
    fnames: str,
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    origin: Literal["upper", "lower", "auto"] = "auto",
    **kwargs
) -> tuple[npt.NDArray[np.float64], list[float]]: ...


@overload
def load_ascii_data2d(
    fnames: Sequence[str],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    origin: Literal["upper", "lower", "auto"] = "auto",
    **kwargs
) -> Array[npt.NDArray[np.float64]]: ...


def load_ascii_data2d(
    fnames: Union[str, Sequence[str]],
    xyz_indices: tuple[int, int, int] = (1, 0, 2),
    permuting: Literal["x", "y"] = "x",
    origin: Literal["upper", "lower", "auto"] = "auto",
    **kwargs
) -> Union[tuple[npt.NDArray[np.float64], list[float]],
           Array[npt.NDArray[np.float64]]]:
    """
    Import 2d histogram data and return an image and the extents of the image.
    If you want to work with the histogram, consider using `load_ascii_hist2d`
    instead.

    Parameters
    ----------
    filenames : str or Sequence[str]
        The filename(s) of the data. If a list is passed, return a list
        of images and extents

    xyz_indiceds: (int, int, int), default (1, 0, 2)
        Specify columns of x, y, and z, starting at 0

    permuting : "x" or "y"
        Order of permutation of x and y in ascii file
        - "x": first permutate through x-values before changing y-values
        - "y": first permutate through y-values before changing x-values

    **kwargs :
        `numpy.loadtxt` keyword arguments

    Returns
    -------
    `numpy.ndarray, numpy.ndarray` or lists thereof
        *image*, *extent*

    Notes
    -----
    Data file should have three columns, the first two specifying x or y data
    (which should match the keyword argument *order*). Third column should
    contain the corresponding z-data. The order of the z-data should match
    the keyword argument *sorting*

    Examples
    --------
    ::

        # 1 file
        data, extent = load_ascii_data2d("filename.txt")
        plt.imshow(data, extent=extent)

        # 2 files
        filenames = ["filename1.txt", "filename2.txt"]
        data = load_ascii_data2d(filenames)
        plt.imshow(data[0, 0], extent=data[0, 1])
        plt.imshow(data[1, 0], extent=data[1, 1])
    """
    if isinstance(fnames, str):
        fnames = [fnames]

    if permuting not in ["x", "y"]:
        raise ValueError(
            f"{permuting=}, but it needs to be 'x' or 'y'"
        )

    if origin not in ["upper", "lower", "auto"]:
        raise ValueError(
            f"{origin=}, but it needs to be 'upper' or 'lower'"
        )

    if origin == "auto":
        origin = plt.rcParams["image.origin"]

    idx_x, idx_y, idx_z = xyz_indices

    output = []
    for fname in fnames:
        data = np.loadtxt(fname, **kwargs)

        x = np.unique(data[:, idx_x])
        y = np.unique(data[:, idx_y])

        if permuting == "x":
            if origin == "upper":
                z = np.flip(data[:, 2].reshape(y.shape[0], x.shape[0]),
                            axis=0)
            else:
                z = data[:, 2].reshape(y.shape[0], x.shape[0])
        else:
            if origin == "upper":
                z = np.flip(data[:, idx_z].reshape(x.shape[0], y.shape[0]),
                            axis=1).T
            else:
                z = data[:, idx_z].reshape(x.shape[1], y.shape[0]).T

        binsize_x = x[1] - x[0]
        binsize_y = y[1] - y[0]
        extent = np.array(
            [np.min(x) - binsize_x / 2.0, np.max(x) + binsize_x / 2.0,
             np.min(y) - binsize_y / 2.0, np.max(y) + binsize_y / 2.0]
        )

        output.append((z, extent))

    output = np.array(output, dtype=object)
    return output if len(output) > 1 else output[0]


@overload
def load_root_data1d(
    root_filename: str,
    histogram_names: str
) -> npt.NDArray[np.float64]: ...


@overload
def load_root_data1d(
    root_filename: str,
    histogram_names: Sequence[str]
) -> Array[npt.NDArray[np.float64]]: ...


def load_root_data1d(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]]
) -> Union[npt.NDArray[np.float64],
           Array[npt.NDArray[np.float64]]]:
    """
    Import 1d histogram(s) from a root file

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names: str or Sequence thereof
        The name of the histogram(s) within the root file,
        e.g., 'path/to/histogram1d' or
        ['path/to/histogram1d_1', 'path/to/histogram1d_2']

    Returns
    -------
    data `numpy.ndarray` or list[`numpy.ndarray`]
        data[0] is x, data[1] is y. If *histogram_names* was provided as a
        sequence, list[data] is returned instead.


    Examples
    --------
    ::

        # single histogram
        data = ap.load_root_hist1d("rootfile.root",
                                   "path/to/hist1d")
        plt.plot(*data)

        # multiple histograms
        data = ap.load_root_hist1d("rootfile.root",
                                   ["path/to/hist1d_0",
                                    "path/to/hist1d_1"])
        for date in data:
            plt.plot(*date)
    """
    if isinstance(histogram_names, str):
        histogram_names = [histogram_names]
    with uproot.open(root_filename) as file:
        output = np.empty((len(histogram_names), 2), dtype=object)
        for i, h in enumerate(histogram_names):
            y, x = file[h].to_numpy()
            x = x[:-1] + 0.5 * np.diff(x)
            output[i, 0] = x
            output[i, 1] = y
    return output if len(output) > 1 else output[0]


@overload
def load_root_data2d(
    root_filename: str,
    histogram_names: str,
    origin: Literal["auto", "upper", "lower"] = "auto"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def load_root_data2d(
    root_filename: str,
    histogram_names: Sequence[str],
    origin: Literal["auto", "upper", "lower"] = "auto"
) -> Array[npt.NDArray[np.float64]]: ...


def load_root_data2d(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]],
    origin: Literal["auto", "upper", "lower"] = "auto"
) -> Union[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
           Array[npt.NDArray[np.float64]]]:
    """
    Import a 2d histogram from a root file to be plottable by
    `maptlotlib.pyplot.imshow()`

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names: str or Sequence[str]
        The name of the histogram within the root file,
        e.g., 'path/to/histogram2d'
        If a list of strings is passed, get multiple 2d histograms from the
        foot file

    Returns
    -------
    * *histogram_names* is str

        image: `numpy.ndarray`
            2d array of the z-values

        extent: tuple[float, float, float, float]
            extent of the histogram

    * *histogram_names* is Sequence[str]

        list of tuples of the above

    Examples
    --------
    ::

        # Import one histogram and plot it
        # with matplotlib.pyplot.imshow()
        data = load_root_hist2d("rootfile.root",
                                "histogram_name")
        plt.imshow(data[0], extent=data[1])

        # Import multiple histograms and
        # plot them with plt.imshow()
        data = load_root_hist2d("rootfile.root",
                                ["histogram_name_1",
                                 "histogram_name_2"])
        for date in data:
            plt.imshow(date[0], extent=date[1]),

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
        output = []
        for h in histogram_names:
            image, xedges, yedges = file[h].to_numpy()  # type: ignore
            if origin == "upper":
                image = np.flip(image.T, axis=0)
            else:
                image = image.T
            extent = np.array((np.min(xedges), np.max(xedges),
                               np.min(yedges), np.max(yedges)))

            output.append((image, extent))

        output = np.array(output, dtype=object)
        return output if len(output) > 1 else output[0]


@overload
def load_root_hist1d(
    root_filename: str,
    histogram_names: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def load_root_hist1d(
    root_filename: str,
    histogram_names: Sequence[str]
) -> Array[npt.NDArray[np.float64]]: ...


def load_root_hist1d(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]]
) -> Union[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
           Array[npt.NDArray[np.float64]]]:
    """
    Import 1d histogram(s) from a root file. Returns output equivalent to
    `numpy.histogram`, i.e. (histogram_values, edges)
    If you want (bincenters, histogram_values), use `load_root_data1d` instead

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names: str or Sequence thereof
        The name of the histogram(s) within the root file,
        e.g., 'path/to/histogram1d' or
        ['path/to/histogram1d_1', 'path/to/histogram1d_2']

    Returns
    -------
    * *histogram_names* is str

        histogram: `numpy.ndarray, shape (N,)`
            Histogram data

        edges: `numpy.ndarray, shape (N+1,)`
            edges of the bins

    * *histogram_names* is Sequence[str]

        list of tuples of the above

    Examples
    --------
    ::

        # single histogram
        hist = ap.load_root_hist1d("rootfile.root",
                                   "path/to/hist1d")
        plt.step(hist[1][1:], hist[0])

        # multiple histograms
        hists = ap.load_root_hist1d("rootfile.root",
                                    ["path/to/hist1d_0",
                                     "path/to/hist1d_1"])
        for hist in hists:
            plt.step(hist[1][1:], hist[0])
    """
    if isinstance(histogram_names, str):
        histogram_names = [histogram_names]
    with uproot.open(root_filename) as file:
        output = np.empty((len(histogram_names), 2), dtype=object)
        for i, h in enumerate(histogram_names):
            hist, edges = file[h].to_numpy()
            output[i, 0] = hist
            output[i, 1] = edges

    return output if len(output) > 1 else output[0]


@overload
def load_root_hist2d(
    root_filename: str,
    histogram_names: str,
) -> tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64],
           npt.NDArray[np.float64]]: ...


@overload
def load_root_hist2d(
    root_filename: str,
    histogram_names: Sequence[str],
) -> Array[npt.NDArray[np.float64]]: ...


def load_root_hist2d(
    root_filename: str,
    histogram_names: Union[str, Sequence[str]],
) -> Union[tuple[npt.NDArray[np.float64],
                 npt.NDArray[np.float64],
                 npt.NDArray[np.float64]],
           Array[npt.NDArray[np.float64]]]:
    """
    Import a 2d histogram from a root file equivalent to `numpy.histogram2d`

    Parameters
    ----------
    root_filename : str
        The filename of the root file,
        e.g., 'important_data.root'

    histogram_names: str or Sequence[str]
        The name of the histogram within the root file,
        e.g., 'path/to/histogram2d'
        If a list of strings is passed, get multiple 2d histograms from the
        foot file

    Returns
    -------
    * *histogram_names* is str

        H: `numpy.ndarray, shape (nx, ny)`
            2d array of the z-values

        xedges: `numpy.ndarray, shape (nx+1,)`

        yedges: `numpy.ndarray, shape (ny+1,)`

    * *histogram_names* is Sequence[str]

        `np.ndarray` of tuples of the above

    Examples
    --------
    ::

        # Import one histogram and plot it
        # with matplotlib.pyplot.imshow()
        from atompy.histogram import for_pcolormesh
        hist = load_root_hist2d("rootfile.root",
                                "histogram_name")
        plt.pcolormesh(*for_pcolormesh(*hist))

        # Import multiple histograms and
        # plot them with plt.imshow()
        hists = load_root_hist2d("rootfile.root",
                                 ["histogram_name_1",
                                  "histogram_name_2"])
        for hist in hists:
            plt.pcolormesh(*for_pcolormesh(*hist))

    """
    if isinstance(histogram_names, str):
        histogram_names = [histogram_names]

    with uproot.open(root_filename) as file:
        output = np.empty((len(histogram_names), 3), dtype=object)
        for i, h in enumerate(histogram_names):
            hist, xedges, yedges = file[h].to_numpy()
            output[i, 0] = hist
            output[i, 1] = xedges
            output[i, 2] = yedges
        return output if len(output) > 1 else output[0]

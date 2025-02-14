import uproot
import matplotlib as mpl
import inspect
import os
import pathlib 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Literal, overload, Sequence
from . import _histogram
from . import _miscellaneous as _misc
from . import _errors


def savefig(
    fname: Optional[str] = None,
    ftype: Optional[Union[str, Sequence[str]]] = None,
    fig: Optional[Figure] = None,
    **savefig_kwargs
) -> None:
    r"""
    Save a :class:`matplotlib.figure.Figure` to a file.
 
    Wraps :func:`matplotlib.pyplot.savefig`.

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure`, optional
        If ``None``, use last active figure.

    fname : str, optional
        File name (and path).

        If ``None``, uses the filename of the programs entry point.

        If a file name without a file-type extension is provided, uses
        `rcParams["savefig.format"] <https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample>`__
        unless `ftype` is provided.

        If fname ends in ``/`` or ``\``, it is assumed as a directory in which
        the output will be saved using the file name of the programs entry
        point.

    ftype : str or Sequence[str], optional
        The file type(s).

        If provided, the appropriate extension is appended to
        `fname` and the file is saved as that file type.

        If a sequence is provided, saves one file for each provided type.

        If nothing is provided, infers the file type from `fname`.

    Other Parameters
    ----------------
    **savefig_kwargs
        Keyword arguments of :func:`matplotlib.pyplot.savefig`.

    See also
    --------
    matplotlib.pyplot.savefig

    Examples
    --------
    Assume a script named ``my_plot.py`` with the contents

    .. code-block:: python
        :caption: ``my_plot.py``
        
        import atompy as ap

        plt.plot(some_data)

        ap.savefig()
        ap.savefig(ftype="pdf")
        ap.savefig("output/")
        ap.savefig("a_plot")
        ap.savefig("a_plot", ftype=("pdf", "png"))
        ap.savefig("a_plot.pdf", ftype=("pdf", "png"))

    Executing ``my_plot.py`` will save:

    - ``my_plot.png`` (assuming `plt.rcParams["savefig.format"] <https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file>`__ = "png")
    - ``my_plot.pdf``
    - ``output/my_plot.png``
    - ``a_plot.pdf``
    - ``a_plot.pdf`` and ``a_plot.png``
    - ``a_plot.pdf.pdf`` and ``a_plot.pdf.png``

    """
    fig = fig or plt.gcf()

    main_caller_fname = inspect.stack()[-1].filename
    main_caller_fname_base, _ = os.path.splitext(main_caller_fname)
    if fname is None:
        fname = main_caller_fname_base
        print(fname)
    elif fname.endswith("\\") or fname.endswith("/"):
        if fname.startswith("\\") or fname.startswith("/"):
            fname = fname[1:]
        path = pathlib.Path(main_caller_fname_base)
        fname = path.parent / fname / path.name # type: ignore

    # create directories if not present
    pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True) # type: ignore

    fname_, ftype_ = os.path.splitext(fname) # type: ignore
    if ftype is not None:
        dot = "."
        ftypes = (ftype, ) if isinstance(ftype, str) else tuple(ftype)
    else:
        dot = ""
        ftypes = (ftype_, )

    for type in ftypes:
        fig.savefig(f"{fname_}{dot}{type}", **savefig_kwargs)


def save_1d_as_txt(
    histogram: NDArray[np.float64],
    edges: NDArray[np.float64],
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
    H: NDArray[np.float64],
    xedges: NDArray[np.float64],
    yedges: NDArray[np.float64],
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
        xedges = _misc.centers_to_edges(output[0], xmin, xmax)
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


def for_pcolormesh(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        permuting: str = "x",
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
) -> _misc.PcolormeshData:
    """
    Convert xyz-data such that it can be plotted with
    :func:`matplotlib.pyplot.pcolormesh`.

    Data should look like

    ::

        x0 y0 z00
        x1 y0 z10
        x2 y0 z20
        ...
        xM yN zM0
        x0 y1 z01
        x1 y1 z11
        ...
        xM yN zMN

    Alternatively, `y` could be permuting before `x` does (see
    `permuting` keyword).

    Parameters
    ----------
    x, y, z : ndarray
        x, y, and z data.

    permuting : {``"x"``, ``"y"``}, default ``"x"``
        Specify which if `x` or `y` data permutes first (see example above).

    xmin, ymin : float, optional
        Specify the lower x (y) limit of the data in `fname`. Only necessary if
        the x (y) values in `fname` are not equally spaced. Alternatively,
        specify `xmax` (`ymax`.

    xmax, ymax : float, optional
        Specify the upper x (y) limit of the data in `fname`. Only necessary if
        the x (y) values in `fname` are not equally spaced. Alternatively,
        specify `xmin` (`ymin`). If `xmin` (`ymin`) *and* `xmax` (`ymin`) are
        specified, `xmax` (`ymax`) is not used.

    Returns
    -------
    output : :class:`.PcolormeshData`

    Examples
    --------
    
    .. code-block:: python

        xedges, yedges, counts = ap.for_pcolormesh(x, y, z)
        plt.pcoloremesh(xedges, yedges, counts)
    """
    x_ = np.unique(x)
    y_ = np.unique(y)

    xedges = _misc.centers_to_edges(x_, xmin, xmax)
    yedges = _misc.centers_to_edges(y_, ymin, ymax)

    if permuting == "x":
        z_ = z.reshape(y_.size, x_.size)
    elif permuting == "y":
        z_ = z.reshape(x_.size, y_.size).T
    else:
        msg = f"'{permuting=}', but should be 'x' or 'y'"
        raise ValueError(msg)

    return _misc.PcolormeshData(xedges, yedges, z_)


def for_imshow(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        permuting: Literal["x", "y"] = "x",
        origin: Optional[Literal["lower", "upper"]] = None
) -> _misc.ImshowData:
    """
    Convert xyz-data such that it can be plotted with
    :func:`matplotlib.pyplot.imshow`.

    Data should look like

    ::

        x0 y0 z00
        x1 y0 z10
        x2 y0 z20
        ...
        xM yN zM0
        x0 y1 z01
        x1 y1 z11
        ...
        xM yN zMN

    Alternatively, `y` could be permuting before `x` does (see
    `permuting` keyword).

    Parameters
    ----------
    x, y, z : ndarray
        x, y, and z data.

    permuting : {``"x"``, ``"y"``}, default ``"x"``
        Specify which if `x` or `y` data permutes first (see example above).

    origin : {``"lower"``, ``"upper"``}, optional
        Specify the origin of the imshow-image.
        If ``None``, use ``plt.rcParams["image.origin"]``.

    Returns
    -------
    output : :class:`.ImshowData`

    Examples
    --------
    
    .. code-block:: python

        image, extent = ap.for_imshow(x, y, z)
        plt.imshow(image, extent=extent)
    """
    try:
        xedges, yedges, H = for_pcolormesh(x, y, z, permuting)
    except _errors.UnderdeterminedBinsizeError:
        msg = "Non-constant binsize, use for_pcolormesh instead"
        raise ValueError(msg)

    origin = origin or mpl.rcParams["image.origin"]
    if origin == "lower":
        pass
    elif origin == "upper":
        H = np.flip(H, axis=0)
    else:
        msg = f"{origin=}, but it needs to be 'upper' or 'lower'"
        raise ValueError(msg)

    edges = np.array([xedges.min(), xedges.max(), yedges.min(), yedges.max()])
    return _misc.ImshowData(H, edges)



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
    origin: Optional[Literal["lower", "upper"]] = None,
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
    origin: Optional[Literal["lower", "upper"]] = None,
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
    origin: Optional[Literal["lower", "upper"]] = None,
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
    origin: Optional[Literal["lower", "upper"]] = None,
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

    origin : {``"lower"``, ``"upper"``}, optional
        Specify the origin of the imshow-image.
        If ``None``, use ``plt.rcParams["image.origin"]``.


    **loadtxt_kwargs
        Other :func:`numpy.loadtxt` keyword arguments. Useful if, e.g., you want
        to skip a certain number of lines in the text file.

    Returns
    -------
    output : :class:`.PcolormeshData`, :class:`.ImshowData` or :class:`.Hist2d` 
        Depends on `output_format`.

        - ``output_format == "pcolormesh"``: Return :class:`.PcolormeshData`.
        - ``output_format == "imshow"``: Return :class:`.ImshowData`.
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

    Alternatively, immediately unpack the loaded data into their respective
    :class:`numpy.ndarray`.

    .. code-block:: python

        x, y, z = ap.load_2d_from_txt("data.txt", output_format="pcolormesh")
        plt.pcolormesh(x, y, z)

        image, extent = ap.load_2d_from_txt("data.txt", output_format="imshow")
        plt.imshow(image, extent=extent)


    Load data as a :class:`.Hist2d"`.

    .. code-block:: python

        hist = ap.load_2d_from_txt("data.txt", output_format="Hist2d")
        data = hist.column_normalized_to_sum.for_pcolormesh
        plt.pcolormesh(data.x, data.y, data.z)


    """

    data = np.loadtxt(fname, **loadtxt_kwargs)
    x = data[:, xyz_indices[0]]
    y = data[:, xyz_indices[1]]
    z = data[:, xyz_indices[2]]

    if output_format == "Hist2d":
        xedges, yedges, H = for_pcolormesh(
            x, y, z, permuting, xmin, xmax, ymin, ymax)
        return _histogram.Hist2d(H.T, xedges, yedges)
    elif output_format == "imshow":
        return for_imshow(x, y, z, permuting, origin)
    elif output_format == "pcolormesh":
        return for_pcolormesh(
            x, y, z, permuting, xmin, xmax, ymin, ymax)
    else:
        valid_output_formats = ["imshow", "pcolormesh", "Hist2d"]
        errmsg = (
            f"{output_format=}, but it must be one of {valid_output_formats}"
        )
        raise ValueError(errmsg)

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
        - ``output_format == "imshow"``: Return :class:`.ImshowData`.
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
    
    Alternatively, immediately unpack the loaded data into their respective
    :class:`numpy.ndarray`.

    .. code-block:: python

        x, y, z = ap.load_2d_from_txt("rootfile.root", "path/to/hist",
                                      output_format="pcolormesh")
        plt.pcolormesh(x, y, z)

        image, extent = ap.load_2d_from_txt("rootfile.root", "path/to/hist",
                                            output_format="imshow")
        plt.imshow(image, extent=extent)

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

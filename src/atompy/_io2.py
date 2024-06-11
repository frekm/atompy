import uproot
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Literal, overload
from . import _histogram


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

    with uproot.open(fname) as file: # type: ignore
        histogram, edges = file[hname].to_numpy()  # type: ignore

    if output_format == "ndarray":
        output = np.empty((2, histogram.shape[0]))
        output[0] = edges[:-1] + 0.5 * np.diff(edges)
        output[1] = histogram
        return output

    if output_format == "Hist1d":
        return _histogram.Hist1d(histogram, edges)
    


if __name__ == "__main__":
    print("Hi")

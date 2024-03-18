import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from numpy.typing import NDArray
from dataclasses import dataclass
from . import _io as apio
from ._miscellaneous import get_all_dividers, ImshowData
from typing import Literal, Optional, Union, overload


@dataclass
class _Hist1dIterator:
    histogram: NDArray[np.float64]
    edges: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.index = 0

    def __iter__(self) -> "_Hist1dIterator":
        return self

    def __next__(self) -> NDArray[np.float64]:
        self.index += 1
        if self.index == 1:
            return self.histogram
        if self.index == 2:
            return self.edges
        raise StopIteration


@dataclass
class Hist1d:
    histogram: NDArray[np.float64]
    edges: NDArray[np.float64]

    def __iter__(self) -> _Hist1dIterator:
        return _Hist1dIterator(self.histogram, self.edges)

    @property
    def centers(
        self
    ) -> NDArray[np.float64]:
        """ Return centers of the histogram's bins """
        return self.edges[:-1] + 0.5 * np.diff(self.edges)

    def rebinned(
        self,
        factor: int
    ) -> "Hist1d":
        """
        Rebin histogram

        Parameters
        ----------
        factor : int
            This is how many old bins will be combined to a new bin.
            Number of old bins divided by factor must be an integer

        Returns
        -------
        new_histogram : Hist1d
            The new, rebinned histogram
        """
        old_n = self.edges.size - 1

        if old_n % factor != 0:
            raise ValueError(
                f"Invalid {factor=}. Possible factors for this "
                f"histogram are {get_all_dividers(old_n)}."
            )

        new_hist = np.empty(self.histogram.size // factor)
        for i in range(new_hist.size):
            new_hist[i] = np.sum(
                self.histogram[i * factor: i * factor + factor])

        new_edges = np.full(new_hist.size + 1, self.edges[-1])
        for i in range(new_edges.size - 1):
            new_edges[i] = self.edges[i * factor]

        return Hist1d(new_hist, new_edges)

    def save_to_file(
            self,
            fname: str,
            **kwargs
    ) -> None:
        """
        Save histogram to file (i.e., centers and bin-values)

        Parameters
        ----------
        fname : str
            Filename

        Other Parameters
        ----------------
        kwargs
            Additional keyword arguments for numpy.savetxt()
        """
        apio.save_ascii_hist1d(self.histogram, self.edges, fname, **kwargs)

    @property
    def for_step(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ Returns right edges and bin-values

        By default, `plt.step <https://matplotlib.org/stable/api/_as_gen/
        matplotlib.pyplot.step.html>`_ needs the right edges of a bin
        and the corresponding bin value. See the *where* keyword argument.

        Returns
        -------
        right_edges : `np.ndarray <https://numpy.org/doc/stable/reference/ \
        generated/numpy.ndarray.html>`_
            right edges, that is, :code:`Hist1d.edges[1:]`

        bin_values : `np.ndarray <https://numpy.org/doc/stable/reference/ \
        generated/numpy.ndarray.html>`_
            bin values, that is, :code:`Hist1d.histogram`

        Examples
        --------
        ::

            # 'hist' is a Hist1d object
            plt.step(*hist.for_step)

            # doesn't work if you specify anything else but 'pre' as a
            # plt.step keyword
            plt.step(*hist.for_step, where="mid") # this will have shifted bins
        """
        return self.edges, np.append(self.histogram[0], self.histogram)

    @property
    def for_plot(self):
        """ Returns bin-centers and bin-values

        Returns
        -------
        centers : `np.ndarray <https://numpy.org/doc/stable/reference/ \
        generated/numpy.ndarray.html>`_

        bin_values : `np.ndarray <https://numpy.org/doc/stable/reference/ \
        generated/numpy.ndarray.html>`_

        Examples
        --------
        ::

            # 'hist' is a Hist1d object
            plt.step(*hist.for_step)

            # doesn't work if you specify anything else but 'pre' as a
            # plt.step keyword
            plt.step(*hist.for_step, where="mid") # this will have shifted bins
        """
        return self.centers, self.histogram


@dataclass
class _Hist2dIterator:
    H: NDArray[np.float64]
    xedges: NDArray[np.float64]
    yedges: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.index = 0

    def __iter__(self) -> "_Hist2dIterator":
        return self

    def __next__(self) -> NDArray[np.float64]:
        self.index += 1
        if self.index == 1:
            return self.H
        if self.index == 2:
            return self.xedges
        if self.index == 3:
            return self.yedges
        raise StopIteration


@dataclass
class Hist2d:
    """
    A numpy wrapper class for the return of `np.histogram2d
    <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`_

    Parameters
    ----------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram as returned by
        `np.histogram2d <https://numpy.org/doc/stable/reference/generated/
        numpy.histogram2d.html>`_

    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension of *H*

    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension of *H*

    Examples
    --------
    :: 

        import numpy as np
        import atompy as ap
        import matplotlib.pyplot as plt

        # get some random data going
        rng = np.random.default_rng()
        sample_x, sample_y = rng.normal(size=(2, 1_000))

        # initiate a Hist2d instance
        H, x, y = np.histogram2d(sample_x, sample_y)
        hist2d = ap.Hist2d(H, x, y)
        # or do a one-liner
        hist2d = Hist2d(*np.histogram2d(sample_x, sample_y))

        # plot
        plt.pcolormesh(*hist2d.for_pcolormesh)

        # or rebin, then plot
        plt.pcolormesh(*hist2d.rebinned_x(2).for_pcolormesh)

    """
    H: NDArray[np.float64]
    xedges: NDArray[np.float64]
    yedges: NDArray[np.float64]

    def __iter__(self) -> _Hist2dIterator:
        return _Hist2dIterator(self.H, self.xedges, self.yedges)

    @property
    def xcenters(self) -> NDArray[np.float64]:
        """
        Get center of bins along first dimension of *H*

        Returns
        -------
        xcenters : `np.ndarray <https://numpy.org/doc/stable/reference/ \
        generated/numpy.ndarray.html>`_
        """
        return self.xedges[:-1] + 0.5 * np.diff(self.xedges)

    @property
    def ycenters(self) -> NDArray[np.float64]:
        """
        Get center of bins along second dimension of *H*

        Returns
        -------
        xcenters : `np.ndarray <https://numpy.org/doc/stable/reference/ \
        generated/numpy.ndarray.html>`_
        """
        return self.yedges[:-1] + 0.5 * np.diff(self.yedges)

    def rebinned_x(self, factor: int) -> "Hist2d":
        """
        Rebin x-dimension of histogram.

        Return a *Hist2d* where the first dimension was rebinned with a
        factor of *factor*

        Parameters
        ----------
        factor : int
            This is how many old bins will be combined to a new bin. The number
            of old bins divided by *factor* must be integer.

        Returns
        -------
        rebinned_hist2d : Hist2d
            the new (rebinned) histogram
        """
        old_n = self.xedges.size - 1
        if old_n % factor != 0:
            raise ValueError(
                f"Invalid {factor=}. Possible factors for this "
                f"histogram are {get_all_dividers(old_n)}."
            )
        new_hist = np.empty((self.H.shape[0] // factor, self.H.shape[1]))
        for i in range(new_hist.shape[0]):
            new_hist[i, :] = np.sum(
                self.H[i * factor:i * factor + factor, :], axis=0)
        new_xedges = np.full(new_hist.shape[0] + 1, self.xedges[-1])
        for i in range(new_xedges.size):
            new_xedges[i] = self.xedges[i * factor]
        return Hist2d(new_hist, new_xedges, self.yedges)

    def rebinned_y(self, factor: int) -> "Hist2d":
        """
        Rebin y-dimension of histogram.

        Return a *Hist2d* where the second dimension was rebinned with a
        factor of *factor*

        Parameters
        ----------
        factor : int
            This is how many old ybins will be combined to a new bin.
            Number of old ybins divided by factor must be an integer

        Returns
        -------
        rebinned_hist2d : Hist2d
            the new (rebinned) histogram
        """
        old_n = self.yedges.size - 1
        if old_n % factor != 0:
            raise ValueError(
                f"Invalid {factor=}. Possible factors for this "
                f"histogram are {get_all_dividers(old_n)}."
            )
        new_hist = np.empty((self.H.shape[0], self.H.shape[1] // factor))
        for i in range(new_hist.shape[1]):
            new_hist[:, i] = np.sum(
                self.H[:, i * factor:i * factor + factor], axis=1)
        new_yedges = np.full(new_hist.shape[1] + 1, self.yedges[-1])
        for i in range(new_yedges.size):
            new_yedges[i] = self.yedges[i * factor]
        return Hist2d(new_hist, self.xedges, new_yedges)

    @property
    def for_pcolormesh(
        self
    ) -> tuple[npt.NDArray[np.float64],
               npt.NDArray[np.float64],
               npt.NDArray[np.float64]]:
        """
        Return such that it can be plotted using \
        `plt.pcolormesh <https://matplotlib.org/stable/api/_as_gen/ \
        matplotlib.pyplot.pcolormesh.html>`_

        The input order for `pcolormesh` is xedges, yedges, H.T

        Returns
        -------
        xedges : `np.ndarray <https://numpy.org/doc/stable/reference/generated/\
        numpy.ndarray.html>`_

        yedges : `np.ndarray <https://numpy.org/doc/stable/reference/generated/\
        numpy.ndarray.html>`_

        H.T : `np.ndarray <https://numpy.org/doc/stable/reference/generated/\
        numpy.ndarray.html>`_
            Transposed matrix of input *Hist2d.H*

        Examples
        --------
        ::

            hist = Hist2d(*np.histogram2d(xsamples, ysamples))
            plt.pcolormesh(*hist.for_pcolormesh)
        """
        return self.xedges, self.yedges, self.H.T

    @property
    def for_imshow(
        self,
    ) -> ImshowData:
        """
        Return corresponding :class:`.ImshowData` object.

        Assumes that the origin of the image is specified by 
        `matplotlib.rcParams["image.origin"]`

        Returns
        -------
        imshow_data : :class:`.ImshowData`
            A data type storing an image and the extent of the image.

            - :code:`imshow_data.image`: 2d pixel array
            - :code:`imshow_data_extent`: the extents of the image

        Examples
        --------
        ::

            hist = Hist2d(*np.histogram2d(xsamples, ysamples))
            image, extent = hist.for_imshow
            plt.imshow(image, extent=extent)

            # or (calling *imshow_data* returns a dictionary)
            plt.imshow(**hist.for_imshow())

            # or
            imshow_data = hist.for_imshow
            plt.imshow(imshow_data.image, extent=imshow_data.extent)
        """
        origin = plt.rcParams["image.origin"]
        if origin == "lower":
            H_ = self.H.T
        else:
            H_ = np.flip(self.H.T, axis=0)
        edges = [self.xedges.min(),
                 self.xedges.max(),
                 self.yedges.min(),
                 self.yedges.max()]
        return ImshowData(H_, np.array(edges))

    def within_xrange(
        self,
        xrange: tuple[float, float],
        keepdims: bool = False
    ) -> "Hist2d":
        """
        Apply a gate along x.

        Return a histogram where *xrange[0]* <= xedges <= *xrange[1]*

        Parameters
        ----------
        xrange : (float, float)
            lower/upper xedge to be included in the final histogram. Edges
            are included.

        keepdims : bool, default False
            If *True*, keep original dimensions, i.e., the length of xedges
            won't change

        Returns
        -------
        new_histogram : Hist2d
        """
        idx = np.flatnonzero(np.logical_and(
            self.xedges >= xrange[0], self.xedges <= xrange[1]))
        if keepdims:
            _H = np.zeros_like(self.H)
            _H[idx[0]:idx[-1], :] = self.H[idx[0]:idx[-1], :]
            rtn = Hist2d(_H, self.xedges, self.yedges)

        else:
            rtn = Hist2d(
                self.H[idx[0]:idx[-1], :],
                self.xedges[idx[0]:idx[-1] + 1],
                self.yedges)
        return rtn

    def within_yrange(
        self,
        yrange: tuple[float, float],
        keepdims: bool = False
    ) -> "Hist2d":
        """
        Apply a gate along y.

        Return a histogram where *yrange[0]* <= yedges <= *yrange[1]*

        Parameters
        ----------
        yrange : (float, float)
            lower/upper xedge to be included in the final histogram. Edges
            are included.

        keepdims : bool, default False
            If *True*, keep original dimensions, i.e., the length of yedges
            won't change

        Returns
        -------
        new_histogram : Hist2d
        """
        idx = np.flatnonzero(np.logical_and(
            self.yedges >= yrange[0], self.yedges <= yrange[1]))
        if keepdims:
            _H = np.zeros_like(self.H)
            _H[:, idx[0]:idx[-1]] = self.H[:, idx[0]:idx[-1]]
            rtn = Hist2d(_H, self.xedges, self.yedges)

        else:
            rtn = Hist2d(
                self.H[:, idx[0]:idx[-1]],
                self.xedges,
                self.yedges[idx[0]:idx[-1] + 1])
        return rtn

    @property
    def projected_onto_x(self) -> "Hist1d":
        """
        Project histogram onto its x-axis

        Returns
        -------
        hist1d : :class:`.Hist1d`
            A 1D histogram where the *bins* are the x-bins of the original
            2D histogram, and the *bin-values* are the projection.

        Examples
        --------

        .. plot:: _examples/projection_x.py
            :include-source:
        """
        return Hist1d(np.sum(self.H, axis=1), self.xedges)

    @property
    def prox(self) -> "Hist1d":
        """
        Alias for :meth:`.projected_onto_x`.
        """
        return self.projected_onto_x

    @property
    def projected_onto_y(self) -> "Hist1d":
        """
        Project histogram onto its y-axis

        Returns
        -------
        hist1d : :class:`.Hist1d`
            A 1D histogram where the *bins* are the x-bins of the original
            2D histogram, and the *bin-values* are the projection.
        """
        return Hist1d(np.sum(self.H, axis=0), self.yedges)

    @property
    def proy(self) -> "Hist1d":
        """
        Alias for :meth:`.projected_onto_y`.
        """
        return self.projected_onto_y

    def __get_profile(
        self,
        counts: NDArray,
        bin_centers: NDArray,
        option: Literal["", "s", "i", "g"] = ""
    ):
        """
        See `TProfile <https://root.cern.ch/doc/master/classTProfile.html>`_
        of the ROOT Data Analysis Framework 
        """
        H = np.sum(counts * bin_centers)
        E = np.sum(counts * bin_centers**2)
        W = np.sum(counts)
        h = H / W
        s = np.sqrt(E / W - h**2)
        e = s / np.sqrt(W)
        if option == "":
            out = e
        elif option == "s":
            out = s
        elif option == "i":
            raise NotImplementedError
        elif option == "g":
            out = 1. / np.sqrt(W)
        else:
            msg = f"{option=}, but it needs to be '', 's', 'i', or 'g'"
            raise ValueError(msg)
        return h, out

    def get_profile_along_x(
        self,
        option: Literal["", "s", "i", "g"] = ""
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Get the x-profile.

        Calculate the mean value and its error per column. Which type of error
        is controled with *option*.

        Parameters
        ----------
        option : :code:`""`, :code:`"s"`, :code:`"i"`, or :code:`"g"`, default :code:`""`
            Control type of errors, see *Notes*.

            - :code:`option == ""`:  Error of the mean of all Y values
            - :code:`option == "s"`: Standard deviation of all Y
            - :code:`option == "i"`: See *Notes*
            - :code:`option == "g"`: Error of a weighted mean for combining
              measurements with variances of :math:`w`.


        Returns
        -------
        h : `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_

            Mean values of each Y (i.e., column)

        errors : `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_

            Errors of the mean values. Depends on *option*.

        Notes
        -----
        See `TProfile <https://root.cern.ch/doc/master/classTProfile.html>`_
        of the ROOT Data Analysis Framework.

        For a histogram :math:`X` vs :math:`Y`, the following is calculated for
        each :math:`X` value.

        .. math::

            \begin{align}
            H(j) &= \sum w Y \\
            E(j) &= \sum w Y^2 \\
            W(j) &= \sum w \\
            h(j) &= H(j) / W(j) \\
            s(j) &= \sqrt{E(j) / W(j) - h(j)^2} \\
            e(j) &= s(j) / \sqrt{W(j)}
            \end{align}

        Here, :math:`w` are the counts of bin `j`.

        Examples
        --------

        .. plot:: _examples/profile_x.py
            :include-source:

        """
        means = np.zeros(self.H.shape[0])
        errors = np.zeros(self.H.shape[0])
        for idx_col in range(self.H.shape[0]):
            col = self.H[idx_col]
            mean, error = self.__get_profile(col, self.ycenters, option=option)
            means[idx_col] = mean
            errors[idx_col] = error
        return means, errors

    @property
    def profile_along_x(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Alias for :meth:`get_profile_along_x(option="")  \
        <.Hist2d.get_profile_along_x>`
        """
        return self.get_profile_along_x(option="")

    def get_profile_along_y(
        self,
        option: Literal["", "s", "i", "g"] = ""
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        r"""
        Get the y-profile.

        Calculate the mean value and its error per row. Which type of error
        is controled with *option*.

        Parameters
        ----------
        option : :code:`""`, :code:`"s"`, :code:`"i"`, or :code:`"g"`, default :code:`""`
            Control type of errors, see *Notes*.

            - :code:`option == ""`:  Error of the mean of all Y values
            - :code:`option == "s"`: Standard deviation of all Y
            - :code:`option == "i"`: See *Notes*
            - :code:`option == "g"`: Error of a weighted mean for combining
              measurements with variances of :math:`w`.


        Returns
        -------
        h : `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_

            Mean values of each X (i.e., row)

        errors : `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_

            Errors of the mean values. Depends on *option*.

        Notes
        -----
        See `TProfile <https://root.cern.ch/doc/master/classTProfile.html>`_
        of the ROOT Data Analysis Framework.

        For a histogram :math:`X` vs :math:`Y`, the following is calculated for
        each :math:`Y` value.

        .. math::

            \begin{align}
            H(j) &= \sum w X \\
            E(j) &= \sum w X^2 \\
            W(j) &= \sum w \\
            h(j) &= H(j) / W(j) \\
            s(j) &= \sqrt{E(j) / W(j) - h(j)^2} \\
            e(j) &= s(j) / \sqrt{W(j)}
            \end{align}

        Here, :math:`w` are the counts of bin `j`.

        Examples
        --------

        .. plot:: _examples/profile_y.py
            :include-source:

        """
        means = np.zeros(self.H.shape[1])
        errors = np.zeros(self.H.shape[1])
        for idx_row in range(self.H.shape[1]):
            row = self.H[:, idx_row]
            mean, error = self.__get_profile(row, self.xcenters, option=option)
            means[idx_row] = mean
            errors[idx_row] = error
        return means, errors

    @property
    def profile_along_y(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Alias for :meth:`get_profile_along_y(option="")  \
        <.Hist2d.get_profile_along_y>`
        """
        return self.get_profile_along_y(option="")

    @property
    def without_zeros(
        self
    ) -> "Hist2d":
        """
        Replace zeros with :code:`None`, removing them from colormaps

        Examples
        --------

        .. plot:: _examples/without_zeros.py
            :include-source:

        """
        _H = self.H.copy()
        _H[_H == 0] = None
        return Hist2d(_H, self.xedges, self.yedges)

    def save_to_file(
        self,
        fname: str,
        **kwargs
    ) -> None:
        """
        Save the histogram to a file.

        The first column in the file will be y, the second x, the third z.
        (this is chosen as such because the the standard hist2ascii-macro of 
        our group outputs this order)

        Parameters
        ----------
        fname : str
            Filename, including filetype

        Other Parameters
        ----------------
        **kwargs : `np.savetxt <https://numpy.org/doc/stable/reference/ \
            generated/numpy.savetxt.html>` keyword arguments
        """
        apio.save_ascii_hist2d(
            self.H, self.xedges, self.yedges, fname, **kwargs)

    @property
    def column_normalized_to_sum(self) -> "Hist2d":
        """ Normalize each column to their sum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the columns are normalized.
        """
        return Hist2d(
            self.H / self.H.sum(axis=1, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def column_normalized_to_max(self) -> "Hist2d":
        """ Normalize each column to their maximum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the columns are normalized.
        """
        return Hist2d(
            self.H / self.H.max(axis=1, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def row_normalized_to_sum(self) -> "Hist2d":
        """ Normalize each row to their sum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the rows are normalized.
        """
        return Hist2d(
            self.H / self.H.sum(axis=0, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def row_normalized_to_max(self) -> "Hist2d":
        """ Normalize each row to their maximum.

        Returns
        -------
        new_hist2d : :class:`.Hist2d`
            A new histogram where the rows are normalized.
        """
        return Hist2d(
            self.H / self.H.max(axis=0, keepdims=True),
            self.xedges,
            self.yedges)


if __name__ == "__main__":
    print("achwas")

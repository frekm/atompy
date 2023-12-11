import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from numpy.typing import NDArray
from dataclasses import dataclass
from ._io import save_ascii_hist1d, save_ascii_hist2d
from ._miscellaneous import get_all_dividers


@dataclass
class Hist1d:
    histogram: NDArray[np.float64]
    edges: NDArray[np.float64]

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
        save_ascii_hist1d(self.histogram, self.edges, fname, **kwargs)


@dataclass
class Hist2d:
    """
    A numpy wrapper class for the return of `numpy.histogram2d
    <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`_

    Parameters
    ----------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram as returned by `numpy.histogram2d
        <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`_

    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension of *H*

    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension of *H*
    """
    H: NDArray[np.float64]
    xedges: NDArray[np.float64]
    yedges: NDArray[np.float64]

    @property
    def xcenters(self) -> NDArray[np.float64]:
        """ Get center of bins along first dimension of *H* """
        return self.xedges[:-1] + 0.5 * np.diff(self.xedges)

    @property
    def ycenters(self) -> NDArray[np.float64]:
        """ Get center of bins along second dimension of *H* """
        return self.yedges[:-1] + 0.5 * np.diff(self.yedges)

    def rebinned_x(self, factor: int) -> "Hist2d":
        """
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
        Return such that it can be plotted using
        `plt.pcolormesh
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html>`_

        The input order for `pcolormesh` is xedges, yedges, H.T

        Returns
        -------
        xedges: np.ndarray

        yedges: np.ndarray

        H.T: np.ndarray
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
    ) -> tuple[npt.NDArray[np.float64], list[float]]:
        """
        Return an image and the extents of the image. 
        Assumes that the origin of the image is specified by 
        `matplotlib.rcParams["image.origin"]`

        Returns
        -------
        image: `np.ndarray`
            2d pixel map

        extent: [float, float, float, float]
            xmin, xmax, ymin, ymax

        Examples
        --------
        ::

            hist = Hist2d(*np.histogram2d(xsamples, ysamples))
            image, extent = hist.for_imshow
            plt.imshow(image, extent=extent)
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
        return H_, edges

    def within_xrange(
        self,
        xrange: tuple[float, float],
        keepdims: bool = False
    ) -> "Hist2d":
        """
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
        """
        return Hist1d(np.sum(self.H, axis=1), self.xedges)

    @property
    def prox(self) -> "Hist1d":
        """
        Alias for *projected_onto_x*
        """
        return self.projected_onto_x

    @property
    def projected_onto_y(self) -> "Hist1d":
        """
        Project histogram onto its y-axis
        """
        return Hist1d(np.sum(self.H, axis=0), self.yedges)

    @property
    def proy(self) -> "Hist1d":
        """
        Alias for *projected_onto_x*
        """
        return self.projected_onto_y

    def __get_profile(
        self,
        counts: NDArray,
        bin_centers: NDArray
    ):
        """ See ROOT `TProfile` """
        H = np.sum(counts * bin_centers)
        E = np.sum(counts * bin_centers**2)
        W = np.sum(counts)
        h = H / W
        s = np.sqrt(E / W - h**2)
        e = s / np.sqrt(W)
        return h, e

    @property
    def profile_along_x(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the x-profile, that is, the mean value and its error per column

        Returns
        -------
        mean : ndarray
            mean values per column

        error : ndarray
            standard errors of the mean values

        Notes
        -----
        See `TProfile` of the ROOT Data Analysis Framework 
        """
        means = np.zeros(self.H.shape[0])
        errors = np.zeros(self.H.shape[0])
        for idx_col in range(self.H.shape[0]):
            col = self.H[:, idx_col]
            mean, error = self.__get_profile(col, self.ycenters)
            means[idx_col] = mean
            errors[idx_col] = error
        return means, errors

    @property
    def profile_along_y(
        self,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        Get the y-profile, that is, the mean value and its error per row

        Returns
        -------
        mean : ndarray
            mean values per column

        error : ndarray
            standard errors of the mean values

        Notes
        -----
        See `TProfile` of the ROOT Data Analysis Framework 
        """
        means = np.zeros(self.H.shape[1])
        errors = np.zeros(self.H.shape[1])
        for idx_row in range(self.H.shape[1]):
            row = self.H[idx_row]
            mean, error = self.__get_profile(row, self.xcenters)
            means[idx_row] = mean
            errors[idx_row] = error
        return means, errors

    @property
    def without_zeros(
        self
    ) -> "Hist2d":
        """ Replace zeros with None, removing them from colorsmaps """
        _H = self.H.copy()
        _H[_H == 0] = None
        return Hist2d(_H, self.xedges, self.yedges)

    def save_to_file(
        self,
        fname: str,
        **savetxt_kwargs
    ) -> None:
        """
        The first column in the file will be y, the second x, the third z.
        (this is chosen as such because the the standard hist2ascii-macro of 
        our group outputs this order)

        Parameters
        ----------
        fname : str
            Filename, including filetype

        **savetxt_kwargs : `numpy.savetxt` keyword args
        """
        save_ascii_hist2d(
            self.H, self.xedges, self.yedges, fname, **savetxt_kwargs)

    @property
    def column_normalized_to_sum(self) -> "Hist2d":
        """ Normalize each column to their sum """
        return Hist2d(
            self.H / self.H.sum(axis=1, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def column_normalized_to_max(self) -> "Hist2d":
        """ Normalize each column to their maximum """
        return Hist2d(
            self.H / self.H.max(axis=1, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def row_normalized_to_sum(self) -> "Hist2d":
        """ Normalize each row to their sum """
        return Hist2d(
            self.H / self.H.sum(axis=0, keepdims=True),
            self.xedges,
            self.yedges)

    @property
    def row_normalized_to_max(self) -> "Hist2d":
        """ Normalize each row to their maximum """
        return Hist2d(
            self.H / self.H.max(axis=0, keepdims=True),
            self.xedges,
            self.yedges)


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    xsamples = rng.normal(size=1000)
    ysamples = rng.normal(size=1000)

    hist = Hist2d(*np.histogram2d(xsamples, ysamples))

    hist.save_to_file("bla")

    plt.pcolormesh(
        *hist.row_normalized_to_sum.for_pcolormesh)
    plt.show()

import numpy as np
import numpy.typing as npt
import time
from typing import overload, Union
from .._vector import Vector
from .._histogram import Hist1d


def subtract_binding_energy(
    pin: Vector, Ebind: Union[float, npt.NDArray[np.float64]]
) -> Vector:
    """Substracts binding energy from p, conserves direction of p

    Parameters
    ----------
    pin : :class:`.Vector`
        ingoing momenta
    Ebind : float or `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        binding energy in a.u.

    Returns
    -------
    vectors : :class:`.Vector`
        shortened momentum vector
    """
    if isinstance(Ebind, np.ndarray) and len(Ebind) != len(pin):
        m = f"Length mismatch of Ebind ({len(Ebind)}) and pin ({len(pin)})"
        raise ValueError(m)

    radicands = 2 * (pin.mag**2 / 2.0 - Ebind)
    pmag_new = np.array(
        [np.sqrt(radicand) if radicand > 0 else -1.0 for radicand in radicands]
    )

    thetas = pin.theta
    phis = pin.phi
    pout = np.array(
        [
            [
                p * np.sin(theta) * np.cos(phi),
                p * np.sin(theta) * np.sin(phi),
                p * np.cos(theta),
            ]
            for p, theta, phi in zip(pmag_new, thetas, phis)
            if p > 0.0
        ]
    )

    return Vector(pout)


@overload
def rho_p_microcanonical(
    pmag: float, E_bind: float, normalize: bool = True
) -> float: ...


@overload
def rho_p_microcanonical(
    pmag: npt.NDArray[np.float64], E_bind: float, normalize: bool = True
) -> npt.NDArray[np.float64]: ...


def rho_p_microcanonical(
    pmag: Union[float, npt.NDArray[np.float64]], E_bind: float, normalize: bool = True
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Momentum distribution of one component in hydrogen-like system
    from Abrines Proc Phys. Soc 88 861 (1966)

    Parameters
    ----------
    pmag : float or `np.narray`
        momentum magnitude in a.u.

    E_bind : float
        binding energy in a.u.

    normalize : bool, default True
        if True, normalize resulting distribution to maximum

    Returns
    -------
    float or `np.ndarray`
    """
    p0 = np.sqrt(2.0 * E_bind)
    out = 8.0 * p0**5 / np.pi**2 / (pmag**2 + p0**2) ** 4
    return out / np.amax(out) if normalize else out


def mom_init_distr_elec(size: int, E_bind_au: float, scale: float = 3.0) -> Vector:
    """
    Dice-throw three momentum components following a microcanonical
    distribution

    Parameters
    ----------
    size : int
        Number of throws

    E_bind : float
        Binding energy in a.u.

    scale : float
        scale factor for maximum momentum magnitude (max_mom = p0 * scale)

    Returns
    -------
    `atompy.vector.Vector`
        Momentum vectors of the distribution
    """
    t0 = time.time()
    line0 = "Generating random microcanonical momentum distribution " f"(size {size}). "

    succesful_throws = 0
    p = np.zeros((size, 3))
    while succesful_throws < size:
        line = "\r" + line0 + "%.0lf percent done." % (100.0 * succesful_throws / size)
        print(line, end="")

        buffer = size - succesful_throws

        p0 = np.sqrt(2.0 * E_bind_au)

        pmag = np.random.uniform(0, p0 * scale, buffer)

        density = 8.0 * p0**5 / np.pi**2 / (pmag**2 + p0**2) ** 4 * pmag**2
        density /= np.max(density)

        second = np.random.random(buffer)

        pmag = np.ma.compressed(np.ma.masked_array(pmag, mask=second >= density))

        theta = np.arccos(2.0 * np.random.random(pmag.size) - 1.0)
        phi = 2.0 * np.pi * np.random.random(pmag.size)

        p[succesful_throws : succesful_throws + pmag.size, 0] = (
            pmag * np.sin(theta) * np.cos(phi)
        )
        p[succesful_throws : succesful_throws + pmag.size, 1] = (
            pmag * np.sin(theta) * np.sin(phi)
        )
        p[succesful_throws : succesful_throws + pmag.size, 2] = pmag * np.cos(theta)

        succesful_throws += pmag.size

    t1 = time.time()

    print(f"\r{line0}Total runtime: {t1-t0:.2f}s")

    return Vector(p)


def mom_init_distr_elec_mol(
    distr_atomic: Vector, stretch_factor: float
) -> tuple[Vector, Vector]:
    """
    Create molecular momentum distribution

    Parameters
    ----------
    distr_atomic : `atompy.vector.Vector`
        The (atomic) distribution

    stretch_factor : float
        The factor how much the distribution will be stretched along that axis

    Returns
    -------
    `atompy.vector.Vector`
        The new momentum distribution

    `atompy.vector.Vector`
        The distribution of directions perpendicular to the directions along
        which the stretch factor was applied

    Notes
    -----
    Creates a isotropic distribution of molecular orientations
    Takes the atomic distribution and stretches y and z component of it. This
    would mean that all molecules are aligned along x
    Rotate stretched atomic distribution and rotates it such that it
    corresponds to the isotropic distribution of molecules
    """
    N = distr_atomic.ndarray.shape[0]

    print("Creating molecular orbitals... ")
    t0 = time.time()

    # stretch along x and y, i.e., molecule is aligned along z
    distr_molecular = distr_atomic.ndarray.copy()
    distr_molecular[:, :2] = distr_molecular[:, :2] * stretch_factor
    distr_molecular = Vector(distr_molecular)

    # rotate them randomly
    theta = np.arccos(2.0 * np.random.random(N) - 1)
    phi = 2.0 * np.pi * np.random.random(N)

    molecular_orientation = np.zeros(distr_molecular.ndarray.shape)
    molecular_orientation[:, 0] = np.sin(theta) * np.cos(phi)
    molecular_orientation[:, 1] = np.sin(theta) * np.sin(phi)
    molecular_orientation[:, 2] = np.cos(theta)

    distr_molecular = distr_molecular.rotated_around_y(theta).rotated_around_z(phi)

    t1 = time.time()

    print(f"Done. Total runtime: {t1-t0:.2f}s")

    return distr_molecular, Vector(molecular_orientation)


def get_ffunction_histos(
        theta_e: npt.NDArray[np.float64],
        phi_e: npt.NDArray[np.float64],
        theta_mol: npt.NDArray[np.float64],
        nbins: int,
) -> tuple[Hist1d, Hist1d, Hist1d, Hist1d]:
    """
    Parameters
    ----------
    theta_e : ndarray
        Polar angle of electron momentum vector to molecular axis in rad.
    
    phi_e : ndarray
        Azimuthal angle of electron momentum to molecular axis in rad.

    theta_mol : ndarray
        Polar angle of molecular axis and polarisation axis in rad.

    nbins : int
        Number of bins in the F-Function histograms.

    Returns
    -------
    F00, F20, F21, F22 : :class:`.Hist1d`
        F-Function histograms of FLN(theta_e).
    """
    if (
        np.min(theta_e) < 0.0 or np.max(theta_e) > 2*np.pi or
        np.min(theta_mol) < 0.0 or np.max(theta_mol) > 2*np.pi
    ):
        raise ValueError("polar angles must be between 0 and pi")

    thetarange = (0.0, np.pi)

    #######
    # F00 #
    #######
    f00 = Hist1d(*np.histogram(theta_e, bins=nbins, range=thetarange))
    f00.histogram = f00.normalized_to_integral.histogram / 2.0

    #######
    # F20 #
    #######
    # get indices where 60deg < theta_mol < 120deg
    i = np.flatnonzero(np.logical_and(
        theta_mol > np.deg2rad(60.0),
        theta_mol < np.deg2rad(120.0)
    ))
    f20 = Hist1d(*np.histogram(theta_e[i], bins=nbins, range=thetarange))
    f20.histogram = f20.normalized_to_integral.histogram
    f20.histogram = (f00.histogram - f20.histogram) / 0.375

    #######
    # F22 #
    #######
    i = np.flatnonzero(np.abs(np.cos(phi_e)) > 0.2)
    j = np.digitize(theta_e, f00.edges)
    weights = (1 - 2.0*f00.histogram[j]) / np.cos(2.0*phi_e)
    f22 = Hist1d(*np.histogram(
        theta_e[i],
        bins=nbins,
        range=thetarange,
        weights=weights
    ))
    f22.histogram = f22.normalized_to_integral.histogram / 4.0

    #######
    # F21 #
    #######
    weights = np.full(theta_e.shape, 1)
    weights[theta_e > np.deg2rad(90)] = -1
    f21 = Hist1d(*np.histogram(
        theta_e,
        bins=nbins,
        range=thetarange,
        weights=weights
    ))
    f21 = f21.normalized_to_integral

    return f00, f20, f21, f22







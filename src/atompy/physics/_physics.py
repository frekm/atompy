import numpy as np
import numpy.typing as npt
import time
from typing import overload, Union
from .._vector import Vector


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

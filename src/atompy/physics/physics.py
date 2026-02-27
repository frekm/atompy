import numpy as np
import numpy.typing as npt
import time
from typing import overload, Union
from .._vectors import Vector, VectorArray

from .._utils import sample_distribution_func


@overload
def subtract_binding_energy(
    p_in: Vector, Ebind: Union[float, npt.NDArray[np.float64]]
) -> Vector: ...
@overload
def subtract_binding_energy(
    p_in: VectorArray, Ebind: Union[float, npt.NDArray[np.float64]]
) -> VectorArray: ...


def subtract_binding_energy(
    p_in: Vector | VectorArray, Ebind: Union[float, npt.NDArray[np.float64]]
) -> Vector | VectorArray:
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
    is_vector_array = isinstance(p_in, VectorArray)
    if not is_vector_array:
        p_in = VectorArray(p_in)
    if isinstance(Ebind, np.ndarray) and len(Ebind) != len(p_in):
        m = f"Length mismatch of Ebind ({len(Ebind)}) and p_in ({len(p_in)})"
        raise ValueError(m)

    radicands = 2 * (p_in.mag() ** 2 / 2.0 - Ebind)
    pmag_new = np.array(
        [np.sqrt(radicand) if radicand > 0 else -1.0 for radicand in radicands]
    )

    thetas = p_in.theta()
    phis = p_in.phi()
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

    out = VectorArray(pout)
    return out if is_vector_array else out[0]


def rho_p_microcanonical(
    pmag: npt.ArrayLike, E_bind: float, normalize_to_max: bool = True
) -> npt.NDArray[np.float64]:
    """
    Momentum distribution of one component in hydrogen-like system
    from Abrines Proc Phys. Soc 88 861 (1966)

    Parameters
    ----------
    pmag : array_like
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
    out = 8.0 * p0**5 / np.pi**2 / (np.asarray(pmag) ** 2 + p0**2) ** 4
    return out / np.amax(out) if normalize_to_max else out


def mom_init_distr_elec(
    size: int, E_bind_au: float, scale: float = 3.0, rng_seed: int | None = None
) -> VectorArray:
    """
    Dice-throw three momentum components following a microcanonical
    distribution

    Parameters
    ----------
    size : int
        Number of throws

    E_bind_au : float
        Binding energy in a.u.

    scale : float
        scale factor for maximum momentum magnitude (max_mom = p0 * scale)

    Returns
    -------
    `atompy.vector.Vector`
        Momentum vectors of the distribution
    """
    p0 = np.sqrt(2.0 * E_bind_au)

    def f(pmag):
        density = 8.0 * p0**5 / np.pi**2 / (pmag**2 + p0**2) ** 4 * pmag**2
        return density / np.amax(density)

    rng = np.random.default_rng(rng_seed)

    pmags = sample_distribution_func(f, size, (0.0, p0 * scale), (0.0, 1.0), rng)
    thetas = np.arccos(rng.uniform(-1.0, 1.0))
    phis = rng.uniform(0.0, 2.0 * np.pi, size)

    momenta = np.empty((size, 3))
    momenta[:, 0] = pmags * np.sin(thetas) * np.cos(phis)
    momenta[:, 1] = pmags * np.sin(thetas) * np.sin(phis)
    momenta[:, 2] = pmags * np.cos(thetas)

    return VectorArray(momenta)


def mom_init_distr_elec_mol(
    distr_atomic: VectorArray, stretch_factor: float
) -> tuple[VectorArray, VectorArray]:
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
    N = distr_atomic.asarray().shape[0]

    print("Creating molecular orbitals... ")
    t0 = time.time()

    # stretch along x and y, i.e., molecule is aligned along z
    distr_molecular = distr_atomic.asarray().copy()
    distr_molecular[:, :2] = distr_molecular[:, :2] * stretch_factor
    distr_molecular = VectorArray(distr_molecular)

    # rotate them randomly
    theta = np.arccos(2.0 * np.random.random(N) - 1)
    phi = 2.0 * np.pi * np.random.random(N)

    molecular_orientation = np.zeros(distr_molecular.asarray().shape)
    molecular_orientation[:, 0] = np.sin(theta) * np.cos(phi)
    molecular_orientation[:, 1] = np.sin(theta) * np.sin(phi)
    molecular_orientation[:, 2] = np.cos(theta)

    distr_molecular = distr_molecular.rotate(theta, axis=(0, 1, 0)).rotate(
        phi, (0, 0, 1)
    )

    t1 = time.time()

    print(f"Done. Total runtime: {t1-t0:.2f}s")

    return distr_molecular, VectorArray(molecular_orientation)

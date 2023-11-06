from typing import overload, Union
import numpy as np
import numpy.typing as npt
import time


###############################################################################
###############################################################################
###############################################################################
# Physics stuff
###############################################################################
###############################################################################
###############################################################################


@overload
def thomson_cross_section(
    thetas: float,
    normalize: bool = False
) -> float: ...


@overload
def thomson_cross_section(
    thetas: npt.NDArray[np.float_],
    normalize: bool = False
) -> npt.NDArray[np.float_]: ...


def thomson_cross_section(
    thetas: Union[float, npt.NDArray[np.float_]],
    normalize: bool = False
) -> Union[float, npt.NDArray[np.float_]]:
    """Calculate the differential thomson cross section dsigma/dOmega in
    square cm

    Parameters
    ----------
    thetas : float or `numpy.ndarray`
        scattering angles in deg

    normalize : bool
        if `True`, normalize output to 1

    Returns
    -------
    float or `numpy.ndarray`
        The cross section in cm**2 or normalized to 1
    """
    cross_section = 1.0 / 137.0**4 * (1.0 + np.cos(thetas)**2) / 2.0
    return (cross_section / np.amax(cross_section) if normalize
            else cross_section * 0.5292**2 * 10**-16)


@overload
def compton_photon_energy_out(
    Ein: float,
    cos_theta: float
) -> float: ...


@overload
def compton_photon_energy_out(
    Ein: float,
    cos_theta: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]: ...


def compton_photon_energy_out(
    Ein: float,
    cos_theta: Union[float, npt.NDArray[np.float_]]
) -> Union[float, npt.NDArray[np.float_]]:
    """Energy of an outgoing Compton photon

    Parameters
    ----------
    E1 : float
        energy of incoming photon in a.u.
    cos_theta : float or `np.ndarray`
        the cosine of the scattering angle theta

    Returns
    -------
    float or `np.ndarray`
        The energy of the scattered photon in a.u.
    """
    return Ein / (1.0 + Ein / 137.0**2 * (1.0 - cos_theta))


@overload
def klein_nishina_cross_section(
    Ein: float, cos_theta: float, normalize: bool = False
) -> float: ...


@overload
def klein_nishina_cross_section(
    Ein: float, cos_theta: npt.NDArray[np.float_], normalize: bool = False
) -> npt.NDArray[np.float_]: ...


def klein_nishina_cross_section(
    Ein: float,
    cos_theta: Union[float, npt.NDArray[np.float_]],
    normalize: bool = False
) -> Union[npt.NDArray[np.float_], float]:
    r"""Calculate Klein Nishina cross section

    Parameters
    ----------
    E1 : float
        energy of incoming photon in a.u.
    cos_theta : float or `np.ndarray`
        cosine of the scattering angles

    Returns
    -------
    float or `np.ndarray`
        The differential cross section $d\sigma/d\Omega$ in cm^2 or,
        if *normalize* is True, normalized to its maximum
    """
    ratio = compton_photon_energy_out(Ein, cos_theta) / Ein
    out = 0.5 / 137.0**4 * ratio**2 \
        * (1. / ratio + ratio - (1.0 - cos_theta**2))
    return out / np.amax(out) if normalize else out * 0.5292**2 * 10**-16


def scattering_angle_distr(
    N: int,
    k1_mag_au: float
) -> npt.NDArray[np.float_]:
    """
    Get distribution of N scattering angles following Klein Nishina cross
    section

    Parameters
    ----------
    N : int
        size of output distribution

    k1_mag_au : float
        Incoming photon momentum in a.u.

    Returns
    -------
    `numpy.ndarray`
        The scattering angles in rad
    """
    E1 = 137.0 * k1_mag_au
    succesful_throws = 0
    rtn = np.zeros(N)
    t0 = time.time()
    line0 = "Dice-throwing %d Compton photon scattering angles... " % (N)
    while succesful_throws < N:
        line = (
            "\r" + line0
            + "%.0lf percent done." % ((100.0 * succesful_throws / N))
        )
        print(line, end="")

        buffer = N - succesful_throws
        cos_theta_throw = 2 * np.random.random(buffer) - 1
        second_throw = np.random.random(buffer)
        kn = klein_nishina_cross_section(E1, cos_theta_throw, normalize=True)

        cos_theta_throw = np.ma.compressed(np.ma.masked_array(
            cos_theta_throw, mask=second_throw >= kn))

        theta = np.arccos(cos_theta_throw)
        rtn[succesful_throws:succesful_throws + theta.size] = theta

        succesful_throws += theta.size
    t1 = time.time()
    print(f"\nTotal runtime: {t1-t0}s")
    return rtn


def mom_final_distr_photon(
    N: int,
    k1_mag: float,
    theta_min: float = 0.0
) -> Vector:
    """Scatter randomly N photons with Klein Nishina cross section.

    Parameters
    ----------
    N : int
        The number of randomly scattering photons

    k1_mag : float
        Incoming photon vector interpreted as (k1_mag, 0, 0) in a.u.

    write_ascii : {None, string}, optional
        save momenta in an ascii file with filename <write_ascii>
        if None, do not write ascii (default None)

    theta_min: float, default = 0.0
        minimum scattering angle in rad

    Returns
    -------
    `atompy.vector.Vector`
        Photon momentum vectors
    """
    E1 = 137.0 * k1_mag

    succesful_throws = 0

    phi = 2 * np.pi * np.random.random(N)
    rtn = np.zeros((N, 3))
    max = klein_nishina_cross_section(E1, 1)

    t0 = time.time()
    line0 = "Dice-throwing %d Compton photon momenta... " % (N)
    while succesful_throws < N:
        line = (
            "\r" + line0
            + "%.0lf percent done." % ((100.0 * succesful_throws / N))
        )
        print(line, end="")

        buffer = N - succesful_throws

        a = -1.0
        b = np.cos(theta_min)
        cos_theta_throw = (b - a) * np.random.random(buffer) + a
        second_throw = np.random.random(buffer)
        kn = klein_nishina_cross_section(E1, cos_theta_throw) / max

        cos_theta_throw = np.ma.compressed(np.ma.masked_array(
            cos_theta_throw, mask=second_throw >= kn))

        mag = compton_photon_energy_out(E1, cos_theta_throw) / 137.0
        theta = np.arccos(cos_theta_throw)

        rtn[succesful_throws:succesful_throws + theta.size, 0] = (
            mag * cos_theta_throw)
        rtn[succesful_throws:succesful_throws + theta.size, 1] = (
            mag *
            np.sin(theta) *
            np.cos(phi[succesful_throws:succesful_throws + theta.size]))
        rtn[succesful_throws:succesful_throws + theta.size, 2] = (
            mag *
            np.sin(theta) *
            np.sin(phi[succesful_throws:succesful_throws + theta.size]))

        succesful_throws += len(cos_theta_throw)

    t1 = time.time()
    print(f"\nTotal runtime: {t1-t0:.2f}s")

    return Vector(rtn)


@overload
def mom_transfer_approx(
    kin_au: float,
    scattering_angles_rad: float,
) -> float: ...


@overload
def mom_transfer_approx(
    kin_au: float,
    scattering_angles_rad: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]: ...


def mom_transfer_approx(
    kin_au: float,
    scattering_angles_rad: Union[float, npt.NDArray[np.float_]],
) -> Union[float, npt.NDArray[np.float_]]:
    """
    Calculate momentum transfer assuming in and outgoing photon momentum
    is unchanged.

    Calculates `sqrt(2*kin_au**2 - 2*kin_au**2 * cos(scattering_angles_rad))`

    Parameters
    ----------
    kin_au : float
        Incoming photon momentum in a.u.

    scattering_angles_rad: float or `numpy.ndarray`
        Scattering angle(s) in rad

    Returns
    -------
    float or `numpy.ndarray`
        The momentum transfer(s)
    """
    Ein = kin_au * 137.0
    mom_difference = (Ein - compton_photon_energy_out(Ein, -1.0)) / Ein * 100.0
    print("I am calculating the momentum transfer assuming that the photon "
          f"momentum magnitude does not change. At {kin_au=}, the maximum "
          f"difference (backscattering) is {mom_difference:.2f} percent")
    cos_theta = np.cos(scattering_angles_rad)
    return np.sqrt(2 * kin_au**2 - 2 * kin_au**2 * cos_theta)


def mom_final_distr_elec(
    k1_mag_au: float,
    k2: Vector,
    kinit: Vector
) -> Vector:
    """Calculates kinit - k2 + (k1_mag,0,0)

    Parameters
    ----------
    k1_mag : float
        Magnitude of incoming photon in a.u.
    k2 : array-like, shape (,3)
        the scattered photon in a.u.
    kinit : array-like, shape (,3)
        initial momentum of particle in a.u.
    write_ascii : {None, string}, optional
        save momenta in an ascii file with filename <write_ascii>
        if None, do not write ascii (default None)

    Returns
    -------
    `atompy.vector.Vector`
        Momentum vector
    """
    Q = (k2 * -1.0) + Vector([k1_mag_au, 0, 0])
    return Q + kinit


@overload
def rho_p_microcanonical(
    pmag: float,
    E_bind: float,
    normalize: bool = True
) -> float: ...


@overload
def rho_p_microcanonical(
    pmag: npt.NDArray[np.float_],
    E_bind: float,
    normalize: bool = True
) -> npt.NDArray[np.float_]: ...


def rho_p_microcanonical(
    pmag: Union[float, npt.NDArray[np.float_]],
    E_bind: float,
    normalize: bool = True
) -> Union[float, npt.NDArray[np.float_]]:
    """
    Momentum distribution of one component in hydrogen-like system
    from Abrines Proc Phys. Soc 88 861 (1966)

    Parameters
    ----------
    pmag : float or `np.narray`
        momentum magnitude in a.u.

    E_bind : float
        binding energy in a.u.

    normalize: bool, default True
        if True, normalize resulting distribution to maximum

    Returns
    -------
    float or `np.ndarray`
    """
    p0 = np.sqrt(2.0 * E_bind)
    out = 8.0 * p0**5 / np.pi**2 / (pmag**2 + p0**2)**4
    return out / np.amax(out) if normalize else out


def mom_init_distr_elec(
    size: int,
    E_bind_au: float,
    scale: float = 3.0
) -> Vector:
    """
    Dice-throw three momentum components following a microcanonical
    distribution

    Parameters
    ----------
    size : int
        Number of throws

    E_bind : float
        Binding energy in a.u.

    scale: float
        scale factor for maximum momentum magnitude (max_mom = p0 * scale)

    Returns
    -------
    `atompy.vector.Vector`
        Momentum vectors of the distribution
    """
    t0 = time.time()
    line0 = ("Generating random microcanonical momentum distribution "
             f"(size {size}). ")

    succesful_throws = 0
    p = np.zeros((size, 3))
    while succesful_throws < size:
        line = (
            "\r" + line0
            + "%.0lf percent done." % (100.0 * succesful_throws / size)
        )
        print(line, end="")

        buffer = size - succesful_throws

        p0 = np.sqrt(2.0 * E_bind_au)

        pmag = np.random.uniform(0, p0 * scale, buffer)

        density = 8.0 * p0**5 / np.pi**2 / (pmag**2 + p0**2)**4 * pmag**2
        density /= np.max(density)

        second = np.random.random(buffer)

        pmag = np.ma.compressed(
            np.ma.masked_array(pmag, mask=second >= density))

        theta = np.arccos(2.0 * np.random.random(pmag.size) - 1.0)
        phi = 2.0 * np.pi * np.random.random(pmag.size)

        p[succesful_throws:succesful_throws + pmag.size, 0] = \
            pmag * np.sin(theta) * np.cos(phi)
        p[succesful_throws:succesful_throws + pmag.size, 1] = \
            pmag * np.sin(theta) * np.sin(phi)
        p[succesful_throws:succesful_throws + pmag.size, 2] = \
            pmag * np.cos(theta)

        succesful_throws += pmag.size

    t1 = time.time()

    print(f"\r{line0}Total runtime: {t1-t0:.2f}s")

    return Vector(p)


def mom_init_distr_elec_mol(
    distr_atomic: Vector,
    stretch_factor: float
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
    N = distr_atomic.vectors.shape[0]

    print("Creating molecular orbitals... ")
    t0 = time.time()

    # stretch along x and y, i.e., molecule is aligned along z
    distr_molecular = distr_atomic.vectors.copy()
    distr_molecular[:, :2] = distr_molecular[:, :2] * stretch_factor
    distr_molecular = Vector(distr_molecular)

    # rotate them randomly
    theta = np.arccos(2.0 * np.random.random(N) - 1)
    phi = 2.0 * np.pi * np.random.random(N)

    molecular_orientation = np.zeros(distr_molecular.vectors.shape)
    molecular_orientation[:, 0] = np.sin(theta) * np.cos(phi)
    molecular_orientation[:, 1] = np.sin(theta) * np.sin(phi)
    molecular_orientation[:, 2] = np.cos(theta)

    distr_molecular = \
        distr_molecular.rotated_around_y(theta).rotated_around_z(phi)

    t1 = time.time()

    print(f"Done. Total runtime: {t1-t0:.2f}s")

    return distr_molecular, Vector(molecular_orientation)


def stretch_Compton_electron_onto_sphere(
    pe_shifted: Vector,
    kout: Vector,
    kin_mag_au: float
) -> Vector:
    """
    The final electron momentum is not on a perfect sphere since the photon
    loses energy in the scattering process. Stretch the electron momenta
    to offset this energy loss

    Parameters
    ----------
    pe: `atompy.vector.Vector`
        shape (N, 3), the electron momenta after the scattering process

    kout: `atompy.vector.Vector`
        shape (N, 3), the corresponding distribution of photon momenta

    kin_mag_au: float
        momentum of the incoming photon. The incoming photon is assumed to be
        along x

    Returns
    -------
    `atompy.vector.Vector`
        The new electron momentum distribution

    Notes
    -----
    Calculates the energy loss of the photon depending on the scattering angle.
    Then stretches the electron momenta corresponding to that energy loss
    """
    stretch_factors = kin_mag_au / kout.mag
    out = pe_shifted.copy()
    out *= stretch_factors
    return out


def subtract_binding_energy(
    pin: Vector,
    Ebind: float
) -> Vector:
    """ Substracts binding energy from p, conserves direction of p

    Parameters
    ----------
    pin : `atompy.VectorCollection`
        ingoing momenta
    Ebind : float
        binding energy in a.u.

    Returns
    -------
    `atompy.VectorCollection`
        shortened momentum vector
    """

    radicands = 2 * (pin.mag**2 / 2.0 - Ebind)
    pmag_new = np.array([
        np.sqrt(radicand) if radicand > 0 else -1.0 for radicand in radicands])

    thetas = pin.theta
    phis = pin.phi
    pout = np.array([
        [p * np.sin(theta) * np.cos(phi),
         p * np.sin(theta) * np.sin(phi),
         p * np.cos(theta)]
        for p, theta, phi in zip(pmag_new, thetas, phis)
        if p > 0.0])

    return Vector(pout)


def calculate_Q_neglecting_mom_init(
    incoming_photon_momentum: Vector,
    final_elec_momentum: Vector,
) -> Vector:
    """
    Calculate momentum transfer with the approximation that the energy of 
    the outgoing photon equals the energy of the incoming photon.

    Parameters
    ----------
    incoming_photon_momentum: `vector.Vector`
        momentum of incoming photon in a.u., incoming photon is assumed to be
        along x-axis

    final_elec_momentum: `vector.Vector`
        final electron momenta in a.u.

    momentum_transfer: `vector.Vector`
        momentum transfer

    Returns
    -------
    `vector.Vector`
        shape (N, 3), the momentum transfers
    """
    alpha = np.arccos(final_elec_momentum.x / final_elec_momentum.mag)
    beta = np.pi - 2.0 * alpha

    k_out_approx_mag = incoming_photon_momentum.mag

    # law of cosine
    Q_approx_mag = np.sqrt(
        incoming_photon_momentum.mag**2 + k_out_approx_mag**2 -
        2.0 * incoming_photon_momentum.mag *
        k_out_approx_mag * np.cos(beta)
    )
    Q_approx = (final_elec_momentum /
                final_elec_momentum.mag) * Q_approx_mag
    return Q_approx

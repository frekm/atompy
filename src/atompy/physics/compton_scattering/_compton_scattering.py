from typing import overload, Union, Optional, Any
import numpy as np
import numpy.typing as npt
import time
from ..._vectors import VectorArray, Vector


def thomson_cross_section(
    thetas: npt.ArrayLike, normalize_to_max: bool = False
) -> Union[float, npt.NDArray[np.float64]]:
    r"""Calculate the differential thomson cross section.

    Parameters
    ----------
    thetas : float or `numpy.ndarray`
        scattering angles in deg

    normalize_to_max : bool
        if `True`, normalize output to 1

    Returns
    -------
    cross_section : float or ``numpy.ndarray``
        The cross section in cm\ :sup:`2` or with a maximum normalized to 1
        (depending on `normalize`).
    """
    cross_section = 1.0 / 137.0**4 * (1.0 + np.cos(thetas) ** 2) / 2.0
    return (
        cross_section / np.amax(cross_section)
        if normalize_to_max
        else cross_section * 0.5292**2 * 10**-16
    )


def compton_photon_energy_out(
    Ein: npt.ArrayLike, cos_theta: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Energy of an outgoing Compton photon

    Parameters
    ----------
    E1 : array_like
        energy of incoming photon in a.u.
    cos_theta : array_like
        the cosine of the scattering angle theta

    Returns
    -------
    energy : ndarray
        The energy of the scattered photon in a.u.
    """
    Ein = np.asarray(Ein)
    return Ein / (1.0 + Ein / 137.0**2 * (1.0 - np.asarray(cos_theta)))


def klein_nishina_cross_section(
    Ein: npt.ArrayLike, cos_theta: npt.ArrayLike, normalize: bool = False
) -> npt.NDArray[np.float64]:
    r"""Calculate Klein Nishina cross section

    Parameters
    ----------
    E1 : array_like
        energy of incoming photon in a.u.
    cos_theta : array_like
        cosine of the scattering angles

    Returns
    -------
    cross_section : ndarray
        The differential cross section $d\sigma/d\Omega$ in cm^2 or,
        if *normalize* is True, normalized to its maximum
    """
    Ein = np.asarray(Ein)
    ratio = compton_photon_energy_out(Ein, cos_theta) / Ein
    out: Any = (
        0.5
        / 137.0**4
        * ratio**2
        * (1.0 / ratio + ratio - (1.0 - np.asarray(cos_theta) ** 2))
    )
    return out / np.amax(out) if normalize else out * 0.5292**2 * 10**-16


def scattering_angle_distr(N: int, k1_mag_au: float) -> npt.NDArray[np.float64]:
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
    angles : `numpy.ndarray`
        The scattering angles in rad
    """
    E1 = 137.0 * k1_mag_au
    succesful_throws = 0
    rtn = np.zeros(N)
    t0 = time.time()
    line0 = "Dice-throwing %d Compton photon scattering angles... " % (N)
    while succesful_throws < N:
        line = "\r" + line0 + "%.0lf percent done." % ((100.0 * succesful_throws / N))
        print(line, end="")

        buffer = N - succesful_throws
        cos_theta_throw = 2 * np.random.random(buffer) - 1
        second_throw = np.random.random(buffer)
        kn = klein_nishina_cross_section(E1, cos_theta_throw, normalize=True)

        cos_theta_throw = np.ma.compressed(
            np.ma.masked_array(cos_theta_throw, mask=second_throw >= kn)
        )

        theta = np.arccos(cos_theta_throw)
        rtn[succesful_throws : succesful_throws + theta.size] = theta

        succesful_throws += theta.size
    t1 = time.time()
    print(f"\nTotal runtime: {t1-t0}s")
    return rtn


def mom_final_distr_photon_var(
    k1_mags_au: npt.NDArray[np.float64],
    theta_min: float = 0.0,
    rng_seed: Optional[float] = None,
) -> VectorArray:
    """
    Scatter photons randomly with Klein Nishina cross section.

    Photons have a momentum magnitude as given by ``k1_mags_au``.
    This is slow, if all photons have the same momentum magnitude, use
    :func:`.mom_final_distr_photon` instead.

    Parameters
    ----------
    k1_mag : ndarray
        Incoming photon vector interpreted as (k1_mag, 0, 0) in a.u.

    theta_min : float, default = 0.0
        minimum scattering angle in rad

    rng_seed : float, optional
        Seed value for random number generator

    Returns
    -------
    vectors : :class:`.Vector`
        Photon momentum vectors

    See also
    --------
    mom_final_distr_photon
    """
    incoming_energies = k1_mags_au * 137.0

    if rng_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rng_seed)  # type: ignore

    n_samples = 0
    output_costhetas = np.empty(incoming_energies.shape)

    kn_max = np.max(klein_nishina_cross_section(incoming_energies, 1))

    t0 = time.time()
    line0 = f"Randomly sampling {len(k1_mags_au)} Compton photon momenta... "

    while n_samples < len(k1_mags_au):
        line = (
            "\r"
            + line0
            + "%.0lf percent done." % ((100.0 * n_samples / len(k1_mags_au)))
        )
        print(line, end="")
        rand_costheta = rng.uniform(-1, np.cos(theta_min))
        rand_klein_nishina = rng.uniform(0, kn_max * 1.001)

        if rand_klein_nishina > klein_nishina_cross_section(
            incoming_energies[n_samples], rand_costheta
        ):
            continue

        output_costhetas[n_samples] = rand_costheta

        n_samples += 1

    output = np.empty((len(k1_mags_au), 3))
    output_phis = rng.uniform(0, 2 * np.pi, size=len(k1_mags_au))
    output_thetas = np.arccos(output_costhetas)
    magnitudes = compton_photon_energy_out(incoming_energies, output_costhetas) / 137.0

    output[:, 0] = magnitudes * output_costhetas
    output[:, 1] = magnitudes * np.sin(output_thetas) * np.cos(output_phis)
    output[:, 2] = magnitudes * np.sin(output_thetas) * np.sin(output_phis)

    t1 = time.time()
    print(f"\nTotal runtime: {t1-t0:.2f}s")

    return VectorArray(output)


def mom_final_distr_photon(
    N: int, k1_mag: float, theta_min: float = 0.0
) -> VectorArray:
    """Scatter randomly N photons with Klein Nishina cross section.

    Parameters
    ----------
    N : int
        The number of randomly scattering photons

    k1_mag : float
        Incoming photon vector interpreted as (k1_mag, 0, 0) in a.u.

    theta_min : float, default = 0.0
        minimum scattering angle in rad

    Returns
    -------
    vectors : :class:`.Vector`
        Photon momentum vectors
    """
    phot_ener_in = 137.0 * k1_mag

    succesful_throws = 0

    phi = 2 * np.pi * np.random.random(N)
    rtn = np.zeros((N, 3))
    max = klein_nishina_cross_section(phot_ener_in, 1)

    t0 = time.time()
    line0 = "Dice-throwing %d Compton photon momenta... " % (N)
    while succesful_throws < N:
        line = "\r" + line0 + "%.0lf percent done." % ((100.0 * succesful_throws / N))
        print(line, end="")

        buffer = N - succesful_throws

        a = -1.0
        b = np.cos(theta_min)
        cos_theta_throw = (b - a) * np.random.random(buffer) + a
        second_throw = np.random.random(buffer)
        kn = klein_nishina_cross_section(phot_ener_in, cos_theta_throw) / max

        cos_theta_throw = np.ma.compressed(
            np.ma.masked_array(cos_theta_throw, mask=second_throw >= kn)
        )

        mag = compton_photon_energy_out(phot_ener_in, cos_theta_throw) / 137.0
        theta = np.arccos(cos_theta_throw)

        rtn[succesful_throws : succesful_throws + theta.size, 0] = mag * cos_theta_throw
        rtn[succesful_throws : succesful_throws + theta.size, 1] = (
            mag
            * np.sin(theta)
            * np.cos(phi[succesful_throws : succesful_throws + theta.size])
        )
        rtn[succesful_throws : succesful_throws + theta.size, 2] = (
            mag
            * np.sin(theta)
            * np.sin(phi[succesful_throws : succesful_throws + theta.size])
        )

        succesful_throws += len(cos_theta_throw)

    t1 = time.time()
    print(f"\nTotal runtime: {t1-t0:.2f}s")

    return VectorArray(rtn)


@overload
def mom_transfer_approx(
    kin_au: float,
    scattering_angles_rad: float,
) -> float: ...


@overload
def mom_transfer_approx(
    kin_au: float,
    scattering_angles_rad: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...


def mom_transfer_approx(
    kin_au: float,
    scattering_angles_rad: Union[float, npt.NDArray[np.float64]],
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Calculate momentum transfer assuming in and outgoing photon momentum
    is unchanged.

    Calculates :math:`\sqrt{2\times\texttt{kin_au}^2 - 2\times
    \texttt{kin_au}^2  \times \cos(\texttt{scattering_angles_rad})}`.

    Parameters
    ----------
    kin_au : float
        Incoming photon momentum in a.u.

    scattering_angles_rad : float or `numpy.ndarray`
        Scattering angle(s) in rad

    Returns
    -------
    float or `numpy.ndarray`
        The momentum transfer(s)
    """
    Ein = kin_au * 137.0
    mom_difference = (Ein - compton_photon_energy_out(Ein, -1.0)) / Ein * 100.0
    print(
        "I am calculating the momentum transfer assuming that the photon "
        f"momentum magnitude does not change. At {kin_au=}, the maximum "
        f"difference (backscattering) is {mom_difference:.2f} percent"
    )
    cos_theta = np.cos(scattering_angles_rad)
    return np.sqrt(2 * kin_au**2 - 2 * kin_au**2 * cos_theta)


def mom_final_distr_elec(
    k1_mag_au: float, k2: VectorArray, kinit: VectorArray
) -> VectorArray:
    """Calculates kinit - k2 + (k1_mag,0,0)

    Parameters
    ----------
    k1_mag : float
        Magnitude of incoming photon in a.u.
    k2 : :class:`.VectorArray`
        the scattered photon in a.u.
    kinit : :class:`.VectorArray`
        initial momentum of particle in a.u.

    Returns
    -------
    `atompy.vector.Vector`
        Momentum vector
    """
    Q = (k2 * -1.0) + VectorArray([k1_mag_au, 0, 0])
    return Q + kinit


def stretch_Compton_electron_onto_sphere(
    pe_shifted: VectorArray, kout: VectorArray, kin_mag_au: float
) -> VectorArray:
    """
    The final electron momentum is not on a perfect sphere since the photon
    loses energy in the scattering process. Stretch the electron momenta
    to offset this energy loss

    Parameters
    ----------
    pe : :class:`.VectorArray`
        shape (N, 3), the electron momenta after the scattering process

    kout : :class:`.VectorArray`
        shape (N, 3), the corresponding distribution of photon momenta

    kin_mag_au : float
        momentum of the incoming photon. The incoming photon is assumed to be
        along x

    Returns
    -------
    vectors : :class:`.VectorArray`
        The new electron momentum distribution

    Notes
    -----
    Calculates the energy loss of the photon depending on the scattering angle.
    Then stretches the electron momenta corresponding to that energy loss
    """
    stretch_factors = kin_mag_au / kout.mag()
    out = pe_shifted.copy()
    out *= stretch_factors
    return out


def calculate_Q_neglecting_mom_init(
    incoming_photon_momentum: VectorArray,
    final_elec_momentum: VectorArray,
) -> VectorArray:
    """
    Calculate momentum transfer with the approximation that the energy of
    the outgoing photon equals the energy of the incoming photon.

    Parameters
    ----------
    incoming_photon_momentum : :class:`.VectorArray`
        momentum of incoming photon in a.u., incoming photon is assumed to be
        along x-axis

    final_elec_momentum : :class:`.VectorArray`
        final electron momenta in a.u.

    momentum_transfer : :class:`.VectorArray`
        momentum transfer

    Returns
    -------
    vectors : :class:`.Vector`
        shape (N, 3), the momentum transfers
    """
    alpha = np.arccos(final_elec_momentum.x / final_elec_momentum.mag())
    beta = np.pi - 2.0 * alpha

    k_out_approx_mag = incoming_photon_momentum.mag()

    # law of cosine
    Q_approx_mag = np.sqrt(
        incoming_photon_momentum.mag() ** 2
        + k_out_approx_mag**2
        - 2.0 * incoming_photon_momentum.mag() * k_out_approx_mag * np.cos(beta)
    )
    Q_approx = (final_elec_momentum / final_elec_momentum.mag()) * Q_approx_mag
    return Q_approx

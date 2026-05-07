import pickle
import os

import numpy as np
import joblib
import tqdm

from atompy.physics.particles import Molecule
from atompy import _vectors as vec
from atompy.physics import constants


def calc_coulomb_force(mol: Molecule, idx_probe: int) -> vec.Vector:
    """
    Calculate Coulomb force acting on atom `idx_probe` of `mol`.

    Parameters
    ----------
    mol : :class:`.Molecule`
        The molecule.

        It is assumed that all attributes of the molecule (positions, speeds, masses)
        are given in a.u.

    idx_probe : int
        The index of :attr:`.~Molecule.atoms` on which the calculated force acts.

    Returns
    -------
    :class:`.Vector`
        The vectorial force acting on atom `idx_probe` in a.u..
    """
    force = vec.Vector(0.0, 0.0, 0.0)

    for i in range(mol.size):
        if i == idx_probe:
            continue
        direction = mol.atoms[idx_probe].pos - mol.atoms[i].pos
        distance = direction.mag()
        force += direction.norm().scale(
            mol.atoms[idx_probe].charge * mol.atoms[i].charge / distance**2
        )

    return force


def _coulomb_explode_step(mol: Molecule, dt: float) -> Molecule:
    """
    Advance time of the Coulomb explosion by `dt`.

    Updates the positions and speeds of `mol`.

    `dt` in a.u.
    """
    updated_mol = mol.copy()
    for i in range(mol.size):
        atom = mol.atoms[i]
        force = calc_coulomb_force(mol, i)
        accel = force.scale(1.0 / atom.mass)
        new_pos = 0.5 * accel * dt**2 + atom.speed * dt + atom.pos
        new_speed = accel * dt + mol.atoms[i].speed
        updated_mol.atoms[i].pos = new_pos
        updated_mol.atoms[i].speed = new_speed
    return updated_mol


def coulomb_explode(
    mol: Molecule, time_end_fs: float = 5000.0, time_stepsize_fs: float = 1.0
) -> Molecule:
    """
    Coulomb explode a molecule.

    .. attention::

        This function is slow as it uses native Python code for its core computations.

    Parameters
    ----------
    mol : :class:`.Molecule`
        The initial state of the molecule.

        It is assumed that all attributes of the molecule (positions, speeds, masses)
        are given in a.u.

    time_end_fs : float, default 5000 fs
        The time up to which the Coulomb explosion is simulated (in fs).

    time_step_fs : float, default 1 fs
        The time steps in which the Coulomb explosion is simulated (in fs).

    Returns
    -------
    :class:`.Molecule`
        A ``Molecule`` instance describing the state of the initial
        molecule after *time_end_fs*.
    """
    t1 = time_end_fs * constants.AU_PER_FS
    dt = time_stepsize_fs * constants.AU_PER_FS
    steps = int(t1 // dt)

    final_mol = mol.copy()

    for _ in range(steps):
        final_mol = _coulomb_explode_step(final_mol, dt)

    return final_mol


def coulomb_explode_batch(
    molecules: np.ndarray[tuple[int], np.dtype[np.object_]],
    time_end_fs: float,
    time_stepsize_fs: float,
    pickle_fname: str | os.PathLike | None = None,
    n_jobs: int = -2,
) -> np.ndarray[tuple[int], np.dtype[np.object_]]:
    """
    Coulomb explode a batch of molecules.

    .. attention::

        This function is slow. It uses native Python code and inefficient memory layout
        for its core computations.

    Parameters
    ----------
    molecules : ndarray of :class:`.Molecule`
        The molecules.

        It is assumed that all attributes of the molecules (positions, speeds, masses)
        are given in a.u.

    time_end_fs : float, default 5000 fs
        The time up to which the Coulomb explosion is simulated (in fs).

    time_step_fs : float, default 1 fs
        The time steps in which the Coulomb explosion is simulated (in fs).

    pickle_fname : str | PathLike, optional
        If provided, pickle output and save it.

    n_jobs : int, default -2
        The number of jobs started.

        The default value creates *number-of-CPUs* minus 1 jobs.

        See *n_jobs* descriptions of
        `joblibs.Parallel <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`__
        for more information.

    Returns
    -------
    :class:`.Molecule`
        A :class:`.!Molecule` instance describing the state of the initial
        molecule after *time_end_fs*.
    """
    final_molecules = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(coulomb_explode)(mol, time_end_fs, time_stepsize_fs)
        for mol in tqdm.tqdm(molecules, desc="Processing Molecules")
    )
    final_molecules = np.array(final_molecules)

    if pickle_fname is not None:
        print(f"pickling data to {pickle_fname} ...", end="")
        with open(pickle_fname, "wb") as file:
            pickle.dump(final_molecules, file)
        print(" done")

    return final_molecules

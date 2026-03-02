import time
import pickle
import os

import numpy as np

from atompy.physics.particles import Molecule
from atompy import _vectors as vec
from atompy.physics import constants


def calc_coulomb_force(mol: Molecule, idx_probe: int) -> vec.Vector:
    """
    Calculate Coulomb force acting on atom `idx_probe` of `mol`.
    """
    force = vec.Vector(0.0, 0.0, 0.0)

    for i in range(mol.size):
        if i == idx_probe:
            continue
        direction = mol.positions()[idx_probe] - mol.positions()[i]
        distance = direction.mag()
        force += direction.norm().scale(
            mol.charges()[idx_probe] * mol.charges()[i] / distance**2
        )

    return force


def coulomb_explode_step(mol: Molecule, dt: float) -> Molecule:
    """
    Advance time of the Coulomb explosion by `dt`.

    Updates the positions and speeds of `mol`.

    `dt` in a.u.
    """
    updated_mol = mol.copy()
    for i in range(mol.size):
        force = calc_coulomb_force(mol, i)
        acceleration = force.scale(1.0 / mol.masses()[i])
        updated_mol.positions()[i] = (
            0.5 * acceleration * dt**2 + mol.speeds()[i] * dt + mol.positions()[i]
        )
        updated_mol.speeds()[i] = acceleration * dt + mol.speeds()[i]
    return updated_mol


def coulomb_explode(
    mol: Molecule, time_end_fs: float, time_stepsize_fs: float
) -> Molecule:
    """
    Coulomb explode a molecule with an initial state described by `mol`.
    """
    t1 = time_end_fs * constants.AU_PER_FS
    dt = time_stepsize_fs * constants.AU_PER_FS
    steps = int(t1 // dt)

    final_mol = mol.copy()

    for _ in range(steps):
        final_mol = coulomb_explode_step(final_mol, dt)

    return final_mol


def coulomb_explode_batch(
    molecules: np.ndarray[tuple[int], np.dtype[np.object_]],
    time_end_fs: float,
    time_stepsize_fs: float,
    pickle_fname: str | os.PathLike | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.object_]]:
    print(f"Simulating Coulomb explosion of {len(molecules)} molecules ...")
    t0 = time.time()

    final_molecules = np.empty_like(molecules)

    for i, mol in enumerate(molecules):
        final_molecules[i] = coulomb_explode(mol, time_end_fs, time_stepsize_fs)

    print(f"Finished coulomb exploding. Elapsed time: {time.time() - t0:.2f}")

    if pickle_fname is not None:
        print(f"pickling data to {pickle_fname} ...", end="")
        with open(pickle_fname, "wb") as file:
            pickle.dump(final_molecules, file)
        print(" done")

    return final_molecules

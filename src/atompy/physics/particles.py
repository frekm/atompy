from collections import UserList
from typing import Self, Union
import copy

import numpy as np

from atompy import _vectors as vec


class Particle:
    def __init__(
        self,
        mom: vec.VectorLike,
        pos: vec.VectorLike,
        mass: float,
        charge: float,
        name: str,
    ) -> None:
        self._mass = mass
        self._charge = charge
        self._mom = vec.asvector(mom)
        self._pos = vec.asvector(pos)
        self._name = name

    @property
    def pos(self) -> vec.Vector:
        """Position of the particle."""
        return self._pos

    @pos.setter
    def pos(self, pos) -> None:
        self._pos = pos

    @property
    def mass(self) -> float:
        """Mass of the particle."""
        return self._mass

    @mass.setter
    def mass(self, mass) -> None:
        self._mass = mass

    @property
    def charge(self) -> float:
        """Charge of the particle."""
        return self._charge

    @charge.setter
    def charge(self, charge) -> None:
        self._charge = charge

    @property
    def mom(self) -> vec.Vector:
        """Momentum of the particle."""
        return self._mom

    @mom.setter
    def mom(self, vec: vec.Vector) -> None:
        self.mom = vec

    @property
    def speed(self) -> vec.Vector:
        """Speed (derived from its momentum) of the particle."""
        return self.mom / self.mass

    @speed.setter
    def speed(self, speed: vec.Vector) -> None:
        self._mom = speed * self.mass

    @property
    def energy(self) -> float:
        """Energy (derived from its momentum) of the particle."""
        return self.mom.mag() ** 2 / 2.0 / self.mass

    @property
    def name(self) -> str:
        """Descriptor of the particle."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name


class Electron(Particle):
    """
    Class describing electrons (as classical particles).

    .. note::

        If you want to use anything but atomic units (a.u.), be sure to pass
        the (optional) values of *charge* and *mass* (which assume a.u.).

    Parameters
    ----------
    mom : :class:`.Vector`
        Momentum of the electron.

    pos : :class:`.Vector`
        Position of the electron.

    mass : float, default 1.0 a.u.
        Mass of the electron.

    charge : float, default -1.0 a.u.
        Charge of the electron.

    name : str, default ``"e"``
        Name of the electron.

    Attributes
    ----------
    mom : :class:`.Vector`

    pos : :class:`.Vector`

    mass : float

    charge : float

    speed : :class:`.Vector`

    energy : float

    name : str
    """

    def __init__(
        self,
        mom: vec.Vector,
        pos: vec.Vector,
        mass: float = 1.0,
        charge: float = -1.0,
        name="e",
    ) -> None:
        super().__init__(mom, pos, mass, charge, name)


class Atom(Particle):
    """
    Class describing atoms (as classical particles).

    Parameters
    ----------
    mom : :class:`.Vector`
        Momentum of the atom.

    pos : :class:`.Vector`
        Position of the atom.

    mass : float
        Mass of the atom

    charge : float
        Charge of the atom.

    name : str
        Name of the atom.

    Attributes
    ----------
    mom : :class:`.Vector`

    pos : :class:`.Vector`

    mass : float

    charge : float

    speed : :class:`.Vector`

    energy : float

    name : str
    """

    pass


class ParticleList(UserList[Atom]):
    def momenta(self) -> vec.VectorArray:
        """
        Retrieve all momenta of particles in the list.

        Returns
        -------
        momenta : :class:`.VectorArray`
            *momenta[i]* corresponds to particle *i* in the list.
        """
        return vec.VectorArray([d.mom for d in self])

    def positions(self) -> vec.VectorArray:
        """
        Retrieve all positions of particles in the list.

        Returns
        -------
        positions : :class:`.VectorArray`
            *positions[i]* corresponds to particle *i* in the list.
        """
        return vec.VectorArray([d.pos for d in self])

    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Retrieve all masses of particles in the list.

        Returns
        -------
        masses : ndarray
            *masses[i]* corresponds to particle *i* in the list.
        """
        return np.array([d.mass for d in self])

    def charges(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Retrieve all charges of particles in the list.

        Returns
        -------
        charges : ndarray
            *charges[i]* corresponds to particle *i* in the list.
        """
        return np.array([d.charge for d in self])

    def speeds(self) -> vec.VectorArray:
        """
        Retrieve all speeds of particles in the list.

        Returns
        -------
        speeds : :class:`.VectorArray`
            *speeds[i]* corresponds to particle *i* in the list.
        """
        return vec.VectorArray([d.speed for d in self])

    def names(self) -> list[str]:
        """
        Retrieve all names of particles in the list.

        Returns
        -------
        names : ndarray
            *names[i]* corresponds to particle *i* in the list.
        """
        return [self.data[i].name for i in range(len(self))]


class ElectronList(ParticleList):
    """Wrapper for Python lists only containing :class:`.Electron`"""

    pass


class AtomList(ParticleList):
    """Wrapper for Python lists only containing :class:`.Atom`"""

    pass


AtomListLike = Union[tuple[Atom, ...], list[Atom], AtomList]
ElectronListLike = Union[tuple[Electron, ...], list[Electron], ElectronList]


class Molecule:
    def __init__(self, atoms: AtomListLike) -> None:
        self._atoms = AtomList(atoms)

    @property
    def size(self) -> int:
        """Number of atoms within the molecule."""
        return len(self._atoms)

    @property
    def atoms(self) -> AtomList:
        """Atoms within the molecule."""
        return self._atoms

    def momenta(self) -> vec.VectorArray:
        """
        Retrieve all momenta of atoms in the molecule.

        Returns
        -------
        momenta : :class:`.VectorArray`
            *momenta[i]* corresponds to particle *i* in the molecule.
        """
        return self.atoms.momenta()

    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Retrieve all masses of atoms in the molecule.

        Returns
        -------
        masses : ndarray
            *masses[i]* corresponds to particle *i* in the molecule.
        """
        return self.atoms.masses()

    def speeds(self) -> vec.VectorArray:
        """
        Retrieve all speeds of atoms in the molecule.

        Returns
        -------
        speeds : :class:`.VectorArray`
            *speeds[i]* corresponds to particle *i* in the molecule.
        """
        return self.atoms.speeds()

    def positions(self) -> vec.VectorArray:
        """
        Retrieve all positions of atoms in the molecule.

        Returns
        -------
        positions : :class:`.VectorArray`
            *positions[i]* corresponds to particle *i* in the molecule.
        """
        return self.atoms.positions()

    def charges(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Retrieve all charges of atoms in the molecule.

        Returns
        -------
        charges : ndarray
            *charges[i]* corresponds to particle *i* in the molecule.
        """
        return self.atoms.charges()

    def names(self) -> list[str]:
        """
        Retrieve all names of atoms in the molecule.

        Returns
        -------
        names : ndarray
            *names[i]* corresponds to atom *i* in the molecule.
        """
        return self.atoms.names()

    def copy(self) -> Self:
        """Retrieve a copy of the molecule."""
        return type(self)(copy.deepcopy(self._atoms))

    def mom_sum(self) -> vec.Vector:
        """
        Sum momentum of all atoms in the molecule.

        Returns
        -------
        momentum : :class:`.Vector`
        """
        mom_sum = vec.Vector(0, 0, 0)
        mom_sum.x = np.sum(self.momenta().x)
        mom_sum.y = np.sum(self.momenta().y)
        mom_sum.z = np.sum(self.momenta().z)
        return mom_sum

    def kinetic_energy_release(self) -> float:
        """
        Kinetic energy of all atoms of the molecule in rest frame.

        Returns
        -------
        energy : float
            The energy in the frame where :meth:`~.Molecule.mom_sum` is zero.
        """
        momenta = self.momenta() - self.mom_sum()
        kinetic_energies = momenta.mag() ** 2 / self.masses() / 2.0
        return float(np.sum(kinetic_energies))

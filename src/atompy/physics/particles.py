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
        return self._pos

    @pos.setter
    def pos(self, pos) -> None:
        self._pos = pos

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, mass) -> None:
        self._mass = mass

    @property
    def charge(self) -> float:
        return self._charge

    @charge.setter
    def charge(self, charge) -> None:
        self._charge = charge

    @property
    def mom(self) -> vec.Vector:
        return self._mom

    @mom.setter
    def mom(self, vec: vec.Vector) -> None:
        self.mom = vec

    @property
    def speed(self) -> vec.Vector:
        return self.mom / self.mass

    @speed.setter
    def speed(self, speed: vec.Vector) -> None:
        self._mom = speed * self.mass

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name


class Electron(Particle):
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
    pass


class ParticleList(UserList[Atom]):
    def momenta(self) -> vec.VectorArray:
        momenta = np.empty(len(self), dtype=object)
        for i in range(len(self)):
            momenta[i] = self.data[i].mom
        return vec.VectorArray(momenta)

    def positions(self) -> vec.VectorArray:
        positions = np.empty(len(self), dtype=object)
        for i in range(len(self)):
            positions[i] = self.data[i].pos
        return vec.VectorArray(positions)

    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        masses = np.empty(len(self), dtype=np.float64)
        for i in range(len(self)):
            masses[i] = self.data[i].mass
        return masses

    def charges(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        charges = np.empty(len(self), dtype=np.float64)
        for i in range(len(self)):
            charges[i] = self.data[i].charge
        return charges

    def speeds(self) -> vec.VectorArray:
        speeds = np.empty(len(self), dtype=object)
        for i in range(len(self)):
            speeds[i] = self.data[i].speed
        return vec.VectorArray(speeds)

    def names(self) -> list[str]:
        names = []
        for i in range(len(self)):
            names.append(self.data[i].name)
        return names


class ElectronList(ParticleList):
    pass


class AtomList(ParticleList):
    pass


AtomListLike = Union[tuple[Atom, ...], list[Atom], AtomList]
ElectronListLike = Union[tuple[Electron, ...], list[Electron], ElectronList]


class Molecule:
    def __init__(self, atoms: AtomListLike) -> None:
        self._atoms = AtomList(atoms)

    @property
    def size(self) -> int:
        return len(self._atoms)

    @property
    def atoms(self) -> AtomList:
        return self._atoms

    def momenta(self) -> vec.VectorArray:
        return self.atoms.momenta()

    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self.atoms.masses()

    def speeds(self) -> vec.VectorArray:
        return self.atoms.speeds()

    def positions(self) -> vec.VectorArray:
        return self.atoms.positions()

    def charges(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self.atoms.charges()

    def names(self) -> list[str]:
        return self.atoms.names()

    def copy(self) -> Self:
        return type(self)(copy.deepcopy(self._atoms))

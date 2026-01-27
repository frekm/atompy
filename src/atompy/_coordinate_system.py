from typing import overload, Self
from . import _vectors as vec
import numpy as np

# TODO docstrings


class CoordinateSystem:
    r"""
    Class representing a single coordinate system.

    .. tip::

        If you want to store an array of coordinate systems, consider
        :class:`.CoordinateSystemArray`.

    Parameters
    ----------

    vector_1 : VectorLike
        First vector $\vec{v}_1$ defining the coordinate system.

    vector_2 : VectorLike
        Second vector $\vec{v}_2$ defining the coordinate system.
        Should not be parallel to `vector_1`.

    Notes
    -----
    The three basis vectors $\hat{x}, \hat{y}, \hat{z}$ will be unit vectors along
    the directions given by

    .. math::

        \vec{z} &= \vec{v_1} \\
        \vec{y} &= \vec{v_z} \times \vec{v_2} \\
        \vec{x} &= \vec{v_y} \times \vec{z}

    Attributes
    ----------
    x_axis, y_axis, z_axis : :class:`.Vector`
        x, y, z basis vectors of the coordinate system.
    """

    def __init__(self, vector_1: vec.VectorLike, vector_2: vec.VectorLike):
        vec1_ = vec.asvector(vector_1)
        vec2_ = vec.asvector(vector_2)
        self._z_axis = vec1_.norm(copy=False)
        self._y_axis = vec1_.cross(vec2_).norm(copy=False)
        self._x_axis = self._y_axis.cross(vec1_).norm(copy=False)

    @property
    def x_axis(self) -> vec.Vector:
        return self._x_axis

    @property
    def y_axis(self) -> vec.Vector:
        return self._y_axis

    @property
    def z_axis(self) -> vec.Vector:
        return self._z_axis

    def project_vector(self, vector: vec.VectorLike) -> vec.Vector:
        vec_ = vec.asvector(vector)
        out = vec.Vector(
            vec_.dot(self.x_axis), vec_.dot(self.y_axis), vec_.dot(self.z_axis)
        )
        return out


class CoordinateSystemArray:
    def __init__(self, vectors_1: vec.VectorArrayLike, vectors_2: vec.VectorArrayLike):
        vec1_ = vec.asvectorarray(vectors_1)
        vec2_ = vec.asvectorarray(vectors_2)
        self._z_axis = vec1_.norm(copy=False)
        self._y_axis = vec1_.cross(vec2_).norm(copy=False)
        self._x_axis = self._y_axis.cross(vec1_).norm(copy=False)

    @property
    def x_axis(self) -> vec.VectorArray:
        return self._x_axis

    @property
    def y_axis(self) -> vec.VectorArray:
        return self._y_axis

    @property
    def z_axis(self) -> vec.VectorArray:
        return self._z_axis

    def project_vector(self, vector: vec.VectorArrayLike) -> vec.VectorArray:
        vec_ = vec.asvectorarray(vector)
        out = vec.VectorArray(np.empty_like(vec_.asarray()))

        out.x = vec_.dot(self.x_axis)
        out.y = vec_.dot(self.y_axis)
        out.z = vec_.dot(self.z_axis)

        return out

    @overload
    def __getitem__(self, i: int) -> CoordinateSystem: ...

    @overload
    def __getitem__(self, i: slice) -> Self: ...

    def __getitem__(self, i: int | slice) -> CoordinateSystem | Self:
        if isinstance(i, int):
            return CoordinateSystem(self.z_axis[i], self.x_axis[i])
        elif isinstance(i, slice):
            return type(self)(self.z_axis[i], self.x_axis[i])
        else:
            raise TypeError(f"i must be int or slice, but is {type(i)}")

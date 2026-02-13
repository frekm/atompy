from typing import overload, Self
from . import _vectors as vec
import numpy as np


class CoordinateSystem:
    r"""
    Class representing a single coordinate system.

    .. tip::

        If you want to store an array of coordinate systems, consider
        :class:`.CoordinateSystemArray`.

    Parameters
    ----------

    vector_1 : VectorLike
        First vector :math:`\vec{v}_1` defining the coordinate system.

    vector_2 : VectorLike
        Second vector :math:`\vec{v}_2` defining the coordinate system.
        Should not be parallel to `vector_1`.

    Notes
    -----
    The three basis vectors :math:`\hat{x}, \hat{y}, \hat{z}` will be unit vectors along
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
        self._z_axis = vec1_.norm()
        self._y_axis = vec1_.cross(vec2_).norm()
        self._x_axis = self._y_axis.cross(vec1_).norm()

    @property
    def x_axis(self) -> vec.Vector:
        """Unit vector defining the x axis"""
        return self._x_axis

    @property
    def y_axis(self) -> vec.Vector:
        """Unit vector defining the y axis"""
        return self._y_axis

    @property
    def z_axis(self) -> vec.Vector:
        """Unit vector defining the z axis"""
        return self._z_axis

    def project_vector(self, vector: vec.VectorLike) -> vec.Vector:
        """
        Project `vector` on the coordinate system.

        Parameters
        ----------
        vector : VectorLike

        Returns
        -------
        :class:`.Vector`
            The projected vector.

        Examples
        --------

        ::

            >>> v1 = ap.Vector(1, 1, 0)
            >>> v2 = ap.Vector(1, 0, 1)
            >>> c = ap.CoordinateSystem(v1, v2)
            >>> c.project_vector(v1)
            Vector(0.0, 0.0, 1.414213562373095)
        """
        vec_ = vec.asvector(vector)
        out = vec.Vector(
            vec_.dot(self.x_axis), vec_.dot(self.y_axis), vec_.dot(self.z_axis)
        )
        return out


class CoordinateSystemArray:
    r"""
    Class representing a collection of coordinate systems.

    .. tip::

        If you want to store a single coordinate systems, consider
        :class:`.CoordinateSystemArray`.

    Parameters
    ----------

    vectors_1 : VectorArrayLike
        First set of vectors :math:`\vec{v}_1` defining the coordinate system.

    vectors_2 : VectorArrayLike
        Second set of vectors :math:`\vec{v}_2` defining the coordinate system.
        Should not be parallel to `vectors_1`.

    Notes
    -----
    The three basis vectors :math:`\hat{x}, \hat{y}, \hat{z}` will be unit vectors
    along the directions given by, respectively

    .. math::

        \vec{z} &= \vec{v_1} \\
        \vec{y} &= \vec{v_z} \times \vec{v_2} \\
        \vec{x} &= \vec{v_y} \times \vec{z}

    Attributes
    ----------
    x_axis, y_axis, z_axis : :class:`.Vector`
        x, y, z basis vectors of the coordinate system.
    """

    def __init__(self, vectors_1: vec.VectorArrayLike, vectors_2: vec.VectorArrayLike):
        vec1_ = vec.asvectorarray(vectors_1)
        vec2_ = vec.asvectorarray(vectors_2)
        self._z_axis = vec1_.norm(copy=False)
        self._y_axis = vec1_.cross(vec2_).norm(copy=False)
        self._x_axis = self._y_axis.cross(vec1_).norm(copy=False)

    @property
    def x_axis(self) -> vec.VectorArray:
        """Unit vectors defining the x axis"""
        return self._x_axis

    @property
    def y_axis(self) -> vec.VectorArray:
        """Unit vectors defining the y axis"""
        return self._y_axis

    @property
    def z_axis(self) -> vec.VectorArray:
        """Unit vectors defining the z axis"""
        return self._z_axis

    def project_vectors(self, vectors: vec.VectorArrayLike) -> vec.VectorArray:
        """
        Project `vectors` on the coordinate system.

        Parameters
        ----------
        vectors : VectorArrayLike
            If a single vector is passed, project it into each coordinate system.

        Returns
        -------
        :class:`.Vector`
            The projected vectors.

        Examples
        --------
        ::

            >>> v1 = ap.Vector(1, 1, 0)
            >>> v2 = ap.Vector(1, 0, 1)
            >>> c = ap.CoordinateSystemArray((v1, v2), (v2, v1))
            >>> c.project_vectors(v1)
            VectorArray([[0.         0.         1.41421356]
                         [1.22474487 0.         0.70710678]])
            >>> c.project_vectors(v2)
            VectorArray([[1.22474487 0.         0.70710678]
                         [0.         0.         1.41421356]])
            >>> c.project_vectors((v1, v2))
            VectorArray([[0.         0.         1.41421356]
                         [0.         0.         1.41421356]])
        """
        vec_ = vec.asvectorarray(vectors)

        if len(vec_) == 1:
            vec_ = vec.VectorArray(np.repeat(vec_.asarray(), len(self), axis=0))

        out = vec.VectorArray(np.empty_like(vec_.asarray()))

        out.x = vec_.dot(self.x_axis)
        out.y = vec_.dot(self.y_axis)
        out.z = vec_.dot(self.z_axis)

        return out

    def __len__(self) -> int:
        return len(self.x_axis)

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

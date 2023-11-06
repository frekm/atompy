import numpy as np
import numpy.typing as npt
from typing import Any, Optional

###############################################################################
###############################################################################
###############################################################################
# Vector
###############################################################################
###############################################################################
###############################################################################


class _VectorIterator:
    def __init__(self, vectors: "Vector") -> None:
        self.vectors = vectors
        self.index = 0

    def __iter__(self) -> "_VectorIterator":
        return self

    def __next__(self) -> "Vector":
        if self.index == len(self.vectors):
            raise StopIteration
        self.index += 1
        return Vector(self.vectors[self.index - 1])


class Vector:
    """
    Wrapper class for numpy arrays that represent vectors

    Parameters
    ----------
    vectors: ArrayLike, shape (N, 3) or (3,)
        a list of vectors [vec1, vec2, vec3, ...]

    Examples
    --------
    ::

        >>> vec  = Vector([1, 2, 3])
        >>> vecs = Vector([[1, 2, 3], [4, 5, 6]])
        >>> vec.x
        1.0
        >>> vecs.x
        [1. 4.]
    """

    def __init__(
        self,
        vectors: npt.ArrayLike
    ) -> None:
        vectors_ = np.array(vectors)
        if vectors_.ndim == 1:
            self._data = np.array([vectors]).astype(np.float64)
        else:
            self._data = np.array(vectors).astype(np.float64)

        if self._data.ndim != 2:
            raise ValueError("ndim")

        if self._data.shape[1] != 3:
            raise ValueError("shape")

    def __getitem__(self, key) -> npt.NDArray[Any]:
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    def __add__(
        self,
        other: "Vector"
    ) -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self._data + other._data)

    def __sub__(
        self,
        other: "Vector"
    ) -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self._data - other._data)

    def __mul__(
        self,
        other: npt.ArrayLike
    ) -> "Vector":
        output = np.empty(self._data.shape, dtype=np.float64)
        for i in range(3):
            output[:, i] = self._data[:, i] * other
        return Vector(output)

    def __rmul__(
        self,
        other: npt.ArrayLike
    ) -> "Vector":
        output = np.empty(self._data.shape, dtype=np.float64)
        for i in range(3):
            output[:, i] = self._data[:, i] * other
        return Vector(output)

    def __truediv__(
        self,
        other: npt.ArrayLike
    ) -> "Vector":
        output = np.empty(self._data.shape, dtype=np.float64)
        for i in range(3):
            output[:, i] = self._data[:, i] / other
        return Vector(output)

    def __floordiv__(
        self,
        other: npt.ArrayLike
    ) -> "Vector":
        output = np.empty(self._data.shape, dtype=np.float64)
        for i in range(3):
            output[:, i] = self._data[:, i] // other
        return Vector(output)

    def __neg__(self) -> "Vector":
        return self.copy() * -1.0

    def __len__(self) -> int:
        return self._data.shape[0]

    def __iter__(self) -> _VectorIterator:
        if len(self) == 1:
            raise TypeError(
                "Cannot iterate through a single Vector"
            )
        return _VectorIterator(self)

    @property
    def vectors(self) -> npt.NDArray[np.float64]:
        """ Get the numpy array """
        return self._data

    @vectors.setter
    def vectors(self, vectors: npt.NDArray[np.float64]) -> None:
        if self._data.shape != vectors.shape:
            raise ValueError(
                f"Old ({self._data.shape}) and new ({vectors.shape})"
                " shapes don't match. Create a new instance of Vector instead."
            )
        self._data = vectors

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """ x-Component of Vector """
        return self[:, 0]

    @x.setter
    def x(self, value: npt.ArrayLike) -> None:
        if self[:, 0].shape != np.array(value).shape:
            raise ValueError(
                "New x-values have wrong length"
            )
        self[:, 0] = value

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """ y-Component of Vector """
        return self[:, 1]

    @y.setter
    def y(self, value: npt.ArrayLike) -> None:
        if self[:, 1].shape != np.array(value).shape:
            raise ValueError(
                "New y-values have wrong length"
            )
        self[:, 1] = value

    @property
    def z(self) -> npt.NDArray[np.float64]:
        """ z-Component of Vector """
        return self[:, 2]

    @z.setter
    def z(self, value: npt.ArrayLike) -> None:
        if self[:, 2].shape != np.array(value).shape:
            raise ValueError(
                "New z-values have wrong length "
                f"old={self[:,2].shape[0]} vs new={np.array(value).shape}"
            )
        self[:, 2] = value

    @property
    def mag(self) -> npt.NDArray[np.float64]:
        """ Return magnitude of vector """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def norm(self) -> "Vector":
        """ Return normalized Vector """
        result = self.copy()
        mag = self.mag
        result.x /= mag
        result.y /= mag
        result.z /= mag
        return result

    @property
    def phi(self) -> npt.NDArray[np.float64]:
        """ Return azimuth angle in rad from -PI to PI """
        return np.arctan2(self.y, self.x)

    @property
    def phi_deg(self) -> npt.NDArray[np.float64]:
        """ Return azimuth angle in degree from -180 to 180 """
        return self.phi * 180.0 / np.pi

    @property
    def cos_theta(self) -> npt.NDArray[np.float64]:
        """ Return cosine of polar angle from -1 to 1 """
        return self.z / self.mag

    @property
    def theta(self) -> npt.NDArray[np.float64]:
        """ Return polar angle in rad from 0 to PI e"""
        return np.arccos(self.cos_theta)

    @property
    def theta_deg(self) -> npt.NDArray[np.float64]:
        """ Return polar angle in degree form 0 to 180 """
        return self.theta * 180.0 / np.pi

    def copy(self) -> "Vector":
        """ Return deep copy """
        return Vector(self._data.copy())

    def dot(self, other: "Vector") -> npt.NDArray[np.float64]:
        """ Returrn dot product of Vector with *other*"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector") -> "Vector":
        """ 
        Return cross product between *self* and *other*
        """
        if self.x.shape[0] == 1:
            result = other.copy()
        else:
            result = self.copy()
        result.x = self.y * other.z - self.z * other.y
        result.y = self.z * other.x - self.x * other.z
        result.z = self.x * other.y - self.y * other.x
        return result

    def angle_between(self, other: "Vector"):
        """
        Return the angle between Vector and *other*

        Parameters
        ----------
        other : Vector
            The other Vector

        Returns
        -------
        angle : float
            The angle between *self* and *other*
        """
        return np.arccos(self.dot(other) / self.mag / other.mag)

    def rotated_around_x(
        self,
        angle: npt.ArrayLike
    ) -> "Vector":
        """
        Return a new vector which is rotated around the x-axis by *angle*

        Parameters
        ----------
        angle : `ArrayLike`
            angle(s) in rad

        Returns
        -------
        rotated_vector: `Vector`
        """
        output = Vector(np.empty(self._data.shape))
        output.x = self.x
        output.y = np.cos(angle) * self.y - np.sin(angle) * self.z
        output.z = np.sin(angle) * self.y + np.cos(angle) * self.z
        return output

    def rotated_around_y(
        self,
        angle: npt.ArrayLike
    ) -> "Vector":
        """
        Return a new vector which is rotated around the y-axis by *angle*

        Parameters
        ----------
        angle : `ArrayLike`
            angle(s) in rad

        Returns
        -------
        rotated_vector: `Vector`
        """
        output = Vector(np.empty(self._data.shape))
        output.x = np.cos(angle) * self.x + np.sin(angle) * self.z
        output.y = self.y
        output.z = - np.sin(angle) * self.x + np.cos(angle) * self.z
        return output

    def rotated_around_z(
        self,
        angle: npt.ArrayLike
    ) -> "Vector":
        """
        Return a new vector which is rotated around the z-axis by *angle*

        Parameters
        ----------
        angle : `ArrayLike`
            angle(s) in rad

        Returns
        -------
        rotated_vector: `Vector`
        """
        output = Vector(np.empty(self._data.shape))
        output.x = np.cos(angle) * self.x - np.sin(angle) * self.y
        output.y = np.sin(angle) * self.x + np.cos(angle) * self.y
        output.z = self.z
        return output

    def remove_where(
        self,
        mask: npt.NDArray[np.bool_]
    ) -> "Vector":
        """
        Remove every vector `i` where `mask[i] == True`

        Parameters
        ----------
        mask: `numpy.ndarray`, shape `(len(self),)` 
            Array of booleans.

        Returns
        -------
        `Vector`

        Examples
        --------
        ::

            >>> vec = Vector([[1, 2, 3], [4, 5, 6]])
            [[1. 2. 3.]
             [4. 5. 6.]]
            >>> vec.remove_where(vec.z == 3)
            [[4. 5. 6.]]
        """
        result_x = np.ma.compressed(
            np.ma.masked_where(mask, self.x, copy=True))
        result_y = np.ma.compressed(
            np.ma.masked_where(mask, self.y, copy=True))
        result_z = np.ma.compressed(
            np.ma.masked_where(mask, self.z, copy=True))
        return Vector(np.array([result_x, result_y, result_z]).T)

    def keep_where(
        self,
        mask: npt.NDArray[np.bool_]
    ) -> "Vector":
        """
        Keep every vector `i` where `mask[i] == True`

        Parameters
        ----------
        mask: `numpy.ndarray`, shape `(len(self),)` 
            Array of booleans.

        Returns
        -------
        `Vector`

        Examples
        --------
        ::

            >>> vec = Vector([[1, 2, 3], [4, 5, 6]])
            [[1. 2. 3.]
             [4. 5. 6.]]
            >>> vec.keep_where(vec.z == 3)
            [[1. 2. 3.]]
        """
        return self.remove_where(np.logical_not(mask))


class CoordinateSystem:
    def __init__(
        self,
        vec1: Vector,
        vec2: Vector,
        vec3: Optional[Vector] = None
    ) -> None:
        if vec3 is not None:
            self.x_axis = vec1.norm
            self.y_axis = vec2.norm
            self.z_axis = vec3.norm
        else:
            self.z_axis = vec1.norm
            self.y_axis = vec1.cross(vec2).norm
            self.x_axis = self.y_axis.cross(vec1).norm

    def __getitem__(self, key):
        if key == 0:
            return self.x_axis
        elif key == 1:
            return self.y_axis
        elif key == 2:
            return self.z_axis
        else:
            raise IndexError

    def project_vector(
        self,
        vec: Vector
    ) -> Vector:
        """ Project vector into coordinate system """
        result = vec.copy()
        result.x = vec.dot(self.x_axis)
        result.y = vec.dot(self.y_axis)
        result.z = vec.dot(self.z_axis)
        return result

    def remove_where(
        self,
        mask
    ) -> "CoordinateSystem":
        """
        Remove every CoordinateSystem `i` where `mask[i] == True`

        Paramters
        ---------
        mask: `numpy.ndarray`, shape `(len(self),)` 
            Array of booleans.

        Returns
        -------
        `CoordinateSystem`
        """
        result_x = np.ma.compressed(
            np.ma.masked_where(mask, self.x_axis, copy=True))
        result_y = np.ma.compressed(
            np.ma.masked_where(mask, self.y_axis, copy=True))
        result_z = np.ma.compressed(
            np.ma.masked_where(mask, self.z_axis, copy=True))
        return CoordinateSystem(result_x, result_y, result_z)

    def keep_where(
        self,
        mask
    ) -> "CoordinateSystem":
        """
        Keep every CoordinateSystem `i` where `mask[i] == True`

        Paramters
        ---------
        mask: `numpy.ndarray`, shape `(len(self),)` 
            Array of booleans.

        Returns
        -------
        `CoordinateSystem`
        """
        return self.remove_where(np.logical_not(mask))

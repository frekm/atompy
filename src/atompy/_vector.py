import math
import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Iterator


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
    Wrapper class for numpy arrays that represent vectors.

    Parameters
    ----------
    vectors : array_like, shape (N, 3) or (3,)
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
            msg = (f"dimension of input array is {vectors_.ndim}, but it "
                   "needs to be 1 or 2")
            raise ValueError(msg)

        if self._data.shape[1] != 3:
            msg = (f"Shape of input array is {vectors_.shape}, but it needs "
                   " to be (N, 3) or (3,)")
            raise ValueError(msg)

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
            output[:, i] = self._data[:, i] * np.array(other)
        return Vector(output)

    def __rmul__(
        self,
        other: npt.ArrayLike
    ) -> "Vector":
        output = np.empty(self._data.shape, dtype=np.float64)
        for i in range(3):
            output[:, i] = self._data[:, i] * np.array(other)
        return Vector(output)

    def __truediv__(
        self,
        other: npt.ArrayLike
    ) -> "Vector":
        output = np.empty(self._data.shape, dtype=np.float64)
        for i in range(3):
            output[:, i] = self._data[:, i] / np.array(other)
        return Vector(output)

    def __floordiv__(
        self,
        other: npt.ArrayLike
    ) -> "Vector":
        output = np.empty(self._data.shape, dtype=np.float64)
        for i in range(3):
            output[:, i] = self._data[:, i] // np.array(other)
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
    def ndarray(self) -> npt.NDArray[np.float64]:
        """ Get the underlying numpy array.

        Examples
        --------
        >>> vec = Vector([[1, 2, 3], [4, 5, 6]])
        >>> vec.ndarray
        array([[1., 2., 3.],
               [4., 5., 6.]])
        """
        return self._data

    @ndarray.setter
    def ndarray(self, vectors: npt.NDArray[np.float64]) -> None:
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
                f"old={self[:, 2].shape[0]} vs new={np.array(value).shape}"
            )
        self[:, 2] = value

    @property
    def magnitude(self) -> npt.NDArray[np.float64]:
        """ Return magnitude of vector """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def mag(self) -> npt.NDArray[np.float64]:
        """ Alias for :attr:`.Vector.magnitude` """
        return self.magnitude

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
        """ Return polar angle in degree from 0 to 180 """
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
        other : :class:`.Vector`
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
        angle : array_like
            angle(s) in rad

        Returns
        -------
        rotated_vector : :class:`.Vector`
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
        angle : array_like
            angle(s) in rad

        Returns
        -------
        rotated_vector : :class:`.Vector`
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
        angle : array_like
            angle(s) in rad

        Returns
        -------
        rotated_vector : :class:`.Vector`
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
        mask : `numpy.ndarray`, shape `(len(self),)` 
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
        mask : `numpy.ndarray`, shape `(len(self),)` 
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


class SingleVector:
    """
    Data type representing a single vector.

    If you want to work with multiple vectors, consider using
    :class:`.Vector`.

    Parameters
    ----------
    x, y, z : float
        Components of the vector

    Examples
    --------
    ::

        >>> vec  = Vector(1, 2, 3)
        >>> vec.x
        1.0
    """
    def __init__(self, x: float, y: float, z:float) -> None:
        self._data = Vector([x, y, z])

    def __getitem__(self, key) -> float:
        return float(self._data[key])

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    def __add__(self, other: "SingleVector") -> "SingleVector":
        if not isinstance(other, SingleVector):
            return NotImplemented
        return SingleVector(self.x + other.x,
                            self.y + other.y,
                            self.z + other.z)

    def __sub__(self, other: "SingleVector") -> "SingleVector":
        if not isinstance(other, "SingleVector"):
            return NotImplemented
        else:
            return SingleVector(self.x - other.x,
                                self.y - other.y,
                                self.z - other.z)

    def __mul__(self, scale: float) -> "SingleVector":
        if not isinstance(scale, float):
            return NotImplemented
        return SingleVector(self.x * scale, self.y * scale, self.z * scale)

    def __rmul__(self, scale: float) -> "SingleVector":
        if not isinstance(scale, float):
            return NotImplemented
        return SingleVector(self.x * scale, self.y * scale, self.z * scale)

    def __truediv__(self, scale: float) -> "SingleVector":
        if not isinstance(scale, float):
            return NotImplemented
        return SingleVector(self.x / scale, self.y / scale, self.z / scale)

    def __floordiv__(self, scale: float) -> "SingleVector":
        if not isinstance(scale, float):
            return NotImplemented
        return SingleVector(self.x // scale, self.y // scale, self.z // scale)

    def __neg__(self) -> "SingleVector":
        return SingleVector(-self.x, -self.y, -self.z)

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator[float]:
        return [self.x, self.y, self.z].__iter__()


    @property
    def x(self) -> float:
        """ x-Component of Vector """
        return self.x

    @x.setter
    def x(self, value: float) -> None:
        self.x = float(value)

    @property
    def y(self) -> float:
        """ y-Component of Vector """
        return self.y

    @y.setter
    def y(self, value: float) -> None:
        self.y = float(value)

    @property
    def z(self) -> float:
        """ z-Component of Vector """
        return self.z

    @z.setter
    def z(self, value: float) -> None:
        self.z = float(value)

    @property
    def magnitude(self) -> float:
        """ Return magnitude of vector """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def mag(self) -> float:
        """ Alias for :attr:`.SingleVector.magnitude` """
        return self.magnitude

    @property
    def norm(self) -> "SingleVector":
        """ Return normalized Vector """
        mag = self.mag
        return SingleVector(self.x/mag, self.y/mag, self.z/mag)

    @property
    def phi(self) -> float:
        """ Return azimuth angle in rad from -PI to PI """
        return np.arctan2(self.y, self.x)

    @property
    def phi_deg(self) -> float:
        """ Return azimuth angle in degree from -180 to 180 """
        return self.phi * 180.0 / np.pi

    @property
    def cos_theta(self) -> float:
        """ Return cosine of polar angle from -1 to 1 """
        return self.z / self.mag

    @property
    def theta(self) -> float:
        """ Return polar angle in rad from 0 to PI e"""
        return np.arccos(self.cos_theta)

    @property
    def theta_deg(self) -> float:
        """ Return polar angle in degree from 0 to 180 """
        return self.theta * 180.0 / np.pi

    def dot(self, other: "SingleVector") -> float:
        """ Returrn dot product of Vector with *other*"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "SingleVector") -> "SingleVector":
        """ 
        Return cross product between *self* and *other*
        """
        result_x = self.y * other.z - self.z * other.y
        result_y = self.z * other.x - self.x * other.z
        result_z = self.x * other.y - self.y * other.x
        return SingleVector(result_x, result_y, result_z)

    def angle_between(self, other: "SingleVector"):
        """
        Return the angle between Vector and *other*

        Parameters
        ----------
        other : :class:`.SingleVector`
            The other Vector

        Returns
        -------
        angle : float
            The angle between *self* and *other*
        """
        return np.arccos(self.dot(other) / self.mag / other.mag)

    def rotated_around_x(self, angle: float) -> "SingleVector":
        """
        Return a new vector which is rotated around the x-axis by *angle*

        Parameters
        ----------
        angle : float
            angle in rad

        Returns
        -------
        rotated_vector : :class:`.SingleVector`
        """
        output_x = self.x
        output_y = np.cos(angle) * self.y - np.sin(angle) * self.z
        output_z = np.sin(angle) * self.y + np.cos(angle) * self.z
        return SingleVector(output_x, output_y, output_z)

    def rotated_around_y(self, angle: float) -> "SingleVector":
        """
        Return a new vector which is rotated around the y-axis by *angle*

        Parameters
        ----------
        angle : float
            angle in rad

        Returns
        -------
        rotated_vector : :class:`.SingleVector`
        """
        output_x = np.cos(angle) * self.x + np.sin(angle) * self.z
        output_y = self.y
        output_z = - np.sin(angle) * self.x + np.cos(angle) * self.z
        return SingleVector(output_x, output_y, output_z)

    def rotated_around_z(self, angle: float) -> "SingleVector":
        """
        Return a new vector which is rotated around the z-axis by *angle*

        Parameters
        ----------
        angle : float
            angle in rad

        Returns
        -------
        rotated_vector : :class:`.SingleVector`
        """
        output_x = np.cos(angle) * self.x - np.sin(angle) * self.y
        output_y = np.sin(angle) * self.x + np.cos(angle) * self.y
        output_z = self.z
        return SingleVector(output_x, output_y, output_z)


class CoordinateSystem:
    """
    Create a coordinate systems defined by `vec1`, `vec2` (and `vec3`).

    The new coordinate system will be defined by the following unit
    vectors:
    
    - `z` will exactly align with `vec1`.
    - `y` is a unit vector along ``vec1.cross(vec2)``
    - `x` is a unit vector along ``y.cross(vec1)``

    The coordinate system is right-handed and orthogonal.

    If `vec3` is provided, the behaviour changes (see below).

    Parameters
    ----------

    vec1, vec2 : :class:`Vector`
        The collection of vectors defining the coordinate systems.

        Both collections need to have the same length.

    vec3 : :class:`Vector`, optional
        If three collections of vectors are provided, the coordinate
        systems will simply use the normalized `vec1`, `vec2`, `vec3`
        as `x`, `y`, `z`.
    """

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
        """
        Project vector into coordinate system
        
        Parameters
        ----------
        vec : :class:`.Vector`
            The collection of vectors to project.

        Returns
        -------
        projected_vectors : :class:`.Vector`
        """
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
        Remove every CoordinateSystem ``i`` where ``mask[i] == True``

        Parameters
        ----------
        mask : ndarray
            Array of booleans.

        Returns
        -------
        :class:`.CoordinateSystem`
        """
        result_x = Vector(self.x_axis[mask==False])
        result_y = Vector(self.y_axis[mask==False])
        result_z = Vector(self.z_axis[mask==False])
        return CoordinateSystem(result_x, result_y, result_z)

    def keep_where(
        self,
        mask
    ) -> "CoordinateSystem":
        """
        Keep every CoordinateSystem ``i`` where ``mask[i] == True``.

        Parameters
        ----------
        mask : ndarray
            Array of booleans.

        Returns
        -------
        :class:`.CoordinateSystem`
        """
        return self.remove_where(np.logical_not(mask))

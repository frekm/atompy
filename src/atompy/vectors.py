import numpy as np
import collections
from numpy.typing import NDArray, ArrayLike
from typing import TypeVar, Union, overload, Sequence, Self

T = TypeVar("T")

VectorLike = Union[ArrayLike, "Vector"]
VectorArrayLike = Union[ArrayLike, "Vector", "VectorArray", Sequence["Vector"]]


def asvector(input: VectorLike) -> "Vector":
    """
    Convert Vector-like input to a Vector.

    Parameters
    ----------
    input : array_like or :class:`.Vector`

    Returns
    -------
    vector : :class:`.Vector`

    See also
    --------
    asvectorarray

    Examples
    --------

    ::

        >>> import aplepy as ap
        >>> ap.asvector((1, 2, 3))
        Vector(1.0, 2.0, 3.0)
        >>> ap.asvector(ap.Vector(1, 2, 3))
        Vector(1.0, 2.0, 3.0)
    """
    # TODO use `asvector` in operator overloads of Vector
    if isinstance(input, Vector):
        return input
    else:
        try:
            vec = Vector(*input)  # type: ignore
        except (TypeError, ValueError):
            raise ValueError("cannot create a Vector from input")
        return vec


def _process_vectorarray_init(vectors: VectorArrayLike) -> NDArray[np.float64]:
    _vecs = np.asarray(vectors)
    if _vecs.dtype == VectorArray:
        _vecs = np.array([v.asarray() for v in _vecs])
    elif _vecs.ndim == 1:
        _vecs = np.array([_vecs])
    elif _vecs.ndim != 2:
        raise ValueError(
            f"dimension of input array is {_vecs.ndim}, but it needs to be 1 or 2"
        )

    if _vecs.shape[1] != 3:
        raise ValueError(
            f"Shape of input array is {_vecs.shape}, but it needs to be (N, 3) or (3,)"
        )
    return _vecs.astype(np.float64)


def asvectorarray(input: VectorArrayLike) -> "VectorArray":
    """
    Convert VectorArray-like input to a VectorArray.

    Parameters
    ----------
    input : VectorArrayLike
        Either array_like [with shape (3, N)], :class:`.Vector`,
        sequence of :class:`.Vector`, or :class:`.VectorArray`.

    Returns
    -------
    vector_array : :class:`.VectorArray`

    See also
    --------
    asvector

    Examples
    --------

    ::

        >>> ap.asvectorarray(((1, 2, 3), (4, 5, 6)))
        VectorArray([[1. 2. 3.]
                    [4. 5. 6.]])
        >>> ap.asvectorarray(((1, 2, 3), (4, 5, 6)))
        VectorArray([[1. 2. 3.]
                    [4. 5. 6.]])
    """
    # TODO use `asvector` in operator overloads of VectorArray
    if isinstance(input, VectorArray):
        return input
    elif isinstance(input, Vector):
        return VectorArray(input.asarray())
    else:
        return VectorArray(_process_vectorarray_init(input))


class VectorArrayIter:
    def __init__(self, vectors: NDArray[np.float64]):
        self.vectors = vectors
        self.index = 0

    def __iter__(self) -> "VectorArrayIter":
        return self

    def __next__(self) -> "Vector":
        if self.index == len(self.vectors):
            raise StopIteration
        self.index += 1
        return Vector(*self.vectors[self.index - 1])


class Vector:
    """
    Class representing a single vector.

    .. tip::

        If you want to store an array of vectors, consider :class:`.VectorArray`.

    Parameters
    ----------
    x, y, z : float
        The x/y/z component of the vector.

    Attributes
    ----------
    x, y, z : float

    See also
    --------
    asvector

    Examples
    --------
    ::

        >>> vec = ap.Vector(1, 2, 3)
        >>> vec
        Vector(1.0, 2.0, 3.0)
        >>> vec.x
        1.0

    For more examples, see :doc:`../../tutorials/vectors`.
    """

    def __init__(self, x: float, y: float, z: float):
        try:
            array = [float(x), float(y), float(z)]
        except TypeError:
            raise ValueError(
                "Conversion to float failed. Perhaps you want to use VectorArray?"
            )
        self._arr = np.array(array).astype(np.float64)

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, val: float):
        self[0] = val

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, val: float):
        self[1] = val

    @property
    def z(self) -> float:
        return self[2]

    @z.setter
    def z(self, val: float):
        self[2] = val

    def __getitem__(self, i) -> float:
        return float(self._arr[i])

    def __setitem__(self, i, val):
        self._arr[i] = val

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return f"Vector{str(self)}"

    def __iter__(self):
        return iter(self._arr)

    def __neg__(self) -> "Vector":
        return Vector(*(-self._arr))

    def __add__(self, other: "Vector") -> "Vector":
        if isinstance(other, Vector):
            return Vector(*(self._arr + other._arr))
        else:
            return NotImplemented

    def __iadd__(self, other: "Vector") -> "Vector":
        if isinstance(other, Vector):
            self._arr += other._arr
            return self
        else:
            return NotImplemented

    def __sub__(self, other: "Vector") -> "Vector":
        if isinstance(other, Vector):
            return Vector(*(self._arr - other._arr))
        else:
            return NotImplemented

    def __isub__(self, other: "Vector") -> "Vector":
        if isinstance(other, Vector):
            self._arr -= other._arr
            return self
        else:
            return NotImplemented

    def __mul__(self, fac: float) -> "Vector":
        return self.scale(fac, copy=True)

    def __rmul__(self, fac: float) -> "Vector":
        return self.__mul__(fac)

    def __imul__(self, fac: float) -> "Vector":
        return self.scale(fac, copy=False)

    def __truediv__(self, divider: float) -> "Vector":
        return self.scale(1.0 / divider, copy=True)

    def __rtruediv__(self, divider: float) -> "Vector":
        return self.__mul__(1.0 / divider)

    def __itruediv__(self, divider: float) -> "Vector":
        return self.scale(1.0 / divider, copy=False)

    def __eq__(self, other: VectorLike) -> bool:
        other = asvector(other)
        return bool(np.all(self._arr == other._arr))

    def scale(self, fac: float, copy: bool = True) -> "Vector":
        """
        Scale the vector.

        Parameters
        ----------
        fac : float
            The scaling factor.

        copy : bool, default True
            If True, return a copy of the original vector, otherwise scale in-place.

        Returns
        -------
        scaled_vector : :class:`.Vector`
            The scaled vector.
        """
        if copy:
            new_arr = self._arr * fac
            return Vector(*new_arr)
        else:
            self._arr *= fac
            return self

    @overload
    def dot(self, other: "Vector") -> float: ...
    @overload
    def dot(self, other: "VectorArray") -> NDArray[np.float64]: ...
    def dot(self, other: Union["Vector", "VectorArray"]) -> float | NDArray[np.float64]:
        """
        Calculate dot (inner) product with `other`.

        Parameters
        ----------
        other : :class:`.Vector` or :class:`.VectorArray`

        Returns
        -------
        dot_product : float or ndarray
            If `other` is type :class:`.Vector`, returns float.

            If `other` is type :class:`.VectorArray`, returns ndarray.

        Examples
        --------
        ::

            >>> vec_a = ap.Vector(1, 2, 3)
            >>> vec_b = ap.Vector(3, 2, 1)
            >>> vec_a.dot(vec_b)
            10.0
        """
        if isinstance(other, Vector) or isinstance(other, VectorArray):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise ValueError("other must be type Vector or VectorArray")

    @overload
    def cross(self, other: "Vector") -> "Vector": ...
    @overload
    def cross(self, other: "VectorArray") -> "VectorArray": ...
    def cross(
        self, other: Union["Vector", "VectorArray"]
    ) -> Union["Vector", "VectorArray"]:
        """
        Calculate cross (outer) product with `other`.

        Parameters
        ----------
        other : :class:`.Vector` or :class:`.VectorArray`

        Returns
        -------
        cross_product : :class:`Vector` or :class:`.VectorArray`
            If `other` is type :class:`.Vector`, returns :class:`.Vector`.

            If `other` is type :class:`.VectorArray`, returns :class:`.VectorArray`.

        Examples
        --------
        ::

            >>> vec_a = ap.Vector(1, 0, 0)
            >>> vec_b = ap.Vector(0, 1, 0)
            >>> vec_a.cross(vec_b)
            Vector(0.0, 0.0, 1.0)
        """
        result_x = self.y * other.z - self.z * other.y
        result_y = self.z * other.x - self.x * other.z
        result_z = self.x * other.y - self.y * other.x
        if isinstance(other, Vector):
            return Vector(result_x, result_y, result_z)  # type: ignore
        elif isinstance(other, VectorArray):
            result = np.array([result_x, result_y, result_z]).T
            return VectorArray(result)
        else:
            raise ValueError("other must be type Vector or VectorArray")

    def mag(self) -> float:
        """
        Calculate magnitude of vector.

        Returns
        -------
        magnitude : float
        """
        return float(np.sqrt(self.dot(self)))

    def norm(self, copy=True) -> "Vector":
        """
        Return the vector normalized to 1.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy of the vector, otherwise normalize the vector
            in place.

        Returns
        -------
        normalized_vectors : :class:`.Vector`
        """
        return self.scale(1.0 / self.mag(), copy=copy)

    def phi(self) -> float:
        r"""
        Calculate azimuth angle in rad.

        Calculates :math:`\arctan2(y, x)`, corresponding to the azimuth angle
        :math:`\phi` in
        `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions>`__.

        Returns
        -------
        phi : float
            The azimuth angle in rad within (−π, π].
        """
        return float(np.arctan2(self.y, self.x))

    def cos_theta(self) -> float:
        r"""
        Calculate cosine of the polar angle.

        Calculates :math:`v_z / |\vec{v}|`, corresponding to the cosine of the polar
        angle :math:`\theta` in
        `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions>`__.

        Returns
        -------
        cos_theta : float
            The cosine of the polar angle within [−1, 1].
        """
        return float(self.z / self.mag())

    def theta(self) -> float:
        r"""
        Calculate polar angle in rad.

        Calculates :math:`\arccos(v_z / |\vec{v}|)`, corresponding to the polar angle
        :math:`\theta` in
        `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions>`__.

        Returns
        -------
        theta : float
            The angle in rad within [0, π].
        """
        return float(np.arccos(self.cos_theta()))

    @overload
    def angle_to(self, other: "Vector") -> float: ...
    @overload
    def angle_to(self, other: "VectorArray") -> NDArray[np.float64]: ...
    def angle_to(
        self, other: Union["Vector", "VectorArray"]
    ) -> float | NDArray[np.float64]:
        r"""
        Calculate the angle to vector `other`.

        Calculates :math:`\arccos\vec{v}_1 \cdot \vec{v}_2 / (v_1 v_2)`.

        Parameters
        ----------
        other : :class:`.Vector` or :class:`.VectorArray`

        Returns
        -------
        angle : float or ndarray
            The angle(s) in rad within [0, π].

            If `other` is type :class:`.Vector`, returns float.

            If `other` is type :class:`.VectorArray`, returns ndarray.
        """
        res = np.arccos(self.dot(other) / self.mag() / other.mag())
        return float(res) if isinstance(other, Vector) else res

    def copy(self) -> "Vector":
        """Return a copy of the vector."""
        return Vector(*self._arr.copy())

    def asarray(self) -> NDArray[np.float64]:
        """
        Return as a numpy array.

        Returns
        -------
        array : ndarray
            (x, y, z)

        Examples
        --------
        ::

            >>> ap.Vector(1, 2, 3).asarray()
            array([1., 2., 3.])
        """
        return self._arr

    def rotate(self, angle_rad: float, axis: VectorLike) -> "Vector":
        """
        Rotate a vector by `angle_rad` around `axis`

        Rotates counter-clockwise in a right-handed coordinate system, where the axis
        around which is rotated points towards the obsorver.

        Parameters
        ----------
        angle_rad : float
            The rotation angle in radian.

        axis : :class:`.Vector`
            The axis around which the vector is rotated.

            E.g., ``axis = Vector(0, 0, 1)`` will rotate the vector around the z-axis,
            that is, within the xy-plane.

            So far, `axis` must be along x, y, or z-direction.

        Returns
        -------
        rotated_vector : :class:`.Vector`
        """
        axis = asvector(axis).norm(copy=True)
        is_xaxis = axis == (1.0, 0.0, 0.0)
        is_yaxis = axis == (0.0, 1.0, 0.0)
        is_zaxis = axis == (0.0, 0.0, 1.0)
        if not (is_xaxis or is_yaxis or is_zaxis):
            # TODO
            raise NotImplementedError(
                "currently `axis` must be along x, y, or z direction"
            )

        c = angle_rad if is_xaxis else 0.0  # roll
        b = angle_rad if is_yaxis else 0.0  # pitch
        a = angle_rad if is_zaxis else 0.0  # yaw

        TrigAngle = collections.namedtuple("TrigAngle", ["a", "b", "c"])
        cos = TrigAngle(np.cos(a), np.cos(b), np.cos(c))
        sin = TrigAngle(np.sin(a), np.sin(b), np.sin(c))

        rot_matrix = np.empty((3, 3))
        rot_matrix[0, 0] = cos.a * cos.b
        rot_matrix[0, 1] = cos.a * sin.b * sin.c - sin.a * cos.c
        rot_matrix[0, 2] = cos.a * sin.b * cos.c + sin.a * sin.c
        rot_matrix[1, 0] = sin.a * cos.b
        rot_matrix[1, 1] = sin.a * sin.b * sin.c + cos.a * cos.c
        rot_matrix[1, 2] = sin.a * sin.b * cos.c - cos.a * sin.c
        rot_matrix[2, 0] = -sin.b
        rot_matrix[2, 1] = cos.b * sin.c
        rot_matrix[2, 2] = cos.b * cos.c

        new_components = np.zeros(3)
        for (i, j), _ in np.ndenumerate(rot_matrix):
            new_components[i] += rot_matrix[i, j] * self[j]

        return Vector(*new_components)


class VectorArray:
    """
    Class representing an array of vectors.

    .. tip::

        If you want to store a single vector, consider :class:`.Vector`.

    Parameters
    ----------
    vectors : array_like
        The vectors.

        Should have shape (N, 3).

        E.g., storing two vectors, do something like::

            >>> v1 = (1, 2, 3)
            >>> v2 = (4, 5, 6)
            >>> vecs = VectorArray((v1, v2))
            >>> vecs
            VectorArray([[1. 2. 3.]
                         [4. 5. 6.]])

    Attributes
    ----------
    x, y, z : ndarray
        The x, y, z components of all vectors. For instance, in the example of the
        input parameter `vectors`::

            >>> vecs.x
            array([1., 4.])

    Examples
    --------
    For more examples, see :doc:`../../tutorials/vectors`.
    """

    def __init__(self, vectors: VectorArrayLike):
        self._arr = _process_vectorarray_init(vectors)

    def asarray(self) -> NDArray[np.float64]:
        """
        Return as a numpy array.

        Returns
        -------
        array : ndarray

        Examples
        --------
        ::

            >>> vecs = ap.VectorArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> vecs.asarray()
            array([[1., 2., 3.],
                   [4., 5., 6.],
                   [7., 8., 9.]])
            >>> vecs.asarray()[1:, :2]
            array([[4., 5.],
                   [7., 8.]])
            >>> vecs.asarray()[1:,:2] = 0, 1
            vecs
            VectorArray([[1. 2. 3.]
                         [0. 1. 6.]
                         [0. 1. 9.]])
        """
        return self._arr

    @property
    def x(self) -> NDArray[np.float64]:
        return self._arr[:, 0]

    @x.setter
    def x(self, val: ArrayLike):
        self._arr[:, 0] = val

    @property
    def y(self) -> NDArray[np.float64]:
        return self._arr[:, 1]

    @y.setter
    def y(self, val: ArrayLike):
        self._arr[:, 1] = val

    @property
    def z(self) -> NDArray[np.float64]:
        return self._arr[:, 2]

    @z.setter
    def z(self, val: ArrayLike):
        self._arr[:, 2] = val

    @overload
    def __getitem__(self, i: int) -> Vector: ...
    @overload
    def __getitem__(self, i: slice) -> Self: ...

    def __getitem__(self, i: int | slice) -> Vector | Self:
        if isinstance(i, int):
            return Vector(*self._arr[i])
        elif isinstance(i, slice):
            return type(self)(self._arr[i])
        else:
            raise TypeError(f"i must be int or slice, but is {type(i)}")

    def __setitem__(self, i: int | slice, val: VectorArrayLike | VectorLike) -> None:
        if isinstance(i, int):
            val = tuple(iter(asvector(val)))  # type: ignore
        elif isinstance(i, slice):
            val = asvectorarray(val)._arr
        self._arr[i] = val

    def __str__(self) -> str:
        return str(self._arr)

    def __repr__(self) -> str:
        # TODO format using repr(self._arr) (includes updates to tutorial/vector.rst)
        string = str(self).replace("\n", "\n            ")
        return f"VectorArray({string})"

    def __iter__(self) -> VectorArrayIter:
        return VectorArrayIter(self._arr)

    def __len__(self) -> int:
        return len(self._arr)

    def __neg__(self) -> "VectorArray":
        return VectorArray(-self._arr)

    def __add__(self, other: Union["VectorArray", "Vector"]) -> "VectorArray":
        if isinstance(self, VectorArray) or isinstance(self, Vector):
            return VectorArray(self._arr + other._arr)
        else:
            return NotImplemented

    def __radd__(self, other: Union["VectorArray", "Vector"]) -> "VectorArray":
        return self + other

    def __iadd__(self, other: Union["VectorArray", "Vector"]) -> "VectorArray":
        if isinstance(self, VectorArray) or isinstance(self, Vector):
            self._arr += other._arr
            return self
        else:
            return NotImplemented

    def __sub__(self, other: Union["VectorArray", "Vector"]) -> "VectorArray":
        if isinstance(self, VectorArray) or isinstance(self, Vector):
            return VectorArray(self._arr - other._arr)
        else:
            return NotImplemented

    def __rsub__(self, other: Union["VectorArray", "Vector"]) -> "VectorArray":
        return -self + other

    def __isub__(self, other: Union["VectorArray", "Vector"]) -> "VectorArray":
        if isinstance(self, VectorArray) or isinstance(self, Vector):
            self._arr -= other._arr
            return self
        else:
            return NotImplemented

    def __mul__(self, fac: ArrayLike) -> "VectorArray":
        try:
            _fac = np.asarray(fac).astype(np.float64)
        except ValueError:
            return NotImplemented
        return self.scale(_fac, copy=True)

    def __rmul__(self, fac: ArrayLike) -> "VectorArray":
        return self.__mul__(fac)

    def __imul__(self, fac: ArrayLike) -> "VectorArray":
        try:
            _fac = np.asarray(fac).astype(np.float64)
        except ValueError:
            return NotImplemented
        return self.scale(_fac, copy=False)

    def __truediv__(self, fac: ArrayLike) -> "VectorArray":
        try:
            _fac = 1.0 / np.asarray(fac).astype(np.float64)
        except ValueError:
            return NotImplemented
        return self.scale(_fac, copy=True)

    def __rtruediv__(self, fac: ArrayLike) -> "VectorArray":
        return self.__truediv__(fac)

    def __itruediv__(self, fac: ArrayLike) -> "VectorArray":
        try:
            _fac = 1.0 / np.asarray(fac).astype(np.float64)
        except ValueError:
            return NotImplemented
        return self.scale(_fac, copy=False)

    def __eq__(self, other: "VectorArray") -> bool:
        return bool(np.all(self._arr == other._arr))

    def scale(self, fac: ArrayLike, copy: bool = True) -> "VectorArray":
        """
        Scale the vector.

        Parameters
        ----------
        fac : array_like
            The scaling factor(s).

            A single factor scales all vectors the same.

            If as many factors as vectors are provided, scale each vector individually.

        copy : bool, default True
            If True, return a copy of the original vector, otherwise scale in-place.

        Returns
        -------
        scaled_vector : :class:`.VectorArray`
            The scaled vectors.
        """
        fac = np.asarray(fac).astype(np.float64)
        if copy:
            if fac.ndim > 0:
                new_arr = np.einsum("ij,i->ij", self._arr, fac)
            else:
                new_arr = self._arr * fac
            return VectorArray(new_arr)
        else:
            if fac.ndim > 0:
                np.einsum("ij,i->ij", self._arr, fac, out=self._arr)
            else:
                np.multiply(self._arr, fac, out=self._arr)
            return self

    def dot(self, other: Union["Vector", "VectorArray"]) -> NDArray[np.float64]:
        """
        Calculate dot (inner) product with `other`.

        Parameters
        ----------
        other : :class:`.Vector` or :class:`.VectorArray`

        Returns
        -------
        dot_product : ndarray

        Examples
        --------
        ::

            >>> vecs_a = ap.VectorArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> vecs_b = ap.VectorArray([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
            >>> vecs_a.dot(vecs_b)
            array([50., 77., 50.])
        """
        if isinstance(other, Vector) or isinstance(other, VectorArray):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise ValueError("other must be type Vector or VectorArray")

    def cross(self, other: Union["Vector", "VectorArray"]) -> "VectorArray":
        """
        Calculate cross (outer) product with `other`.

        Parameters
        ----------
        other : :class:`.Vector` or :class:`.VectorArray`

        Returns
        -------
        cross_product : :class:`.VectorArray`

        Examples
        --------
        ::

            >>> vecs_a = ap.VectorArray([[1, 0, 0], [0, 1, 0]])
            >>> vecs_b = ap.VectorArray([[0, 1, 0], [0, 0, 1]])
            >>> vecs_a.cross(vecs_b)
            VectorArray([[0. 0. 1.]
                         [1. 0. 0.]])
        """
        if isinstance(other, Vector) or isinstance(other, VectorArray):
            result_x = self.y * other.z - self.z * other.y
            result_y = self.z * other.x - self.x * other.z
            result_z = self.x * other.y - self.y * other.x
            result = np.array([result_x, result_y, result_z]).T
            return VectorArray(result)
        else:
            raise ValueError("other must be type Vector or VectorArray")

    def mag(self) -> NDArray[np.float64]:
        """
        Calculate magnitude of vector.

        Returns
        -------
        magnitude : float
        """
        return np.sqrt(self.dot(self))

    def norm(self, copy=True) -> "VectorArray":
        """
        Return the vectors normalized to 1.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy of the vector, otherwise normalize all vectors
            in place.

        Returns
        -------
        normalized_vectors : :class:`.VectorArray`
        """
        return self.scale(1 / self.mag(), copy=copy)

    def phi(self) -> NDArray[np.float64]:
        r"""
        Calculate azimuth angles in rad.

        For all vectors, calculates :math:`\arctan2(y, x)`, corresponding to the
        azimuth angle :math:`\phi` in
        `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions>`__.

        Returns
        -------
        phis : ndarray
            The azimuth angles in rad within (−π, π].
        """
        return np.arctan2(self.y, self.x)

    def cos_theta(self) -> NDArray[np.float64]:
        r"""
        Calculate cosines of the polar angles.

        For all vectors, calculates :math:`v_z / |\vec{v}|`, corresponding to the
        cosine of the polar angle :math:`\theta` in
        `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions>`__.

        Returns
        -------
        cos_theta : ndarray
            The cosines of the polar angles within [−1, 1].
        """
        return self.z / self.mag()

    def theta(self) -> NDArray[np.float64]:
        r"""
        Calculate polar angles in rad.

        Calculates :math:`\arccos(v_z / |\vec{v}|)`, corresponding to the polar angles
        :math:`\theta` in
        `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions>`__.

        Returns
        -------
        thetas : ndarray
            The angles in rad within [−0, π].
        """
        return np.arccos(self.cos_theta())

    def angle_to(self, other: Union["Vector", "VectorArray"]) -> NDArray[np.float64]:
        r"""
        Calculate the angles to vector `other`.

        Calculates :math:`\arccos\vec{v}_1 \cdot \vec{v}_2 / (v_1 v_2)`.

        Parameters
        ----------
        other : :class:`.Vector` or :class:`.VectorArray`

        Returns
        -------
        angles : ndarray
            The angles in rad within [0, π].
        """
        return np.arccos(self.dot(other) / self.mag() / other.mag())

    def copy(self) -> "VectorArray":
        """Return a copy of the vector."""
        return VectorArray(self._arr.copy())

    def rotate(self, angle_rad: ArrayLike, axis: VectorLike) -> "VectorArray":
        """
        Rotate vectors by `angle_rad` around `axis`

        Rotates counter-clockwise in a right-handed coordinate system, where the axis
        around which is rotated points towards the obsorver.

        Parameters
        ----------
        angle_rad : float
            The rotation angle(s) in radian.

            If multiple angles are provided, rotate each vector with by the
            corresponding angle.

        axis : :class:`.Vector`
            The axis around which the vector is rotated.

            E.g., ``axis = Vector(0, 0, 1)`` will rotate the vector around the z-axis,
            that is, within the xy-plane.

            So far, `axis` must be along x, y, or z-direction.

        Returns
        -------
        rotated_vector : :class:`.Vector`

        Examples
        --------
        ::

            vec = ap.VectorArray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            vec.rotate(np.pi / 2.0, (0, 0, 1))
            VectorArray([[ 0. 1. 0.]
                         [-1. 0. 0.]
                         [ 0. 0. 1.]])
            vec = ap.VectorArray([[1, 0, 0], [0, 1, 0]])
            vec.rotate((np.pi / 2.0, np.pi), (0, 0, 1))
            VectorArray([[0.  1. 0.]
                         [1. -1. 0.]
                         [0.  0. 1.]])
        """
        axis = asvector(axis).norm(copy=True)
        is_xaxis = axis == (1.0, 0.0, 0.0)
        is_yaxis = axis == (0.0, 1.0, 0.0)
        is_zaxis = axis == (0.0, 0.0, 1.0)
        if not (is_xaxis or is_yaxis or is_zaxis):
            # TODO
            raise NotImplementedError(
                "currently `axis` must be along x, y, or z direction"
            )

        angle_rad = np.asarray(angle_rad)
        if not angle_rad.ndim > 0:
            angle_rad = np.array([angle_rad] * len(self._arr))

        if len(angle_rad) != len(self._arr):
            raise ValueError("mismatching number of angle_rad")

        c = angle_rad if is_xaxis else 0.0  # roll
        b = angle_rad if is_yaxis else 0.0  # pitch
        a = angle_rad if is_zaxis else 0.0  # yaw

        TrigAngle = collections.namedtuple("TrigAngle", ["a", "b", "c"])
        cos = TrigAngle(np.cos(a), np.cos(b), np.cos(c))
        sin = TrigAngle(np.sin(a), np.sin(b), np.sin(c))

        rot_matrix = np.empty((len(angle_rad), 3, 3))
        rot_matrix[:, 0, 0] = cos.a * cos.b
        rot_matrix[:, 0, 1] = cos.a * sin.b * sin.c - sin.a * cos.c
        rot_matrix[:, 0, 2] = cos.a * sin.b * cos.c + sin.a * sin.c
        rot_matrix[:, 1, 0] = sin.a * cos.b
        rot_matrix[:, 1, 1] = sin.a * sin.b * sin.c + cos.a * cos.c
        rot_matrix[:, 1, 2] = sin.a * sin.b * cos.c - cos.a * sin.c
        rot_matrix[:, 2, 0] = -sin.b
        rot_matrix[:, 2, 1] = cos.b * sin.c
        rot_matrix[:, 2, 2] = cos.b * cos.c

        rot_matrix = np.round(rot_matrix, 0)
        new_components = np.matvec(rot_matrix, self._arr)

        return VectorArray(new_components)

    def remove(
        self, condition: NDArray[np.bool], squeeze: bool = True, setval: float = np.nan
    ) -> "VectorArray":
        """
        Remove all vectors ``i`` where ``condition[i] == True``.

        Parameters
        ----------
        condition : ndarray of booleans
            Mask controlling which vectors get removed.

        squeeze : bool, default True.
            If true, completely remove vectors. Otherwise fill affected vectors with
            `setval`.

        setval : float, default ``np.nan``
            If `squeeze` is false, fill removed data with this.

        Returns
        -------
        vectors : :class:`.VectorArray`

        Examples
        --------
        ::

            >>> vecs = ap.VectorArray([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
            >>> vecs.remove(vecs.x == 1)
            VectorArray([[4. 5. 6.]
                         [6. 7. 8.]])
            >>> vecs.remove(vecs.x == 1, squeeze = False)
            VectorArray([[nan nan nan]
                         [ 4.  5.  6.]
                         [ 6.  7.  8.]])
            >>> vecs.remove(vecs.x == 1, squeeze = False, setval=0.0)
            VectorArray([[0. 0. 0.]
                         [4. 5. 6.]
                         [6. 7. 8.]])
        """
        result_x = np.ma.masked_where(condition, self.x, copy=True)
        result_y = np.ma.masked_where(condition, self.y, copy=True)
        result_z = np.ma.masked_where(condition, self.z, copy=True)
        if squeeze:
            result_x = np.ma.compressed(result_x)
            result_y = np.ma.compressed(result_y)
            result_z = np.ma.compressed(result_z)
        else:
            result_x = result_x.filled(setval)
            result_y = result_y.filled(setval)
            result_z = result_z.filled(setval)
        return VectorArray(np.array([result_x, result_y, result_z]).T)

    def keep(
        self, condition: NDArray[np.bool], squeeze: bool = True, setval: float = np.nan
    ) -> "VectorArray":
        """
        Keep all vectors ``i`` where ``condition[i] == True``.

        Parameters
        ----------
        condition : ndarray of booleans
            Mask controlling which vectors are kept.

        squeeze : bool, default True.
            If false, fill all vectors that are not kept with `setval` instead
            of removing them.

        setval : float, default ``np.nan``
            If `squeeze` is false, fill removed data with this.

        Returns
        -------
        vectors : :class:`.VectorArray`

        Examples
        --------
        ::

            >>> vecs = ap.VectorArray([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
            >>> vecs.keep(vecs.x != 1)
            VectorArray([[4. 5. 6.]
                         [6. 7. 8.]])
            >>> vecs.keep(vecs.x != 1, squeeze = False)
            VectorArray([[nan nan nan]
                         [ 4.  5.  6.]
                         [ 6.  7.  8.]])
            >>> vecs.keep(vecs.x != 1, squeeze = False, setval=0.0)
            VectorArray([[0. 0. 0.]
                         [4. 5. 6.]
                         [6. 7. 8.]])
        """
        return self.remove(np.logical_not(condition), squeeze=squeeze, setval=setval)

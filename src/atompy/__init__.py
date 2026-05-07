from importlib.resources import files as _files

import matplotlib.pyplot as _plt

from . import physics
from ._coordinate_system import CoordinateSystem, CoordinateSystemArray
from ._data_xy import DataXY
from ._data_xyz import DataXYZ
from ._histogram1d import Hist1d
from ._histogram2d import Hist2d
from ._utils import (
    centers_to_edges,
    cm_atom,
    cm_atom_from_white,
    convert_cosine_to_angles,
    crop,
    edges_to_centers,
    for_pcolormesh,
    for_pcolormesh_from_root,
    for_pcolormesh_from_txt,
    gauss,
    get_all_dividers,
    sample_distribution,
    sample_distribution_discrete,
    sample_distribution_func,
)
from ._vectors import (
    Vector,
    VectorArray,
    VectorArrayLike,
    VectorLike,
    asvector,
    asvectorarray,
)
from ._version import __version__

__all__ = [
    "CoordinateSystem",
    "CoordinateSystemArray",
    "DataXY",
    "DataXYZ",
    "Hist1d",
    "Hist2d",
    "Vector",
    "VectorArray",
    "VectorArrayLike",
    "VectorLike",
    "__version__",
    "asvector",
    "asvectorarray",
    "centers_to_edges",
    "cm_atom",
    "cm_atom_from_white",
    "convert_cosine_to_angles",
    "crop",
    "edges_to_centers",
    "for_pcolormesh",
    "for_pcolormesh_from_root",
    "for_pcolormesh_from_txt",
    "gauss",
    "get_all_dividers",
    "physics",
    "sample_distribution",
    "sample_distribution_discrete",
    "sample_distribution_func",
]


_style_path = _files("atompy.styles").joinpath("atom.mplstyle")
_plt.style.library["atom"] = str(_style_path)  # type:ignore
if "atom" not in _plt.style.available:
    _plt.style.available.append("atom")

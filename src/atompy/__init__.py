import matplotlib.pyplot as _plt
from importlib.resources import files as _files

_style_path = _files("atompy.styles").joinpath("atom.mplstyle")
_plt.style.library["atom"] = str(_style_path)  # type:ignore
if "atom" not in _plt.style.available:
    _plt.style.available.append("atom")


from .utils import (
    convert_cosine_to_angles,
    centers_to_edges,
    edges_to_centers,
    gauss,
    get_all_dividers,
    crop,
    sample_distribution,
    sample_distribution_discrete,
    sample_distribution_func,
    cm_atom,
    cm_atom_from_white,
    for_pcolormesh,
    for_pcolormesh_from_txt,
    for_pcolormesh_from_root,
)

from . import hist_utils
from .histogram1d import Hist1d
from .histogram2d import Hist2d

from .vectors import (
    asvector,
    asvectorarray,
    Vector,
    VectorArray,
    VectorLike,
    VectorArrayLike,
)

from .coordinate_system import CoordinateSystem, CoordinateSystemArray

from . import physics


from ._version import __version__

__all__ = [
    "convert_cosine_to_angles",
    "centers_to_edges",
    "edges_to_centers",
    "gauss",
    "get_all_dividers",
    "crop",
    "sample_distribution",
    "sample_distribution_discrete",
    "sample_distribution_func",
    "cm_atom",
    "cm_atom_from_white",
    "for_pcolormesh",
    "for_pcolormesh_from_txt",
    "for_pcolormesh_from_root",
    "Hist1d",
    "Hist2d",
    "hist_utils",
    "asvector",
    "asvectorarray",
    "Vector",
    "VectorArray",
    "VectorLike",
    "VectorArrayLike",
    "CoordinateSystem",
    "CoordinateSystemArray",
    "physics",
    "__version__",
]

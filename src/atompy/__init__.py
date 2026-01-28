import matplotlib.pyplot as plt
from importlib.resources import files

_style_path = files("atompy.styles").joinpath("atom.mplstyle")
plt.style.library["atom"] = str(_style_path)  # type:ignore
if "atom" not in plt.style.available:
    plt.style.available.append("atom")


from ._errors import UnmatchingEdgesError

from ._misc import (
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
)

from ._histogram1d import Hist1d

from ._histogram2d import Hist2d

from ._vectors import (
    asvector,
    asvectorarray,
    Vector,
    VectorArray,
    VectorLike,
    VectorArrayLike,
)

from ._coordinate_system import CoordinateSystem, CoordinateSystemArray

from ._version import __version__

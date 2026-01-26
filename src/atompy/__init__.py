import matplotlib.pyplot as plt
from importlib.resources import files

style_path = files("atompy.styles").joinpath("atom.mplstyle")
plt.style.library["atom"] = str(style_path)  # type:ignore
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
)

from ._histogram1d import Hist1d

from ._histogram2d import Hist2d

from ._vectors import (
    asvector,
    asvectorarray,
    Vector,
    VectorArray,
)

from ._version import __version__

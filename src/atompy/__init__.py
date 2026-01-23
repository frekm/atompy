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

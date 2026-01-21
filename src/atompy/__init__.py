from ._errors import AliasError, UnmatchingEdgesError

from ._misc import (
    convert_cosine_to_angles,
    centers_to_edges,
    edges_to_centers,
    gauss,
    crop,
    sample_distribution,
    sample_distribution_discrete,
    sample_distribution_func,
)


from . import physics
from ._version import __version__

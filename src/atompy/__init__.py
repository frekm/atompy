from ._io import (
    save_1d_as_txt,
    save_2d_as_txt,
    load_1d_from_txt,
    load_2d_from_txt,
    load_1d_from_root,
    load_2d_from_root,
    for_pcolormesh,
    for_imshow,
)

from ._miscellaneous import (
    ImshowData,
    PcolormeshData,
    get_all_dividers,
    crop,
    convert_cosine_to_angles,
    integral_sum,
    integral_polyfit,
    sample_distribution,
    edges_to_centers,
    centers_to_edges,
    work_out_bin_edges,
)

from ._histogram import Hist1d, Hist2d

from ._vector import Vector

from ._errors import(
    AliasError,
    NonconstantBinsizeError,
    UnderdeterminedBinsizeError,
    FigureWidthTooLargeError,
)

from ._plotting import (
    Edges,
    PTS_PER_INCH,
    MM_PER_INCH,
    colors,
    cm_lmf2root,
    cm_lmf2root_from_white,
    cmap_from_x_to_y,
    textwithbox,
    dotted,
    dash_dotted,
    dashed,
    dashed,
    add_colorbar,
    clear_colorbars,
    add_abc,
    update_colorbars,
    get_renderer,
    set_axes_size,
    get_sorted_axes_grid,
    get_column_pads_inches,
    set_min_column_pads,
    get_row_pads_inches,
    set_min_row_pads,
    get_figure_margins_inches,
    get_axes_position_inch,
    get_axes_tightbbox_inch,
    make_me_nice,
    align_axes_vertically,
    align_axes_horizontally,
    get_axes_margins_inches,
    square_polar_axes,
)

from . import physics

from ._version import __version__

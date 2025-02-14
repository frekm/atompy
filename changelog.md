- Changed default file format of `_set_theme_atompy` to PDF

# v4.17.2
- Expanded documentation of color palettes
- Changed order of colors in `PALETTE_OKABE_ITO`
- Fix x/ytick color mismatch in `_set_theme_atompy`
- Modified `_set_theme_atompy`

# v4.17.1
- Changed order of colors in `PALETTE_OKABE_ITO_ACCENT`

# v4.17.0
- Improved `set_color_cycle`
  - It now updates the color_cycle even after an axes on-the-fly
- Added multiple color palettes (access via `ap.PALETTE_*`)

# v4.16.0
- Added `set_color_cycle`, `_set_theme_atompy`, and `set_latex_backend`
- Bugfixes

# v4.15.0
- Added `create_1d_plot` and `create_2d_plot`
- Added `margin_pad_ignores_labels` keyword to `make_me_nice`
- Added more examples to documetation
- Updated documentation

# v4.14.0
- Added `col/row_pad_ignores_labels` keyword to `make_me_nice`
- Documentation improvements

# v4.13.0
- Added `CoordinateSystem`
- Added `PcolormeshData.z` property (alias for `PcolormeshData.c`)

# v4.12.2
- Replaced `np.bool` with `bool` in type-hinting

# v4.12.1
- Fix hyperlinks in documentation

# v4.12.0
- Added operator overloading for `Hist1d` and `Hist2d`

# v4.11.0
- Deprecated `fit_polar`, `eval_polarfit`, `eval_polarfit_even`.
- Added `fit_yl0_polynomial` and `eval_yl0_polynomial`.
- Fixed `add_polar_guideline`.

# v4.10.0
- Added `fit_polar`, `eval_polarfit`, `eval_polarfit_even`, 
  and `add_polar_guideline`.

# v4.9.1
- Fixed hyperlink on documentation index page.

# v4.9.0
- Added `sample_distribution_func` and `sample_distribution_discrete`
- Added tutorial `Set up Visual Studio Code for Python`

# v4.8.0
- Fixed docstring of `for_pcolormesh`
- Added `SingleVector`

# v4.7.0
- Added `mom_final_distr_photon_var`

# v4.6.1
- Fixed bugged keyword `row_pad_pts` of `make_me_nice()`
- Added more examples

# v4.6.0
- Added `Hist1d.without_range`, `Hist2d.without_xrange`, `Hist2d.without_yrange`
- Added "Colormaps and Colorbars" tutorial
- Added "Resizing Axes" tutorial
- Fixed some docstrings

# v4.5.0
- Added "Sample a random distribution" Tutorial
- Added `edges_to_centers` and `centers_to_edges`.
- Fixed `sample_distribution`

# v4.4.0
- Added `cmap_from_x_to_y`

# v4.3.0
- Added `square_polar_axes`
- Fixed `set_axes_size` not working with polar axes

# v4.2.1
- Added explicit properties for `Hist2d.H`, `Hist2d.xedges`, `Hist2d.yedges`,
  `Hist1d.histogram` and `Hist1d.edges`

# v4.2.0
- Added "Examples & Tutorials" to documentation
- Added "atom" colormap (same as existing "lmf2root" colormap)
- Code refactoring

# v4.1.0

# v4.0.1
- Replaced `np.float_` with `np.float64`
- Fixed missing import of `physics` submodule

# v4.0.0
## Changes to plotting
- Removed `atompy.subplots` as it became obsolete. No need to create axes
  with it anymore to make all the other functions of `atompy` work.
- Added `atompy.make_me_nice`
- Reworked `atompy.add_colorbar`.
- Added lots of functions to manipulate matplotlib axes and figures. See
  Documentation->Plotting for an exhaustive list.

## Changes for data loading
- See documentation->Input/Output for more information.

## Miscellaneous
- General improvements to documentation

# v3.0.7
- Added scipy to requirements list
- Added atompy.physics.coltrims.ion_tof_calibration

# v3.0.6
- New layout implemented with `sphinx.ext.autosummary`
- Changed HTML theme back to `pydata-sphinx-theme` (it works better with 
  `autosummary`)

# v3.0.5
- Updated documentation

# v3.0.4
- Changed HTML theme to Furo

# v3.0.3
- Added `unroll` keyword for `atompy.subplots`

# v3.0.2
- Fixed internal import in _io.py

# v3.0.1
- Fixed internal imports

# v3.0.0
- Updated documentation
  - Expanded documentation
  - Restructured page layout
- Added `Hist1d.for_step` and `Hist1d.for_plot` methods
- Moved physics related stuff into separate submodules `atompy.physics` and
  `atompy.physics.compton_scattering`
- Added `PcolormeshData` class
- renamed `atompy.Vector.nparray` to `atompy.Vector.ndarray`.

# v2.1.0
- Updated documentation
- Added warning when `make_margins_tight` with `fixed_figwidth=True` is called
  after `change_ratio` was called
- `Hist2d.for_imshow` now returns a `ImshowData` object.
- Fixed and expanded `profile_`-methods for `Hist2d`


# v2.0.0
- Changed return value when loading multiple histos at once
- Changed that the _histXd functions return Hist1d and Hist2d instances

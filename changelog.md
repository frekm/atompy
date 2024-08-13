# v4.4.0
- Added `cmap_from_x_to_x`

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
- Add scipy to requirements list
- Add atompy.physics.coltrims.ion_tof_calibration

# v3.0.6
- New layout implemented with `sphinx.ext.autosummary`
- Change HTML theme back to `pydata-sphinx-theme` (it works better with 
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
- changed return value when loading multiple histos at once
- changed that the _histXd functions return Hist1d and Hist2d instances

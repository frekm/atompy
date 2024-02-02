# v2.1.0
- Updated documentation
- Added warning when `make_margins_tight` with `fixed_figwidth=True` is called
  after `change_ratio` was called
- `Hist2d.for_imshow` now returns a `ImshowData` object.
- Fixed and expanded `profile_`-methods for `Hist2d`


# v2.0.0
- changed return value when loading multiple histos at once
- changed that the _histXd functions return Hist1d and Hist2d instances

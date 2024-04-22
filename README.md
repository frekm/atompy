# ATOMPY Python Package

A collection of utility functions to create plots and do analysis.

## Usage

If you don't want to install the package, you can simply download 
`atompy.zip` from the latest
release (see [Release Page](https://github.com/crono-kircher/atompy/releases))
and unpack it into your working directory.

Then you can import just as you import any other python module.

If you use this method, you'll need to install the dependencies manually,
most notably `numpy`, `matplotlib` and `uproot`.

## Installation
Installation is not absolutely necessary, but you do you.

Optionally, create and activate a virtual environment in which to install
the module
```shell
python -m venv .venv
```

Make sure to activate the virtual environment in your terminal.

If you have git installed, You can install `atompy` using git

```shell
pip install git+https://github.com/crono-kircher/atompy
```

Alternatively, you can install from the source code.

Go to the [Release](https://github.com/crono-kircher/atompy/releases)
page and download the latest release Source Code (`zip` or `tar.gz`, not the
`atompy.zip`). Unpack the release to `<path>`, then run

```shell
pip install <path>/atompy-<version>/src
```

## Documentation
The documentaion is avaiable online at
[ReadTheDocs](https://atomicphysics-atompy.readthedocs.io/en/latest/).

## Structure
```
.
└─ atompy/
   ├── src/
   │ └── atompy/                    (atompy module)
   │   ├── physics/                 (atompy.physics submodule)
   │   │ ├── _physics.py            (general physics stuff)
   │   │ └── comptons_scattering.py (physics related to Compton scattering)
   │   ├── _histogram.py            (Hist1d and Hist2d classes)
   │   ├── _io.py                   (loading/saving data)
   │   ├── _miscellaneous.py        
   │   ├── _plotting.py             (related to plotting)
   │   └── _vector.py               (Vector class)
   └── docs/
     └── source/             (documentation source files)
``` 


## License
[atompy](https://github.com/crono-kircher/atompy) by
[Max Kircher](https://github.com/crono-kircher) is licensed under
[CC BY-NC 4.0!](http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1)

# ATOMPY Python Package

A collection of utility functions to create plots and do analysis.

## Installation

Optionally, create and activate a virtual environment in which to install
the module
```shell
python -m venv .venv
```
Activate it by running the activate script in .venv\Scripts\

There are multiple ways to install/use atompy:

### Using git

If you have Git installed, you can install the atompy module by running
```shell
pip install git+https://github.com/crono-kircher/atompy
```

### Install from local tree

Go to the [Release](https://github.com/crono-kircher/atompy/releases)
page and download the latest release. Unpack the release to `<path>`,
then run

```shell
pip install <path>/atompy-<version>/src
```

### Copy atompy-folder to working directory

The quick and dirty way is to simply drop the atompy folder in your 
working directory. You can download `atompy.zip` from the
[Release](https://github.com/crono-kircher/atompy/releases) page.

This does not install the dependencies. You have to install those manually.
They are listed in `pyproject.toml`.

## Documentation
The documentaion is avaiable online at
[readthedocs](https://atomicphysics-atompy.readthedocs.io/en/latest/).

## Structure
```
.
└─ atompy/
   ├── src/
   │ └── atompy/                      (atompy module)
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
        └── _examples/       (python example scripts for docs)

``` 


## License
[atompy](https://github.com/crono-kircher/atompy) by
[Max Kircher](https://github.com/crono-kircher) is licensed under
[CC BY-NC 4.0!](http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1)

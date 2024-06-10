# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
from atompy import __version__

from intersphinx_registry import get_intersphinx_mapping

sys.path.append(os.path.abspath('../../src/'))


project = 'atompy'
copyright = 'Max Kircher, CC BY-NC 4.0'
author = 'Max Kircher'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    'matplotlib.sphinxext.plot_directive',
]
autodoc_typehints = "none"

pygments_style = "sphinx"
python_display_short_literal_types = True

numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {}
numpydoc_xref_ignore = "all"

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
    "show_toc_level": 4,
    "collapse_navigation": False,
    "navigation_with_keys": False,
    "show_nav_level": 4,
}

intersphinx_mapping = get_intersphinx_mapping(
    packages=["python", "numpy", "matplotlib"]) # type: ignore

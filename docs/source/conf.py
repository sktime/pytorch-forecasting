# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
from pathlib import Path
import shutil
import sys

from recommonmark.parser import CommonMarkParser

SOURCE_PATH = Path(os.path.dirname(__file__))  # noqa # docs source
PROJECT_PATH = SOURCE_PATH.joinpath("../..")  # noqa # project root

sys.path.insert(0, str(PROJECT_PATH))  # noqa

import pytorch_forecasting  # isort:skip


# -- Project information -----------------------------------------------------

project = "pytorch-forecasting"
copyright = "2020, Jan Beitner"
author = "Jan Beitner"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
]

source_parsers = {
    ".md": CommonMarkParser,
}

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**/.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# setup configuration
def skip(app, what, name, obj, skip, options):
    """
    Document __init__ methods
    """
    if name == "__init__":
        return True
    return skip


apidoc_output_folder = SOURCE_PATH.joinpath("api")

PACKAGES = [pytorch_forecasting.__name__]


def setup(app):
    app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", skip)


# extension configuration
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"

# theme options
html_theme_options = {
    "github_url": "https://github.com/jdb78/pytorch-forecasting",
}

html_theme_options = {"search_bar_position": "navbar"}

html_sidebars = {
    "index": [],
    "getting_started": [],
    "data": [],
    "models": [],
    "metrics": [],
    "faq": [],
    "contribute": [],
    "CHANGELOG": [],
}


autodoc_member_order = "groupwise"
autoclass_content = "both"

# autosummary
autosummary_generate = True
shutil.rmtree(SOURCE_PATH.joinpath("api"), ignore_errors=True)

# copy changelog
shutil.copy(
    "../../CHANGELOG.md",
    "CHANGELOG.md",
)

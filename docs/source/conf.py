import os
import sys

sys.path.insert(0, os.path.abspath("../stable_worldmodel"))

from stable_worldmodel.__about__ import (
    __version__,
)  # Import the version from __about__.py


project = "stable-worldmodel"
copyright = "2025, Randall Balestriero, Dan Haramati, Lucas Maes"
author = "Randall Balestriero, Dan Haramati, Lucas Maes"

# The full version, including alpha/beta/rc tags
release = __version__  # Set release to the version from __about__.py


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

pygments_style = "sphinx"  # enable syntax highlighting

autosummary_generate = True
copybutton_exclude = ".linenos, .gp"

# Napoleon settings for Google/NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/"],
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/demo_",
    "run_stale_examples": True,
    "ignore_pattern": r"__init__\.py",
    "reference_url": {
        # The module you locally document uses None
        "sphinx_gallery": None
    },
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("stable_pretraining",),
    # objects to exclude from implicit backreferences. The default option
    # is an empty set, i.e. exclude nothing.
    "exclude_implicit_doc": {},
    "nested_sections": False,
}

# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {
    "tex": {"equationNumbers": {"autoNumber": "AMS", "useLabelIds": True}}
}

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

math_numfig = True
numfig = True
numfig_secnum_depth = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = []

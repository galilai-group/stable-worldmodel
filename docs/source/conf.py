import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

from datetime import date

from sphinxawesome_theme.postprocess import Icons

from stable_worldmodel import __version__


# -- Project information ---

project = "stable-worldmodel"
copyright = "2025, Randall Balestriero, Lucas Maes, Dan Haramati"
author = "Randall Balestriero, Lucas Maes, Dan Haramati"
release = __version__
current_year = date.today().year
copyright = f"{current_year}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx.ext.todo",
    "myst_parser",
]

todo_include_todos = False

autosummary_generate = True
autosummary_filename_map = {}
add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_ivar = True

copybutton_exclude = ".linenos, .gp"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {"tex": {"equationNumbers": {"autoNumber": "AMS", "useLabelIds": True}}}


math_numfig = True
numfig = True
numfig_secnum_depth = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# pygments_style = "tango"
# pygments_dark_style = "monokai"

pygments_style = "tango"
pygments_style_dark = "github-dark"
html_permalinks_icon = Icons.permalinks_icon
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]

html_show_sourcelink = False
html_copy_source = False
html_title = "stable-worldmodel"
html_theme = "sphinxawesome_theme"
html_favicon = "_static/img/favicon.ico"

html_theme_options = {
    "logo_light": "_static/img/logo-light.svg",
    "logo_dark": "_static/img/logo-dark.svg",
}

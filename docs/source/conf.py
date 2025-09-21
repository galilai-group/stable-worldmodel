import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from datetime import date
from stable_worldmodel.__about__ import (
    __version__,
)  # Import the version from __about__.py


project = "stable-worldmodel"
copyright = "2025, Randall Balestriero, Dan Haramati, Lucas Maes"
author = "Randall Balestriero, Dan Haramati, Lucas Maes"

# The full version, including alpha/beta/rc tags
release = __version__  # Set release to the version from __about__.py

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
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    #"sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

autosummary_generate = True
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

# sphinx_gallery_conf = {
#     "examples_dirs": ["../../examples/"],
#     "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
#     "filename_pattern": "/demo_",
#     "run_stale_examples": True,
#     "ignore_pattern": r"__init__\.py",
#     "reference_url": {
#         # The module you locally document uses None
#         "sphinx_gallery": None
#     },
#     # directory where function/class granular galleries are stored
#     "backreferences_dir": "gen_modules/backreferences",
#     # Modules for which function/class level galleries are created. In
#     # this case sphinx_gallery and numpy in a tuple of strings.
#     "doc_module": ("stable_pretraining",),
#     # objects to exclude from implicit backreferences. The default option
#     # is an empty set, i.e. exclude nothing.
#     "exclude_implicit_doc": {},
#     "nested_sections": False,
# }

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

pygments_style = "tango"
pygments_dark_style = "monokai"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
html_show_sourcelink = False
html_copy_source = False
html_title = "stable-worldmodel\nDocumentation"
html_theme = "furo"
html_favicon = "_static/img/favicon.ico"
html_theme_options = {
    "light_logo": "img/logo-light.svg",
    "dark_logo": "img/logo-dark.svg",
    "source_repository": "https://github.com/randall-lab/stable-worldmodel/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["edit"],

    # TODO change for dark them as well
    "light_css_variables": {
        "color-brand-primary": "#202123",
        "color-brand-content": "#296cde",
        "color-inline-code-background": "#f8f8f8",
        "color-brand-visited": "#2757dd"
    },


    "announcement": "alpha-pre-release version - API may change",
    "footer_icons": [
        {
            "name": "Discord",
            "url": "https://discord.gg/8M6hT39X",
            "html": "",
            "class": "fa-brands fa-solid fa-discord",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/randall-lab/stable-worldmodel",
            "html": "",
            "class": "fa-brands fa-solid fa-github",
        },
    ],
    
}

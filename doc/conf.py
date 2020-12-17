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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinx_ext"))

# TODO: perhaps replace with https://github.com/westurner/sphinxcontrib-srclinks
from github_link import make_linkcode_resolve


# -- Project information -----------------------------------------------------

project = 'chop'
copyright = '2020, chop developers'
author = 'chop developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx_gallery.gen_gallery",
]


sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    "doc_module": "chop",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "backreferences_dir": os.path.join("modules", "generated"),
    "show_memory": True,
    "reference_url": {"chop": None},
}


mathjax_config = {
    "TeX": {
        "Macros": {
            "argmin": "\\DeclareMathOperator*{\\argmin}{\\mathbf{arg\\,min}}",
            "argmax": "\\DeclareMathOperator*{\\argmin}{\\mathbf{arg\\,max}}",
            "bs": "\\newcommand{\\bs}[1]{\\boldsymbol{#1}}",
        },
    },
    "tex2jax" : {
        "inlineMath": [['$', '$'], ['\(', '\)']],
    }
}

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "copt",
    u"https://github.com/openopt/" "copt/blob/{revision}/" "{package}/{path}#L{lineno}",
)


autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
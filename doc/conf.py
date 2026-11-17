# SPDX-License-Identifier: AGPL-3.0-or-later
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path

project = 'scikit-elasticity'
copyright = '2025-2026, Martin Diehl'
author = 'Martin Diehl'

try:
    release = (Path(__file__).resolve().parents[1] / 'VERSION').read_text().strip()
except FileNotFoundError:
    from importlib.metadata import version as _version
    release = _version('scikit-elasticity')
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode'
             ]

mathjax3_config = {
    'loader': {'load': ['[tex]/physics']},
    'tex': {'packages': {'[+]': ['physics']}},
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_logo = '_static/logo.svg'
html_permalinks_icon = '<span>#</span>'
html_static_path = ['_static']

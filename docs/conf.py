# Hybrid asynchronous solver documentation build configuration file
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
]

autosummary_generate = True

source_suffix = ['.rst', '.md']

master_doc = 'index'

from hybrid import package_info
project = package_info.__title__
copyright = package_info.__copyright__
author = package_info.__author__
version = package_info.__version__
release = package_info.__version__

language = 'en'


add_module_names = False

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'sdk_index.rst']

linkcheck_retries = 2
linkcheck_anchors = False
linkcheck_ignore = [r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    ]

pygments_style = 'sphinx'

todo_include_todos = True

modindex_common_prefix = ['hybrid.']

doctest_global_setup = """
from hybrid.samplers import *
from hybrid.core import *
from hybrid.utils import *
from hybrid.decomposers import *
from hybrid.composers import *
from hybrid.flow import *
"""

# -- Options for HTML output ----------------------------------------------


html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

# TODO: replace oceandocs & sysdocs_gettingstarted
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'oceandocs': ('https://docs.ocean.dwavesys.com/en/latest/', None),
    'sysdocs_gettingstarted': ('https://docs.dwavesys.com/docs/latest/', None),
}

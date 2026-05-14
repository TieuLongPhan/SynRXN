"""Sphinx configuration for the SynRXN documentation."""

from __future__ import annotations

import os
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version as get_version
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# The documentation folder usually lives directly under the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------
project = "SynRXN"
author = "Tieu-Long Phan"
copyright = "2026, Tieu-Long Phan and SynRXN contributors"

try:
    release = get_version("synrxn")
except PackageNotFoundError:
    try:
        with (ROOT / "pyproject.toml").open("rb") as fh:
            release = tomllib.load(fh)["project"]["version"]
    except Exception:
        release = os.environ.get("SYNRXN_DOC_VERSION", "1.0.0")

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

autosummary_generate = True
autosectionlabel_prefix_document = True
bibtex_bibfiles = ["refs.bib"]
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d+\]: | {2,5}\.\.\. "
copybutton_prompt_is_regexp = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = False
suppress_warnings = ["autosectionlabel.*"]

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"SynRXN {release} documentation"
html_short_title = "SynRXN"
html_logo = "_static/synrxn-logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["synrxn-navbar"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    # Keep the left navigation dedicated to the Sphinx toctree.
    # Without this, some PyData versions append extra items below the nav.
    "primary_sidebar_end": [],
    "github_url": "https://github.com/TieuLongPhan/SynRXN",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TieuLongPhan/SynRXN",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/synrxn/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Zenodo",
            "url": "https://doi.org/10.5281/zenodo.17297258",
            "icon": "fa-solid fa-box-archive",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "TieuLongPhan",
    "github_repo": "SynRXN",
    "github_version": "main",
    "doc_path": "doc",
}


html_sidebars = {
    "**": [
        # "search-field.html",
        "synrxn-sidebar-section.html",
    ],
}

# -- Pygments ----------------------------------------------------------------
pygments_style = "sphinx"
pygments_dark_style = "monokai"

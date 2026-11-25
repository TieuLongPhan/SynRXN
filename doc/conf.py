"""Sphinx configuration for the SynRXN documentation."""

from __future__ import annotations

import json
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
    with (ROOT / "pyproject.toml").open("rb") as fh:
        release = tomllib.load(fh)["project"]["version"]
except (FileNotFoundError, KeyError):
    try:
        release = get_version("synrxn")
    except PackageNotFoundError:
        release = "1.5.0"
release = os.environ.get("SYNRXN_DOC_VERSION", release)

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "synrxn.reconstruction.presentation.documentation",
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

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "SynRXN — reaction informatics data"
html_short_title = "SynRXN"
html_logo = "_static/synrxn-logo.svg"
html_favicon = "_static/synrxn-icon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["reaction-viewer.js", "catalog.js"]

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
    "families": [],
    "**": [
        # "search-field.html",
        "synrxn-sidebar-section.html",
    ],
}

# -- Pygments ----------------------------------------------------------------
pygments_style = "sphinx"
pygments_dark_style = "monokai"


def _catalogue_labels(app, docname, source):
    """Use plain wording in the generated family catalogue."""
    if docname == "families":
        for before, after in (
            (">Methods and provenance<", ">Methods and sources<"),
            (">Dataset and provenance<", ">Table<"),
            ("repair reaction records with provenance", "repair reaction records and retain source references"),
            ("Reaxys provenance alone", "a Reaxys source alone"),
            ("retained only in native provenance", "retained only in the original source files"),
        ):
            source[0] = source[0].replace(before, after)


def _task_definitions(app, docname, source):
    """Generate the task guide from the main release catalogue."""
    if docname != "task_definitions":
        return
    from synrxn.reconstruction.presentation.documentation import HIERARCHY, REGISTRY, RELEASE, release_report

    for path in (ROOT / HIERARCHY, ROOT / REGISTRY, ROOT / RELEASE, Path(__file__)):
        app.env.note_dependency(str(path))
    registry = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    definitions = {task["id"]: task["question"] for task in registry["tasks"]}
    release = json.loads((ROOT / RELEASE).read_text(encoding="utf-8"))
    admitted = {(table["namespace"], table["key"]) for table in release["release_tables"]}
    lines = []
    for family in release_report(ROOT)["families"]:
        title = family["title"]
        lines.extend([title, "-" * len(title), "", ".. list-table::", "   :header-rows: 1",
                      "   :widths: 27 29 31 13", "   :class: synrxn-task-counts", "",
                      "   * - Task", "     - Input", "     - Expected result", "     - Stored records"])
        for task in family["tasks"]:
            inputs, targets = definitions[task["id"]].split(" -> ", 1)
            link = f'`{task["title"]} <families.html#task-{task["id"]}>`_'
            records = sum(
                table["rows"] for table in task["tables"]
                if table["role"] == "primary" and (table["namespace"], table["key"]) in admitted
            )
            lines.extend([f"   * - {link}", f"     - {inputs}", f"     - {targets}", f"     - {records:,}"])
        lines.append("")
    source[0] = source[0].replace(".. synrxn-task-definitions", "\n".join(lines))


def _write_data_records(app, exception):
    """Combine the two reviewed source catalogues for the record browser."""
    if exception is not None or app.builder.format != "html":
        return
    static = Path(app.outdir) / "_static"
    main, sharealike = (
        json.loads((static / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("catalog-data", "catalog-sharealike-data")
    )
    records = sorted(main["datasets"] + sharealike["datasets"], key=lambda item: item["id"])
    release = {**main["release"], "dataset_count": len(records)}
    release.pop("partition", None)
    payload = json.dumps({"release": release, "datasets": records}, ensure_ascii=False)
    (static / "data-records.js").write_text("window.SYNRXN_CATALOG = " + payload + ";\n", encoding="utf-8")


def setup(app):
    """Prepare the catalogue and task documentation."""
    app.connect("source-read", _catalogue_labels, priority=600)
    app.connect("source-read", _task_definitions)
    app.connect("build-finished", _write_data_records, priority=600)

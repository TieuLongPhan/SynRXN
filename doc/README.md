# SynRXN documentation source

This folder contains the enhanced Sphinx documentation for SynRXN. The visual layout follows the PyData Sphinx style used by SynKit and keeps the fast, card-based navigation pattern used by SynEdu.

## Build locally

```bash
cd doc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

Open `_build/html/index.html` in a browser.

## Main pages

- `index.rst` — landing page, project summary, quick links, citation.
- `getting_started.rst` — installation, smoke test, first dataset load, splits, troubleshooting.
- `data_concept.rst` — task model, storage layout, schema conventions, reproducibility guidance.
- `data_records.rst` — dataset inventory by task family.
- `tutorials_and_examples.rst` — practical workflows for Zenodo, GitHub releases, commits, splits, and rebuilds.
- `api.rst` — generated API references for public entry points.
- `reference.rst` — citation and bibliography.


## Project navigation updates

This documentation package includes dedicated pages for:

- What's New (`whats_new.rst`) for changelog/release notes.
- Paper (`paper.rst`) for the Scientific Data citation and DOI.
- Issues (`issues.rst`) for reporting dataset/API/documentation problems.

These pages are included in the left sidebar through the root `index.rst` toctree.

## Build note

The Zenodo citation entry is stored as `@misc` in `refs.bib` for compatibility with
`sphinxcontrib-bibtex` and Pybtex's standard `unsrt` formatter. The visible
reference page explains that this is the version-specific Zenodo software/data
release to cite.

## Navigation note

The main documentation tree is configured as **Section Navigation** in `index.rst`. Core project pages such as What's New, API Reference, Paper, Issues, and References are included in the PyData left sidebar rather than the top navbar.

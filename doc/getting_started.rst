.. _getting_started:

.. container:: badges

   .. image:: https://img.shields.io/pypi/v/synrxn.svg
      :alt: PyPI version
      :target: https://pypi.org/project/synrxn/
      :align: left

   .. image:: https://zenodo.org/badge/1062420507.svg
      :alt: Zenodo DOI
      :target: https://doi.org/10.5281/zenodo.17297258

   .. image:: https://github.com/TieuLongPhan/SynRXN/actions/workflows/publish-package.yml/badge.svg
      :alt: CI status
      :target: https://github.com/TieuLongPhan/SynRXN/actions/workflows/publish-package.yml
      :align: left

   .. raw:: html

      <div style="clear: both;"></div>

===============================
Getting Started with SynRXN
===============================

Welcome to the **SynRXN** documentation.

**SynRXN** is a curated, provenance-tracked collection of reaction datasets and
evaluation manifests designed for **reproducible benchmarking** of
reaction-informatics tasks, including:

- reaction balancing and rebalancing
- atom–atom mapping (AAM)
- reaction classification and template prediction
- reaction property prediction (e.g. thermodynamics, kinetics)
- synthesis and retrosynthesis benchmarking

Each dataset is distributed with a machine-readable **manifest** (version,
checksums) so that experiments can be
reproduced exactly across machines, time, and library versions.

If you are new to **SynRXN**, this page shows how to install the package and run
a minimal sanity check.

Installation
------------

Python requirements
~~~~~~~~~~~~~~~~~~~
SynRXN currently targets **Python 3.11 or later**.

You can download Python from: https://www.python.org/downloads/

Create a virtual environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using an isolated environment avoids dependency conflicts with other projects.

Using ``venv``:

.. code-block:: bash

   python -m venv synrxn-env
   source synrxn-env/bin/activate

Using Conda:

.. code-block:: bash

   conda create --name synrxn-env python=3.11
   conda activate synrxn-env

Install from PyPI
~~~~~~~~~~~~~~~~~
Install the core package:

.. code-block:: bash

   pip install synrxn

For the full set of optional dependencies (e.g. plotting, heavier ML stacks),
use:

.. code-block:: bash

   pip install "synrxn[all]"

Developer / editable install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are developing SynRXN or contributing back to the repository:

.. code-block:: bash

   git clone https://github.com/TieuLongPhan/SynRXN.git
   cd SynRXN

   python -m venv .venv
   source .venv/bin/activate

   pip install -e ".[dev]"

This installs SynRXN in editable mode so changes in the source tree are
immediately reflected when you import the package.

Quick sanity check
------------------

Check the version
~~~~~~~~~~~~~~~~~
After installation, verify that Python sees the package:

.. code-block:: bash

   python -c "import importlib.metadata as m; print(m.version('synrxn'))"

You should see a version string such as ``0.0.7``.

List available datasets
~~~~~~~~~~~~~~~~~~~~~~~
The :class:`synrxn.data.DataLoader` class provides a high-level entry point to
the curated datasets.

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

   dl = DataLoader(
       task="classification",                      # or "property"
       source="zenodo",                            # use Zenodo software deposit
       version="0.0.6",                            # SynRXN data release
       cache_dir=Path("~/.cache/synrxn").expanduser(),
   )

   print("Available datasets:", dl.available_names())

If installation and network access are working, this prints a list of dataset
names such as ``["schneider_b", ...]``.

Load a small dataset
~~~~~~~~~~~~~~~~~~~~
As a further check, load one dataset and inspect its columns:

.. code-block:: python

   df = dl.load("schneider_b")
   print("Rows:", len(df))
   print("Columns:", df.columns.tolist())

At this point you have a working **SynRXN** setup and a Pandas ``DataFrame`` in
memory. You can now:

- explore dataset statistics,
- connect it to your own models, and
- reuse SynRXN’s manifests and split definitions for reproducible benchmarking.

Where does the data come from?
------------------------------

SynRXN datasets are distributed via:

- **Zenodo** archives associated with the SynRXN software record (versioned snapshots),
- **GitHub** releases and commits of the repository, and
- local **cache directories** (to avoid repeated downloads).

The :class:`synrxn.data.DataLoader` abstracts these details; you specify:

- ``task``: high-level task type, e.g. ``"classification"`` or ``"property"``;
- ``source``: ``"zenodo"``, ``"github"``, ``"commit"``, or ``"github"+"latest"``;
- ``version``: a release version (e.g. ``"0.0.6"``), a Git tag (e.g. ``"v0.0.6"``),
  or a commit SHA;
- ``cache_dir``: local directory used to store downloaded artifacts.

For more advanced usage (switching between sources, pinning commits, repeated
k-fold splitting), see :doc:`Tutorials and Examples <tutorials_and_examples>`.

Rebuilding datasets from source
-------------------------------

Most users can rely on the published datasets and manifests distributed via
Zenodo and GitHub and accessed through :class:`synrxn.data.DataLoader`.

For full end-to-end reproducibility, **SynRXN** also exposes a command-line
interface that can rebuild dataset artifacts from the original sources.

From a checked-out SynRXN repository (with a developer/editable install):

.. code-block:: bash

   cd SynRXN

   # Show available commands and options
   python -m synrxn --help

   # Example: rebuild property-prediction datasets and their manifests
   python -m synrxn build --property

This rebuild interface is primarily intended for:

- validating that Zenodo/GitHub archives are reproducible from the same scripts,
- regenerating datasets after making changes to the build pipeline, and
- advanced users who want full control over the data-generation process.

When you rebuild datasets locally, treat the combination of:

- SynRXN version (and Git commit),
- the exact ``python -m synrxn build ...`` command, and
- the generated manifest

as part of your experiment provenance.
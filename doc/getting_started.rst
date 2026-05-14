.. _getting_started:

Getting Started
===============

This page takes you from installation to a working SynRXN data-loading workflow.
The examples use released data for reproducibility and exact Git commits for
development snapshots.

.. raw:: html

   <div class="synrxn-steps">
     <div class="synrxn-step"><span class="synrxn-step-icon"><i class="fa-solid fa-download" aria-hidden="true"></i></span><strong>Install</strong><span>Create a Python environment and install SynRXN from PyPI or source.</span></div>
     <div class="synrxn-step"><span class="synrxn-step-icon"><i class="fa-solid fa-database" aria-hidden="true"></i></span><strong>Load</strong><span>Use <code>DataLoader</code> to resolve, cache, and read a curated benchmark table.</span></div>
     <div class="synrxn-step"><span class="synrxn-step-icon"><i class="fa-solid fa-shuffle" aria-hidden="true"></i></span><strong>Split</strong><span>Reuse published split columns or generate deterministic repeated k-fold splits.</span></div>
   </div>

Requirements
------------

- **Python**: 3.11 or newer is recommended.
- **Operating system**: Linux, macOS, or Windows with WSL.
- **Network access**: required the first time you download data from Zenodo or
  GitHub.
- **Persistent cache**: recommended for repeated experiments, for example
  ``~/.cache/synrxn``.

Installation
------------

From PyPI
~~~~~~~~~

Use this route for released package builds and published data snapshots.

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install synrxn

Install the broader optional dependency stack when you need all extras:

.. code-block:: bash

   python -m pip install "synrxn[all]"

From source
~~~~~~~~~~~

Use this route when you are developing SynRXN, editing documentation, or
rebuilding curated datasets.

.. code-block:: bash

   git clone https://github.com/TieuLongPhan/SynRXN.git
   cd SynRXN

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

Build the documentation locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The enhanced documentation uses the PyData Sphinx theme.

.. code-block:: bash

   cd doc
   python -m pip install -r requirements.txt
   sphinx-build -b html . _build/html

Open ``doc/_build/html/index.html`` in a browser.

Verify the installation
-----------------------

Run a minimal import and version check:

.. code-block:: bash

   python - <<'PY'
   import importlib.metadata as metadata
   import synrxn

   print("synrxn", metadata.version("synrxn"))
   print("module", synrxn.__file__)
   PY

Load your first dataset
-----------------------

The example below loads the balanced Schneider classification benchmark from a
versioned Zenodo release and inspects the resulting pandas DataFrame.

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

   cache_dir = Path("~/.cache/synrxn").expanduser()

   loader = DataLoader(
       task="classification",
       source="zenodo",
       version="1.0.0",
       cache_dir=cache_dir,
   )

   print("Available datasets:", loader.available_names())

   df = loader.load("schneider_b")
   print("Shape:", df.shape)
   print("Columns:", df.columns.tolist())
   print(df.head(3))

Use a pinned commit during development
--------------------------------------

For development workflows, pinning a commit gives an exact source snapshot. This
is stronger than ``latest`` because it prevents silent changes in future runs.

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

   loader = DataLoader(
       task="property",
       source="commit",
       version="3e1612e2199e8b0e369fce3ed9aff3dda68e4c32",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )

   df = loader.load("b97xd3")
   print(df[["r_id", "ea", "dh"]].head())

Work with train/validation/test splits
--------------------------------------

Some datasets already include a ``split`` column from the original benchmark.
When a fresh deterministic split is needed, use
:class:`synrxn.split.repeated_kfold.RepeatedKFoldsSplitter`.

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader
   from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter

   loader = DataLoader(
       task="property",
       source="zenodo",
       version="1.0.0",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
   )
   df = loader.load("b97xd3")

   splitter = RepeatedKFoldsSplitter(
       n_splits=5,
       n_repeats=2,
       random_state=42,
       val_ratio=0.1,
   )

   split_indices = splitter.split(df)
   print(split_indices)

.. raw:: html

   <div class="synrxn-callout">
     <p><strong>Reproducibility tip:</strong> For a paper or benchmark report, record the dataset name, task family, source mode, version or commit SHA, cache manifest if available, split seed, and package version.</p>
   </div>

Source modes at a glance
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 24 24
   :class: synrxn-table

   * - Source mode
     - Best use case
     - Example version
     - Reproducibility level
   * - ``zenodo``
     - Published results and long-term archiving.
     - ``"1.0.0"``
     - High, citable release snapshot.
   * - ``github``
     - Aligning data with a GitHub release tag or branch.
     - ``"v1.0.0"``
     - High for tags, lower for branches.
   * - ``commit``
     - Exact development snapshot.
     - full commit SHA
     - Highest for source-state reproducibility.
   * - ``latest``
     - Exploration during active development.
     - ``"latest"``
     - Low unless the resolved commit is recorded.

Troubleshooting
---------------

Installation cannot find optional packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upgrade packaging tools first:

.. code-block:: bash

   python -m pip install --upgrade pip setuptools wheel

Zenodo downloads fail or are rate-limited
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefer GitHub-hosted release assets for routine experiments. For Zenodo-based
loading, check network access and use a persistent cache so assets are not
repeatedly requested.

Autodoc cannot import ``synrxn`` during documentation builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the package in editable mode from the project root before building docs:

.. code-block:: bash

   python -m pip install -e ".[dev]"
   cd doc
   sphinx-build -b html . _build/html

Next steps
----------

- :doc:`Data Concept <data_concept>` explains task folders, schema conventions,
  and source modes.
- :doc:`Data Records <data_records>` lists curated benchmark records and row
  counts.
- :doc:`Tutorials and Examples <tutorials_and_examples>` gives complete workflows
  for released data, pinned commits, splitting, and rebuilds.
- :doc:`API Reference <api>` documents ``DataLoader`` and split utilities.

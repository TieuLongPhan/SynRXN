.. _api-reference:

API Reference
=============

This section documents the public Python API exposed by SynRXN. For practical
usage examples, start with :doc:`Getting Started <getting_started>` and
:ref:`tutorials-and-examples`.

.. raw:: html

   <div class="synrxn-card-grid">
     <a class="synrxn-card" href="#data-access-synrxn-data">
       <strong><i class="fa-solid fa-database" aria-hidden="true"></i> Data access</strong>
       <span>Load curated benchmark tables and discover available records.</span>
     </a>
     <a class="synrxn-card" href="#splitting-utilities-synrxn-split-repeated-kfold">
       <strong><i class="fa-solid fa-shuffle" aria-hidden="true"></i> Splitting</strong>
       <span>Create reproducible repeated k-fold and train/validation/test splits.</span>
     </a>
     <a class="synrxn-card" href="#command-line-interface-synrxn-main">
       <strong><i class="fa-solid fa-terminal" aria-hidden="true"></i> CLI</strong>
       <span>Inspect developer commands for rebuilding datasets and manifests.</span>
     </a>
   </div>

Data access: :mod:`synrxn.data`
-------------------------------

The :mod:`synrxn.data` module is the main entry point for accessing curated
datasets and their manifests.

Typical responsibilities of :class:`synrxn.data.DataLoader` include:

- listing datasets for a task family,
- resolving data sources such as Zenodo, GitHub tags, exact commits, or latest,
- managing local cache directories,
- verifying or tracking downloaded assets when metadata is available,
- returning records as pandas DataFrames.

.. automodule:: synrxn.data
   :members:
   :undoc-members:
   :show-inheritance:

Minimal usage
~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

   loader = DataLoader(
       task="classification",
       source="zenodo",
       version="1.0.0",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
   )

   print(loader.available_names())
   df = loader.load("schneider_b")

Catalog and local projected access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from synrxn import DataLoader, DatasetCatalog

   catalog = DatasetCatalog()
   property_records = catalog.list(task="property", has_split=True)

   loader = DataLoader(task="property", source="local", data_dir="Data")
   test_rows = loader.load(
       "rgd1",
       columns=["r_id", "ea", "split"],
       filters={"split": "test"},
   )

Use :meth:`synrxn.data.DataLoader.iter_batches` for bounded-memory iteration
over local data.

Splitting utilities: :mod:`synrxn.split.repeated_kfold`
-------------------------------------------------------

The :mod:`synrxn.split.repeated_kfold` module provides tools for reproducible
repeated k-fold splitting of datasets.

Typical use cases include:

- generating repeated k-fold splits for property and classification tasks,
- deriving train/validation/test partitions via a user-specified validation
  ratio,
- preserving label distributions through stratified splitting when supported,
- exporting/importing split indices for exact reproducibility.

.. automodule:: synrxn.split.repeated_kfold
   :members:
   :undoc-members:
   :show-inheritance:

Minimal usage
~~~~~~~~~~~~~

.. code-block:: python

   from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter

   splitter = RepeatedKFoldsSplitter(
       n_splits=5,
       n_repeats=3,
       random_state=2026,
       val_ratio=0.1,
   )

   split_indices = splitter.split(df)

Command-line interface: :mod:`synrxn.__main__`
----------------------------------------------

The :mod:`synrxn.__main__` module exposes the command-line interface used when
invoking SynRXN as a module.

.. code-block:: bash

   python -m synrxn --help
   python -m synrxn build --help
   python -m synrxn verify-manifest --help
   python -m synrxn validate --help
   python -m synrxn datasets list --task property --has-split

The ``build`` subcommand is repository-only and intended for maintainers who
need to rebuild datasets from original sources. ``verify-manifest`` verifies
artifact sizes and checksums, while ``validate`` checks the metadata catalog,
schemas, identifiers, split values, and manifest record counts.

.. automodule:: synrxn.__main__
   :members:
   :undoc-members:
   :show-inheritance:

Query layer: :mod:`synrxn.query`
--------------------------------

.. automodule:: synrxn.query
   :members:
   :undoc-members:
   :show-inheritance:

Read-only service: :mod:`synrxn.service`
----------------------------------------

The optional FastAPI application is created with
:func:`synrxn.service.create_app`. See :ref:`query-and-service` for its bounded
HTTP contract and deployment guidance.

.. automodule:: synrxn.service
   :members:
   :undoc-members:
   :show-inheritance:

API usage guidance
------------------

- Use :class:`synrxn.data.DataLoader` for all dataset access instead of hardcoding
  local paths.
- Prefer ``source="zenodo"`` for published experiments.
- Prefer exact commit SHAs for development snapshots.
- Export split indices whenever generated splits are part of a benchmark.
- Keep package version, data version, and split settings in experiment logs.

.. _api-reference:

API Reference
=============

This section documents the public Python API exposed by SynRXN. For practical
usage examples, start with :doc:`Getting Started <getting_started>` and
:ref:`tutorials-and-examples`.

The portable ``synrxn.duckdb`` snapshot is queried through DuckDB's Python or SQL
interface; see :doc:`Database access <query_and_service>`. ``DataLoader`` and
``synrxn.access.query`` below read the canonical CSV/Parquet files. They do not
open the portable snapshot.

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

The :mod:`synrxn.data` module provides access to canonical CSV/Parquet datasets
and their manifests, including earlier published releases.

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
   python -m synrxn.reconstruction families --help
   python -m synrxn verify-manifest --help
   python -m synrxn validate --help
   python -m synrxn datasets list --task property --has-split

Use the family entry points in :doc:`data_concept` for table reconstruction.
``verify-manifest`` checks artifact sizes and checksums. ``validate`` checks the
metadata catalogue, schemas, identifiers, split values and manifest counts.

.. automodule:: synrxn.__main__
   :members:
   :undoc-members:
   :show-inheritance:

Query layer: :mod:`synrxn.access.query`
---------------------------------------

.. automodule:: synrxn.access.query
   :members:
   :undoc-members:
   :show-inheritance:

Read-only service: :mod:`synrxn.access.service`
-----------------------------------------------

The optional FastAPI application serves verified Parquet exports of the
canonical tables. To prepare a local export and start the service::

    python -m pip install -e ".[query,service]"
    synrxn parquet build --data-dir Data --output-dir Parquet --manifest manifest.json
    SYNRXN_PARQUET_DIR=Parquet SYNRXN_MANIFEST=manifest.json synrxn-service

The service publishes its OpenAPI schema at ``/docs``. Use
:func:`synrxn.access.service.create_app` to configure the application in Python.

.. automodule:: synrxn.access.service
   :members:
   :undoc-members:
   :show-inheritance:

Task evaluation and dummy controls
----------------------------------

See :doc:`task_evaluation` for task-specific metrics, weak controls and split
requirements. Discovery does not load datasets or train models.

.. automodule:: synrxn.baseline.dummy
   :members: DummyProtocol, DummyPredictor, get_dummy_protocol, list_dummy_protocols

.. automodule:: synrxn.benchmarks.sanity
   :members: run_metric_sanity

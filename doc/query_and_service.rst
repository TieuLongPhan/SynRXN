.. _query-and-service:

Parquet Queries and Read-only Service
=====================================

SynRXN keeps compressed CSV as the canonical, citable release format. Parquet
is a deterministic derived artifact for typed projection and filtering; DuckDB
queries it directly, so no database server or import migration is required.

Build and verify a derived release
----------------------------------

.. code-block:: bash

   pip install "synrxn[query]"
   synrxn verify-manifest --manifest manifest.json --root Data
   synrxn parquet build --data-dir Data --output-dir Parquet --manifest manifest.json
   synrxn parquet verify --data-dir Data --parquet-dir Parquet

Every Parquet file embeds the canonical source SHA-256 and converter version.
``Parquet/index.json`` records both source and derived checksums plus the release
version, DOI, and canonical manifest checksum. The conversion
uses a stable schema, Zstandard compression, 100,000-row groups, and verifies a
full Arrow round trip before atomically replacing an artifact.

Bounded Python queries
----------------------

.. code-block:: python

   from synrxn import DataLoader

   loader = DataLoader(
       task="classification",
       source="local",
       data_dir="Data",
       parquet_dir="Parquet",
   )

   arrow_table = loader.load("schneider_b", nrows=100, format="arrow")
   with loader.scan("schneider_b") as scan:
       page = scan.collect(
           columns=["r_id", "label", "split"],
           filters={"split": "test"},
           order_by="r_id",
           limit=100,
       )
       summary = scan.stats()

The query API allowlists catalog datasets and observed columns, parameterizes
filter values, caps a single query at 10,000 rows, and records artifact and
query provenance in ``page.attrs["synrxn"]``. It does not accept arbitrary SQL.

Reproducible performance checks are available in
``script/benchmark_query_layer.py``. The script isolates each case in a fresh
process and records wall time, peak resident memory, row count, and dependency
versions for small, medium, and large datasets.

The checked run is available as :download:`query-benchmark.json
<_static/query-benchmark.json>`. On ``synthesis/uspto_mit`` it measured a
479,035-row pandas load at 2.87 s and 485 MiB peak resident memory, versus a
bounded 10,000-row Parquet filter at 0.09 s and 223 MiB. These measurements are
environment-specific; rerun the script on the intended deployment hardware.

Optional HTTP service
---------------------

.. code-block:: bash

   pip install "synrxn[service]"
   SYNRXN_PARQUET_DIR=Parquet SYNRXN_MANIFEST=manifest.json synrxn-service

The service validates the mounted Parquet checksum index and catalog before
starting. It exposes OpenAPI at ``/docs`` and the following read-only routes:

- ``GET /health`` and ``GET /metrics``
- ``GET /v1/datasets``
- ``GET /v1/datasets/{task}/{name}``
- ``GET /v1/datasets/{task}/{name}/rows``
- ``GET /v1/datasets/{task}/{name}/stats``
- ``GET /v1/releases/{version}``

Rows use stable source-row ordering by default. Projection, equality filters,
ordering, offset, and limit are allowlisted; the HTTP page limit defaults to
1,000. Immutable responses carry artifact-derived ETags and cache headers.
Request IDs, timing headers, structured log fields, query-capacity rejection,
and aggregate operational counters provide a small observability baseline.

Container deployment and rollback
---------------------------------

Build ``Dockerfile.service`` and mount a complete verified release read-only at
``/release``. The image runs as UID 10001 and contains no dataset files. To
activate or roll back, atomically change the release mount/symlink and restart
the stateless service; application code does not rewrite release artifacts.

PostgreSQL is not needed for immutable browsing, filters, pagination, release
metadata, or statistics. Add a transactional SQL server only if a future
product introduces mutable shared state such as accounts, private workspaces,
annotations, curation, submissions, or review workflows.

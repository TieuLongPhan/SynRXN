.. _query-and-service:
.. _database-access:

Database access
===============

Open ``synrxn.duckdb`` to use the main database. It contains reaction records,
the task hierarchy, data sources, licenses and native assets.
Normal SQL queries need this file and DuckDB. The reconstruction folders and the
builder's JSON report are not runtime dependencies.

The current ``1.5.0`` snapshot has been built and verified locally. Its public
download and DOI are pending. The files below are available in the author checkout.

Files and availability
----------------------

The current local artifacts are:

.. list-table:: Snapshot files
   :header-rows: 1
   :widths: 37 63

   * - File
     - Use
   * - ``synrxn.duckdb``
     - **11.44 GB.** The full database opened by DuckDB.
   * - ``synrxn.duckdb.zst``
     - **7.63 GB.** Lossless transfer archive. Decompress before opening.
   * - ``SHA256SUMS``
     - A small text file for checking the database bytes against the release checksum.
   * - Builder ``manifest.json``
     - Records construction and verification. Runtime metadata are in
       ``catalog.*``; this report is not needed to query the database.

The main download includes **5,485,077 stored rows** and the admitted native
assets. Some source views share the same observations.

The original portable artifacts are under
``.cache/synrxn/releases/1.5.0/main/`` in the author checkout.
The compact snapshots with the current evaluation views are under
``.cache/synrxn/releases/1.5.0/mapping-panels-full/`` with the Transition1x
configuration file embedded and ``.cache/synrxn/releases/1.5.0/mapping-panels/``
with the same asset listed as an external download.
The data deposit files are staged in
``.cache/synrxn/releases/1.5.0/zenodo-dataset/`` in the author checkout.
GitHub release archiving uses the software files. The dataset deposit contains
the database archive, build manifest, source terms and SHA-256 checksums.

If using the compressed copy, decompress it with the Zstandard command-line
tool, then check the uncompressed file using its matching ``SHA256SUMS``::

    zstd --decompress synrxn.duckdb.zst --output synrxn.duckdb
    sha256sum --check SHA256SUMS

For query use, keep ``synrxn.duckdb`` at any chosen path and pass that path to the connection.

Open and read a table
---------------------

The following example was checked with DuckDB 1.5.4::

    python -m pip install "duckdb==1.5.4"

Use the path to an existing snapshot. ``read_only=True`` prevents a mistyped path
from creating a new, empty database.

.. code-block:: python

   from pathlib import Path
   import duckdb

   database = Path("synrxn.duckdb").resolve(strict=True)
   with duckdb.connect(str(database), read_only=True) as db:
       print(db.sql("SELECT family_id, title FROM catalog.families ORDER BY family_id").fetchall())

       name = db.execute("""
           SELECT view_name
           FROM catalog.dataset_tables
           WHERE namespace = ? AND table_key = ? AND release_admitted
       """, ["canonical", "yield/hitea"]).fetchone()[0]

       table = db.table("data." + name)
       print(table.project(
           "sample_id, reaction_smiles, yield_pct, yield_measurement_type"
       ).limit(5).fetchall())

This reads five HiTEA source records. It does not reconstruct the data or require
``archive/datasets/raw/`` or ``Data/families/``. For a pandas result, replace ``fetchall()`` with
``df()`` in an environment containing pandas. Retrieve the relation name from the
catalogue; its generated hash suffix is not a dataset identifier.

What is inside the file
-----------------------

The scientific hierarchy is **family → task → source table → stored row**.
Task membership is recorded in a separate binding table because a source table
can support several tasks. A recent compact build has the following layout:

.. code-block:: text

   synrxn.duckdb
   ├── catalog
   │   ├── families              scientific families
   │   ├── tasks                 task definitions and parent families
   │   ├── dataset_tables        source IDs, schemas, hashes, counts and terms
   │   ├── task_table_bindings   membership, units, roles and qualifications
   │   ├── assets                paths and hashes of embedded native files
   │   ├── verified_inputs       checked construction inputs
   │   └── build_info            software versions and storage mode
   ├── data
   │   └── v_*                   102 source relations
   ├── source_extensions        source-specific fields used by internal views
   ├── evaluation               qualified scoring and split views
   └── assets
       └── chunks                compressed native files and notices

The database contains **12 families, 40 tasks and 102 admitted table names**.
Primary task counts use 95 of those tables. The held Diels–Alder record is
metadata only. The 144 task–table bindings include references to shared tables.
In a compact build, verified internal views reuse columns or observations across
source relations. The registered names, source row counts, and fields stay available.

Check the storage mode of a snapshot:

.. code-block:: sql

   SELECT value_json AS storage_mode
   FROM catalog.build_info
   WHERE key = 'storage_mode';

The result should be ``"materialized"`` for a self-contained snapshot. A
``"file_views"`` catalogue needs its external source files; a
``"metadata_only"`` catalogue contains no source-row relations. Builders and
these development modes are documented in :doc:`data_concept`.

For a portable build with the audited overlaps consolidated, use:

.. code-block:: bash

   python -m synrxn.reconstruction storage build --materialize --compact \
       --output .cache/synrxn/compact

Add ``--external-transition1x`` for a smaller database. The trajectory index,
final geometries, and source terms remain available in SQL. Reading the per-frame
energies, forces, and positions requires the matching native HDF5 file. The
snapshot lists its path, size, and SHA-256 in ``catalog.external_assets``.

Scoring records and splits
--------------------------

The task registry records ``counted_under`` when a companion table shares the
primary table's observations and prediction target. The ``evaluation`` schema
exposes the primary records and preserves access to named comparisons.
Use the enriched SNAr and cycloaddition views for new evaluations. The source
formats ``property/snar`` and ``property/cycloadd`` support reproduction of
earlier workflows.

* ``evaluation.snar`` supplies 503 activation free energies with solvent and
  temperature. ``evaluation.snar_reaction_only`` uses the same IDs, targets
  and splits for an input ablation. Use ``reaction_smiles`` as the sole model
  feature in the reaction-only comparison.
* ``evaluation.cycloaddition`` supplies 5,269 rows with both free-energy
  targets, all in water at 298.15 K. Use
  ``evaluation.cycloaddition_activation_barrier`` or
  ``evaluation.cycloaddition_reaction_energetics`` to select one target.
* ``evaluation.catechol`` flags the 656 published single-solvent observations
  within 1,227 full-panel records. ``evaluation.catechol_single_solvent`` retains
  the original solvent conditions and adds ``full_panel_record_id`` and
  ``full_panel_split``. The assignments differ for 270 matched observations.
  Use the full-panel split for a comparison across the two views. The subset's
  ``split`` supports reproduction of its published evaluation.
* Score NatComm on the 491 published rows in
  ``evaluation.atom_mapping_natcomm``. Score the rest of Golden on the
  1,267 rows in ``evaluation.atom_mapping_golden_only``. These panels have
  no shared reaction identities under the catalogue's normalization checks.
  The full Golden table has 1,758 rows for reproducing the original evaluation.
  The NatComm view keeps its published mapper outputs and links to full Golden
  by ``golden_record_id``.
* Schneider, TPL and USPTO-50K retain their separate balanced and unbalanced
  reaction inputs. Join both variants to the corresponding
  ``evaluation.classification_{schneider,tpl,uspto_50k}_paired_split`` view
  by source record ID to use the same published assignment.

New builds expose ``evaluation.rbl_complex_references``, which selects the 1,747
Complex rows with reference completions. The original relation retains the case
with a missing completion. ``evaluation.phosphatase_assay_response`` exposes the
original values under ``assay_response`` to identify their measurement endpoint.

The ``evaluation`` schema also exposes proposed group splits for ECReact, RGD1,
RDB7, and the three USPTO synthesis tables. Each view retains the published
assignment. The proposed split assigns each literal reaction string as a unit.
Rows with a missing reaction string retain their source membership and have an
unassigned proposed split. Use separate grouping protocols to test scaffold,
source, or canonical chemistry transfer.

.. code-block:: sql

   SELECT proposed_split, count(*) AS rows, count(DISTINCT group_id) AS groups
   FROM evaluation.property_rgd1_split
   GROUP BY proposed_split;

Find data by scientific task
----------------------------

Run these SQL examples in a DuckDB client, or pass them to ``db.sql(...)`` in an
open Python connection. First list tasks in the outcomes family:

.. code-block:: sql

   SELECT task_id, title
   FROM catalog.tasks
   WHERE family_id = 'outcomes'
   ORDER BY task_id;

Then find the source tables supporting yield, assay response and conversion:

.. code-block:: sql

   SELECT d.namespace, d.table_key, d.stored_rows,
          b.counting_unit, b.role, b.qualification,
          d.effective_license_expression, d.view_name
   FROM catalog.task_table_bindings AS b
   JOIN catalog.dataset_tables AS d
     ON d.namespace = b.namespace AND d.table_key = b.table_key
   WHERE b.task_id = 'yield_conversion' AND d.release_admitted
   ORDER BY d.namespace, d.table_key;

``namespace`` and ``table_key`` together identify a source table, such as
``canonical`` and ``yield/hitea``. ``task_id`` identifies a scientific problem,
such as ``yield_conversion``. The :doc:`family browser <families>` displays these
memberships, observation units and reuse terms without requiring SQL.

Inspect values, units and terms
-------------------------------

Source formats determine the stored types:

.. list-table:: Source values in the snapshot
   :header-rows: 1
   :widths: 22 78

   * - Source format
     - Database representation
   * - CSV or CSV.gz
     - Text columns, preserving identifiers, leading zeros and recorded numerical text.
   * - Parquet
     - Native column types from the prepared table.
   * - JSONL or JSONL.gz
     - One JSON ``payload`` per stored row, preserving nested native fields.

Inspect the selected relation's columns before analysis. For the ``table``
relation in the Python example, use ``table.columns`` and ``table.types``.
Numerical analysis of CSV fields requires an explicit conversion, such as
``TRY_CAST(yield_pct AS DOUBLE)``. Check measurement type and units first:
yield, assay response and conversion can represent different endpoints.
Task qualifications and split protocols are described in :doc:`task_evaluation`.

``catalog.dataset_tables.effective_license_expression`` and ``rights_json``
record the effective reuse terms and admission evidence. ``source_urls_json``
and ``sha256`` identify the original source record and prepared-file checksum.
Native files carry their own terms in ``catalog.assets``. The Apache-2.0 code
license does not replace dataset-specific terms.

To write a small derived table from the selected relation, use
``table.limit(1000).write_parquet("hitea-preview.parquet")`` inside the Python
connection. This preserves the snapshot's column types; it does not convert
text measurements into numbers automatically.

Recover a native file
---------------------

Some rows identify a native spectrum matrix or HDF5 trajectory rather than
containing the full array in a cell. These original paths are logical source
identifiers. The admitted files are embedded in the snapshot and can be
extracted when a native reader needs them.

For example, locate the PMG 284 NMR matrix:

.. code-block:: sql

   SELECT path, bytes, sha256, license_expression
   FROM catalog.assets
   WHERE path = 'Data/raw/coverage/process/magnetstein_reactions/data/PMG_284/preprocessed_mixture.csv';

Use the helper from the current SynRXN checkout to extract its original bytes::

    python -m pip install -e ".[query]"
    python -m synrxn.reconstruction storage extract-asset \
        --database synrxn.duckdb \
        --path Data/raw/coverage/process/magnetstein_reactions/data/PMG_284/preprocessed_mixture.csv \
        --output pmg-284-mixture.csv

Extraction checks the decompressed bytes against the recorded SHA256 before
publishing the output and refuses an existing destination. The same command
extracts embedded source notices and metadata. Use the source-specific reader
for the recovered file; extraction preserves the original format.

Verify the snapshot
-------------------

From the current checkout, run a complete internal verification::

    python -m synrxn.reconstruction storage verify --database synrxn.duckdb

It opens the database read-only with external file access disabled, checks the
admitted physical relations and row counts, and verifies decompressed hashes for
all unique embedded assets. It does not need the reconstruction source folders.
For a quicker table-count check, add ``--skip-asset-hashes``; this omits the native
asset hash scan. The external checksum check and this internal verification
serve different purposes: one verifies the complete release file, the other
checks its registered tables and embedded assets.

The current candidate also passed a relocation check in an independent process
without using the original source folders. These are packaging and integrity
checks; task-specific scientific evaluation remains separate.

.. _compatibility-file-access:

Python file access
------------------

``DataLoader`` reads individual canonical CSV and Parquet tables. See
:doc:`getting_started` for a local example and :doc:`api` for filtering, batch
iteration and the optional HTTP file service.

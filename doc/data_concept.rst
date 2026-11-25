.. _data-concept:

Data model and reconstruction
=============================

A family groups related scientific tasks. Each task defines an input, a target
and an evaluation population. A source table stores the records used by one
or more tasks. Its observation unit can be an assay, an annotated document, a
spectrum, or a computed structure. A stored row may fall outside a task's
eligible evaluation population. See :doc:`task_definitions` for the inputs and
expected results, and :doc:`families` for table membership.

Rebuild SynRXN 1.5.0
--------------------

From the repository root, install the reconstruction and query dependencies.

.. code-block:: bash

   python -m pip install -e ".[expansion,query]"

The main snapshot uses 102 prepared tables and their admitted native assets.
Use the source locks under ``Data/metadata/`` to identify the required assets.
Restore original inputs under ``archive/datasets/raw/`` and restore matching
prepared reference files under ``Data/families/``. Bulk rebuilds compare supported
source recipes with those references. Tables supplied as fixed artifacts are
restored from a checksum-matching prepared cache. The code checkout contains the
catalogue and recipes. A public deposit of the complete data bundle is pending.

Follow the table preparation steps below for files that need to be restored or
regenerated. Then build and verify the main DuckDB snapshot. For exact native
adapter comparisons, use the pinned environment in
``Data/metadata/reconstruction/coverage/requirements.txt``.

Storage layout
--------------

``Data/`` has two directories:

.. code-block:: text

   Data/
   ├── families/           prepared data, organized by family and task
   └── metadata/           catalogues, source locks, licenses and task definitions
       ├── catalog.yaml
       ├── sources/
       ├── coverage/
       ├── expansion/
       └── reconstruction/

Each admitted table has one prepared file under ``Data/families/``. A task that
shares a table links to its file under the first task that uses it in the
registry. The adjacent ``table.json`` records row counts, source identity,
checksum and use terms. The held Diels–Alder record has metadata only.

Git excludes the native cache and prepared table payloads.

Prepare a table
---------------

The entry points in ``synrxn/reconstruction/families/<family>/<task>/`` select
individual tables. The following recipe reproduces the PMG-284 NMR table from
its pinned source files and places the result in the family tree::

    python -m synrxn.reconstruction.families.analysis.analytical_interpretation.nmr_pmg_284_spectra prepare --offline

Use ``plan`` to see a table's inputs. Use ``verify`` to check its pinned assets
and ``build`` to retain an isolated reconstruction report.

``prepare`` uses a pinned local input when present. Recipes with upstream
acquisition can fetch missing inputs. ``--offline`` restricts preparation to
local files. ``--cache DIR`` accepts a directory containing native inputs under
``raw/`` or prepared files under ``families/``. With ``--from-cache``, SynRXN
places a prepared copy after verifying its checksum. Canonical artifacts without
a native preparation recipe use the pinned prepared file.

.. _atom-mapping-preparation:

Atom-mapping tables
-------------------

The main snapshot includes ten atom-mapping tables. Find their identifiers,
source references and reuse terms in :doc:`data_records`. Prepare a table using
its entry point and a matching local artifact.

.. code-block:: bash

   python -m synrxn.reconstruction.families.representation.atom_mapping.ecoli prepare --offline

.. _build-database:

Build a database snapshot
-------------------------

With the prepared tables and admitted native assets available locally, check
the family tree and build a self-contained snapshot.

.. code-block:: bash

   python -m synrxn.reconstruction families organize --check-hashes --allow-missing
   python -m synrxn.reconstruction storage build --materialize --scope main \
       --output .cache/synrxn/storage-001
   python -m synrxn.reconstruction storage verify \
       --database .cache/synrxn/storage-001/synrxn.duckdb

The destination must be a new or empty directory under ``.cache/`` or
``reports/``. The builder validates source hashes, verifies the imported row
counts, and checks embedded assets with external file access disabled. It
writes ``synrxn.duckdb`` and a build report with source pins and the database
checksum. See :doc:`query_and_service` for opening the result.
``--allow-missing`` lets the local tree omit optional files. The main builder
requires all 102 admitted files before it produces a snapshot.

The main scope covers 12 families and 40 tasks with 102 materialized tables.
``--scope catalogue`` also includes the optional FREA task. For local inspection,
a build without ``--materialize`` creates SQL views over prepared files.
``--metadata-only`` builds the catalogue without source-row relations.

Browse, rebuild and verify a family
-----------------------------------

List families, inspect the selected source closure, and write a verification
report to a new directory::

    python -m synrxn.reconstruction families list
    python -m synrxn.reconstruction families plan --family outcomes
    python -m synrxn.reconstruction families verify --family outcomes --check-rows \
        --output .cache/synrxn/runs/outcomes-verify

Use ``families build`` with the same options to rebuild supported views in an
isolated workspace and compare their bytes. Shared tables are checked once per
run. Family verification checks data integrity. Predictive results require the
task-specific procedures in :doc:`task_evaluation`.

Refresh and verify the local family tree after updating its source metadata::

    python -m synrxn.reconstruction families organize --check-hashes

Source-specific acquisition commands are listed by
``python -m synrxn.reconstruction families import-source --list``.

Compare all source recipes
--------------------------

With the full local catalogue and pinned native inputs available, rebuild the
supported expansion and coverage recipes in one isolated run. This comparison
also verifies fixed prepared tables and checks their row counts.

.. code-block:: bash

   python -m synrxn.reconstruction families build --all --check-rows \
       --output .cache/synrxn/runs/catalogue-rebuild

This checks the full local catalogue, including optional and held references.
The run directory contains regenerated files, comparison results and logs.
The :ref:`main snapshot builder <build-database>` verifies the 102 admitted
tables. Use ``prepare`` when a recipe needs to place an individual result into
``Data/families/``.

M-CSA mechanism tables
----------------------

The release adapter rebuilds four overlapping M-CSA views using the pinned
companion at ``archive/research/previous/artifacts/mcsa-001/``. The lock in
``Data/metadata/reconstruction/coverage/adapters/mcsa_sources.json`` lists its
required files and checksums. With the companion and prepared references in
place, check all four in one run.

.. code-block:: bash

   python -m synrxn.reconstruction families build --task mechanistic_prediction \
       --check-rows --output .cache/synrxn/runs/mcsa-release-check

For the independent rebuild from the original drawings, restore the
checksum-pinned native bundle and converter files under
``archive/datasets/raw/mcsa-v5/``. The converter requires Python 3.12.13 and the
versions in ``requirements-converter.txt``. Run the importer from a separate
SynRXN environment with the qualified ``synkit==1.6.3+synrxn.1`` backend.

.. code-block:: bash

   python3.12 -m venv .cache/mcsa-converter-env
   .cache/mcsa-converter-env/bin/python -m pip install \
       -r archive/datasets/raw/mcsa-v5/requirements-converter.txt
   python -m synrxn.reconstruction families import-source mcsa_native \
       --converter-python .cache/mcsa-converter-env/bin/python \
       --output-dir .cache/mcsa-rebuild --jobs 2

``native-build/`` contains the conversion results. ``extension/`` contains the
standalone companion and its verification evidence. The release adapter checks
the companion's files against its source lock before preparing release tables.
See :doc:`task_evaluation` for replay eligibility and split requirements.

Source records and reuse
------------------------

``Data/metadata/sources/`` records upstream references and source versions.
The task registry and release manifest under ``Data/metadata/reconstruction/``
record table membership and redistribution decisions. Dataset IDs and source
checksums remain stable when files move. Rights and citations apply per source.

- :download:`Task registry <../Data/metadata/reconstruction/task_registry.json>`
- :download:`Release manifest <../Data/metadata/reconstruction/coverage/release_manifest.json>`
- :doc:`data_records`

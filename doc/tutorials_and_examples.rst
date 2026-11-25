.. _tutorials-and-examples:

Examples
========

Read a prepared table
---------------------

Inspect the canonical dataset metadata, then read selected columns from a
local table. Run the example from the checkout with prepared data.

.. code-block:: python

   from synrxn import DataLoader, DatasetCatalog

   catalog = DatasetCatalog()
   table = catalog.get("classification", "schneider_b")
   print(table.targets, table.split_values)

   loader = DataLoader(task="classification", source="local", data_dir="Data")
   sample = loader.load("schneider_b", columns=["r_id", "label"], nrows=10)
   print(sample)

For a full pass over a file, read it in batches.

.. code-block:: python

   for batch in loader.iter_batches("schneider_b", batch_size=10_000):
       print(len(batch))

Query yield data by task
------------------------

Open the local snapshot with DuckDB. This query lists admitted tables for the
yield and assay-response task together with their observation units.

.. code-block:: sql

   SELECT d.namespace, d.table_key, d.stored_rows, b.counting_unit
   FROM catalog.task_table_bindings AS b
   JOIN catalog.dataset_tables AS d
     ON d.namespace = b.namespace AND d.table_key = b.table_key
   WHERE b.task_id = 'yield_conversion' AND d.release_admitted
   ORDER BY d.namespace, d.table_key;

See :doc:`query_and_service` for opening the database and querying rows in the
selected relation. A source table's stored count can exceed the eligible
population for a particular evaluation.

Reconstruct an NMR table
------------------------

With the pinned PMG-284 source in the local archive, reproduce its 999 prepared
spectra from the checkout::

    python -m synrxn.reconstruction.families.analysis.analytical_interpretation.nmr_pmg_284_spectra prepare --offline

The recipe integrates the measured spectra over the reference component
windows and checks the output checksum before placing the table. The
analytical-interpretation and monitoring tasks link to the same prepared file.
See :doc:`data_concept` for acquisition, cache options and release builds.

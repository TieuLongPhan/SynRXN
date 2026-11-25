Release notes
=============

1.5.0
-----

- The main DuckDB snapshot covers **12 families, 40 tasks and 102 data tables**.
  Source terms, observation units and admitted native assets are embedded.
- Prepared files live in ``Data/families/``. A table shared across tasks is stored
  once. Catalogues, source locks and license notices live in ``Data/metadata/``.
- The Diels–Alder source record is metadata only. FREA is available as an
  optional source collection through the full local catalogue.
- Per-table preparation verifies source pins and writes to the family tree.
  Metric and split profiles are available for all 40 release tasks.

The wheel and source distribution have been built locally. The main database
has passed table-count and embedded-asset checks. Its public deposit and DOI
are pending. See :doc:`query_and_service` for the local snapshot and
:doc:`data_concept` for reconstruction.

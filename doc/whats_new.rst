What's New
==========

Use this page as the public changelog for SynRXN documentation, package
releases, and versioned dataset archives. Every release entry should summarize
what changed, which datasets moved, and which archive should be cited.

.. raw:: html

   <div class="synrxn-callout-card">
     <span class="synrxn-callout-icon"><i class="fa-solid fa-clock-rotate-left" aria-hidden="true"></i></span>
     <div>
       <strong>Current release: SynRXN 1.0.0.</strong>
       <p>This release aligns the documentation with the Scientific Data descriptor, adds new AAM and synthesis records, and promotes the public API/documentation to a stable 1.0 series.</p>
     </div>
   </div>

v1.0.0 — Scientific Data release
--------------------------------

Added
~~~~~

- Added ``aam/enzyme_map`` with 47,974 enzymatic atom-mapping records from the
  EnzymeMap reaction collection :cite:p:`heid2023enzymemap`.
- Added/standardized ``synthesis/da`` as a Diels--Alder reaction benchmark with
  11,011 records and columns ``r_id``, ``code``, ``reaction_original``,
  ``reaction``, and ``rsmi`` :cite:p:`lam2024every`.
- Added dedicated documentation pages for :doc:`Data Records <data_records>`,
  :doc:`Paper <paper>`, :doc:`Issues <issues>`, and this release history.
- Added source citations directly to the per-dataset inventory tables.

Changed
~~~~~~~

- Moved the project version to ``1.0.0``.
- Refreshed the documentation around the PyData/SynKit-style layout, including
  card navigation, icon cards, source badges, and a clearer dataset-selection
  guide.
- Renamed the inventory page from "Benchmark Data Records" to
  :doc:`Data Records <data_records>`.
- Split the data inventory into five task-family sections: reaction
  rebalancing, atom-to-atom mapping, reaction classification, reaction property
  prediction, and synthesis prediction.
- Updated examples and reproducibility snippets to use the ``1.0.0`` release
  where a released Zenodo/GitHub data version is intended.

Fixed
~~~~~

- Removed stale synthesis error-diagnostic rows from the public data-record
  inventory.
- Corrected the synthesis summary from "4 curated datasets plus error
  diagnostics" to "4 curated datasets".
- Corrected the Diels--Alder record schema to document ``r_id`` as the row
  identifier.
- Updated citation guidance to include BibTeX for the Scientific Data paper.

Data and citation notes
~~~~~~~~~~~~~~~~~~~~~~~

- Cite the primary paper: :cite:p:`phan2026synrxn`.
- Cite the exact Zenodo version DOI used for a benchmark run when the v1.0.0
  archive is minted. Until then, record the Git tag or full commit SHA together
  with ``synrxn==1.0.0``.
- Refresh local caches if you previously cached ``aam`` or ``synthesis`` data,
  because ``enzyme_map`` and the standardized Diels--Alder record change the
  available dataset list.

v0.0.8 — archived pre-1.0 release
---------------------------------

This archived release is kept here as an example of pre-1.0 citation metadata.

.. list-table::
   :header-rows: 1
   :widths: 26 74
   :class: synrxn-table

   * - Field
     - Notes
   * - Release
     - ``v0.0.8``
   * - Date
     - ``2025-11-21``
   * - Archive
     - Zenodo version DOI: ``10.5281/zenodo.17672847``
   * - Citation
     - Cite both :doc:`the Scientific Data paper <paper>` and the exact Zenodo version used.
   * - Migration
     - Update local caches if row counts, checksums, or manifests changed.

Release-entry template
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   v1.0.x — YYYY-MM-DD

   Added
   - Added <dataset_name> to <task_family>.
   - Added manifest metadata for <field>.

   Changed
   - Updated <dataset_name> from <old_rows> to <new_rows> records.
   - Renamed <old_column> to <new_column> for schema consistency.

   Fixed
   - Corrected provenance metadata for <source>.

   Citation
   - Paper: 10.1038/s41597-026-07260-w
   - Zenodo version DOI: 10.5281/zenodo.<version_record>

Maintainer checklist
--------------------

Before publishing a new entry, verify that:

- ``manifest.json`` contains current checksums, row counts, column names, and
  license metadata.
- :doc:`Data Records <data_records>` reflects the updated inventory.
- :doc:`API Reference <api>` still matches the public loader and split APIs.
- The documentation sidebar includes What’s New, Paper, Issues, API Reference,
  and References.

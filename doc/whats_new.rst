What's New
==========

Use this page as the public changelog for SynRXN documentation, package
releases, and versioned dataset archives. Every release entry should summarize
what changed, which datasets moved, and which archive should be cited.

.. raw:: html

   <div class="synrxn-callout-card">
     <span class="synrxn-callout-icon"><i class="fa-solid fa-clock-rotate-left" aria-hidden="true"></i></span>
     <div>
       <strong>Next planned release: SynRXN 1.1.1.</strong>
       <p>This release prepares SynRXN as a verified benchmark workspace with a static dataset catalog, efficient local queries, a read-only service, and SynKit 1.5 compatibility.</p>
     </div>
   </div>

v1.1.1 — verified benchmark workspace (pending release)
-------------------------------------------------------

Added
~~~~~

- Added a manifest schema, release verifier, and metadata validator.  The
  published dataset inventory can now be checked against artifact sizes and
  SHA-256 checksums before derived catalog, Parquet, or service assets are
  generated.
- Added a static, filterable :doc:`Dataset Catalog <catalog>` with task,
  target, license, scale, and split filters; dataset comparison and export;
  compact sample/schema inspection; and manifest-backed citation and loading
  snippets.
- Added generated reaction depictions for catalog datasets.  The catalog opens
  with a representative ``classification/schneider_b`` benchmark example and
  remains usable when viewed from a local static documentation preview.
- Added a deterministic CSV-to-Parquet query layer, checksum index, CLI
  commands for building and verifying derived assets, and an allowlisted query
  API for projections, filters, ordering, pagination, batches, and statistics.
- Added an optional read-only FastAPI service with bounded requests, release
  integrity checks at startup, response caching headers, and a container
  definition for mounting a verified release.
- Added a reproducible SynKit AAM validator comparison script, an AAM
  compatibility test, and :doc:`AAM validation documentation <aam_validation>`.

Changed
~~~~~~~

- Updated the SynKit requirement to ``>=1.5.0,<1.6.0`` and adapted the public
  AAM accuracy helper to the SynKit validator interface.
- Updated the documentation interface from a promotional landing-page style to
  a restrained benchmark-workspace layout, with a simplified abstract SynRXN
  wordmark and benchmark-oriented catalog presentation.
- Extended ``DataLoader`` with local column projection, bounded filtering,
  batches, Arrow output, and Parquet scanning for large benchmark workflows.
- Updated package, documentation, and CI configuration so catalog and query
  assets are generated from the verified release metadata rather than manually
  maintained copies.

Fixed
~~~~~

- Restored compatibility of the legacy SynRXN AAM normalization utilities with
  the installed RDKit standardization API.
- Verified that legacy SynRXN and SynKit 1.5 AAM validation decisions agree for
  both reaction-center and ITS validation across 47,232 evaluated mapper
  decisions; the stricter SynKit mode remains intentionally distinct.
- Fixed catalog startup in Sphinx and standalone local HTML previews.  Catalog
  data is now available as a generated page asset instead of relying solely on
  a browser ``fetch`` request.

Release and citation notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

- This entry describes pending v1.1.1 work.  The package metadata and Zenodo
  archive remain at v1.0.0 until a v1.1.1 release is tagged and deposited.
- At publication, update the package version, manifest release metadata, and
  exact Zenodo version DOI together; benchmark users should cite that exact
  archive alongside the primary paper.

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

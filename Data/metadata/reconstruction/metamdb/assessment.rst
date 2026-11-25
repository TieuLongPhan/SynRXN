MetAMDB source and reference assessment
======================================

Current selection: the broad QC collection described below is retired from the
active AAM pool. Its artifacts are archived under
``Data/archives/metamdb/full-qc-27451-1375c16b/assessment/``. The exact requested
4,456-reaction paper subset remains pending its source table or ID list; the
177-row cyclic dataset is available. See ``paper-subset-status.json`` and
``processing.rst`` in this directory.

Source checked on 2026-10-27
----------------------------

The complete download is
``atom_mappings.zip``, obtained from
https://metamdb.tu-bs.de/img/atom_mappings.zip. Its SHA-256 is
``b460f3ec5f842f757e35ad9eb0df0e5edf3450c289c272b2bfdcfb88c3689616``.
The source directory dates this download to 2022-01-17. It contains 49,928 RXN
files, without a curation-status table, README or license file. RXN headers and
the official documentation identify RDT as the computational mapping source.

This download is separate from the previously supplied ``ringreactions.zip``
containing 141 train and 36 test cases. Candidate ordering in those CSVs does
not itself establish a chemically correct or manually reviewed target.

Human curation and the 1,000 versus 10,000 question
--------------------------------------------------

The 2022 MetAMDB paper reports approximately 43,000 mapped reactions and says
"Around 1000 of these reactions were curated." This is the published historical
curation figure, not a fresh count of the current database or a verified count
inside the downloaded ZIP.

Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8878866/

The newer BRMapper preprint supplied subsequently by the user reports a larger
number. Section 2.3, page 9, describes 10,579 MetAMDB reactions marked as
validated, interpreting that flag as expert inspection. It reports 4,456
remaining reactions after excluding large representation spaces. This supports
the user's recollection of a roughly 10,000-reaction claim. It does not identify
the members of that subset in our ZIP.

New paper: https://ssrn.com/abstract=7404606

The authors' public repository at commit
``b1d428f3ed2d1ef5a1311150cc335ba1a916c06a`` supplies
``data/aam_metamdb.csv`` with 49,928 rows and the same native reaction-ID set as
the complete ZIP. Its columns are ``reaction_id,reaction_smiles``; it has no
curation flag. Thus this file is not the published 10,579-row filtered subset.
The published preprocessing code starts from existing CSV/candidate inputs;
it does not provide the upstream MetAMDB validation-filter query or the
10,579 selected identifiers.

As a concrete crosscheck, native reaction 1 appears in the authors' CSV with
the same normalized mapping as the ZIP, but its live MetAMDB record has
``updated=false`` and ``balanced=true``. Its live RXN payload is byte-identical
to the ZIP member. This demonstrates that the full public CSV cannot be
treated wholesale as the manually reviewed subset. It does not determine the
total current curated count or invalidate the paper's separate subset claim.
See ``reaction-1-crosscheck.json`` in this directory.

Repository: https://github.com/NoraBeier/BRMapper

The same repository supplies a separate 177-row
``code/02_data_preprocessing/input_data/ringreactions_gt.csv`` with native
reaction IDs, formulae, mappings, ring labels and clusters. This is an
author-supplied ground-truth table, preserved separately from the full mapping
collection. Its source designation does not independently prove manual review.
The preprocessing code generates product tautomers and Kekule forms from these
reference mappings, then compares separate enumerated candidate ITS graphs
against them. Alternate reference strings are not designated as negatives
merely because they follow the first string in a row.

The website distinguishes manually curated and automatically generated mappings:
https://metamdb.github.io/docs/database-search/. Its source at commit
``af2e7c1b77a8eac34642d338c42e7feba41115a2`` labels the ``updated`` field
"Curated" in ``client/src/components/database-query/DatabaseQuery.jsx``.
``server/src/models/casm.py`` sets that field when a review is approved by
``server/src/routes/review.py``. The field is not equivalent to the API's
``balanced`` flag or passing our structural QC.

Small live API queries succeeded, including reaction 1 with ``updated=false``.
Complete lightweight metadata retrieval from the read-only search endpoint
``POST /api/query`` with ``type=name`` and an empty query returned HTTP 504
after approximately 72 seconds. Earlier bulk requests also timed out. The
current complete curation count therefore remains unverified. Missing metadata
is recorded as unknown, never as a false curation flag.

Usable computational references
-------------------------------

The archive is streamed through ``scripts/audit_metamdb_aam.py``. Selection
requires RDKit parsing/sanitization, concrete elements and unique conserved
heavy-atom correspondences. Exact mapped spectators are removed; entries with
no active reaction are excluded. Proton participants are retained, but their
map labels are cleared because the assessment covers heavy atoms. Hydrogen
and charge balance are metadata, rather than repaired structures. No source
atom mapping is recomputed.

Results under RDKit 2026.03.1:

.. list-table::
   :header-rows: 1

   * - Category
     - Source records
   * - Usable heavy-atom mapping candidates
     - 27,451
   * - Parse or sanitization failed
     - 8,661
   * - Generic or dummy atoms
     - 6,566
   * - Heavy-atom maps not conserved, missing or duplicated
     - 6,845
   * - Unchanged after spectator removal
     - 405

The usable records represent 26,739 distinct directed canonical unmapped
reactions and 26,944 distinct normalized mapped SMILES. These are string-based
counts, not full ITS-isomorphism clustering. All original memberships remain.
There are 9,423 usable records with different explicit-H map inventories;
this does not invalidate the conserved heavy-atom mapping by itself.

The archived inspection output is
``Data/archives/metamdb/full-qc-27451-1375c16b/assessment/``:

* ``usable_candidates.csv.gz`` contains the 27,451 computational reference
  candidates, endpoints, source pointers and QC/provenance fields.
* ``rejected.csv.gz`` contains the 22,477 exclusions and their reasons.
* ``human_curated_references.csv.gz`` requires an upstream manual-review flag
  and the exact reviewed RXN payload matching the archive. Without supplied
  metadata it has no rows: that means no references established by this audit,
  not that MetAMDB contains no manually curated mappings.
* ``summary.json`` records source/code hashes, counts and metadata coverage.
* With ``--brmapper-root``, ``brmapper-source-comparison.json`` records the
  released table sizes, columns and ID-set comparison. The separate
  ``ring_author_ground_truth.csv.gz`` preserves the authors' 177 native GT rows
  without changing their source columns or mappings.

Passing QC does not certify the chemical mechanism. Computational candidates
can serve as source-derived targets for development or training, with their
provenance retained. They do not provide independent gold labels for evaluating
RDT. Human-curated references and computational candidates remain distinct.

Reproduce the assessment
-------------------------

.. code-block:: bash

   python scripts/audit_metamdb_aam.py \
     --archive atom_mappings.zip \
     --brmapper-root Data/raw/metamdb/brmapper

Saved MetAMDB API reaction metadata may be supplied with ``--metadata``. An
``updated=true`` flag alone does not prove that an older ZIP contains the
reviewed mapping: the metadata must also include a matching ``rxnFile``.
The count remains unknown when curation metadata is incomplete. These
assessment files are not added to the selected SynRXN AAM registry.

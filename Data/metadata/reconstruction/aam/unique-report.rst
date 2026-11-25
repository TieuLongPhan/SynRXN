Unique raw AAM reference results
=================================

The common aam.csv.gz now contains all 79,670 retained unique reactions.
The former 5,666 review inputs are included with diagnostic annotations.
Raw source mapped strings, partial map labels and original QC metadata remain
unchanged. No generated completion is adopted. The schema is synrxn.aam-unique.v5.

The dated cleaning history and per-dataset segmentation are saved under
cleaning-checkpoint-2026-10-27/. It records the earlier 91,581 to 79,670 count
reconciliation and both overlapping source memberships and disjoint chosen
reference counts. Reproduce that report with scripts/report_aam_cleaning.py.
Further cleaning is deferred for later review; this checkpoint changes no data.

Deduplication and agreed exclusions
------------------------------------

The 101,480 original rows yield 81,597 stereo-free directed inputs before
exclusions. The same 1,047 identical-endpoint inputs and 880 certified heavy-atom
mapping-conflict inputs remain excluded. All three exclusion payloads are byte
identical to the prior version. No conflict winner is selected.

The retained source cohort has 95,216 original records. Merging 15,546 repeated
records yields 79,670 unique retained inputs. Active and excluded provenance
still cover every one of the 101,480 original records exactly once.

Retained selected-reference scopes
------------------------------------

.. list-table:: Disjoint mapping scopes
   :header-rows: 1

   * - Mapping scope
     - Reactions
   * - heavy_atoms
     - 74,766
   * - heavy_atom_equivalence_classes
     - 325
   * - partial_heavy_atoms
     - 4,484
   * - partial_heavy_atom_equivalence_classes
     - 1
   * - unverified_heavy_atoms
     - 94

Flags are annotations
-----------------------

diagnostics.csv.gz annotates 5,666 inputs already present in aam.csv.gz.
It is not an exclusion or holdout list. The mutually exclusive primary
annotations are:

.. list-table:: Primary diagnostic annotation
   :header-rows: 1

   * - Annotation
     - Retained inputs
   * - partial_mapping
     - 4,485
   * - import_qc_flag
     - 1,086
   * - unverifiable_reference
     - 95

Annotations can also overlap: 4,574 groups have a partial source reference,
1,232 have a local import-QC flag, and 95 contain an unverifiable source
reference. Of the latter, one group has a comparable source target that is
selected instead. The other 94 selected targets retain explicit unverified
scope: 92 input/reference endpoint differences and two mixed-element mapping
classes. Their raw targets remain included; no ITS equivalence is asserted.

All six H-only groups (12 FlowER train records) remain included, with their
raw proton-transfer, MC, step and parent history preserved. Local SIHUMIx QC
labels are importer annotations, separate from upstream curation declarations.

Source memberships
--------------------

.. list-table:: Memberships overlap across sources
   :header-rows: 1

   * - Source
     - Original rows
     - Retained unique inputs
   * - brenda
     - 3,138
     - 2,309
   * - ecoli
     - 273
     - 211
   * - ehreact
     - 939
     - 738
   * - enzyme_map
     - 47,974
     - 37,252
   * - flower
     - 34,815
     - 33,116
   * - golden
     - 1,758
     - 1,758
   * - metambd_cyclic
     - 177
     - 117
   * - natcomm
     - 491
     - 491
   * - recon3d
     - 382
     - 350
   * - sihumix
     - 8,533
     - 1,772
   * - uspto_3k
     - 3,000
     - 3,000

Source memberships cannot be summed to obtain the unique pool total.

Reproduction and verification
-------------------------------

Reproduce with scripts/build_unique_aam.py using the pinned overlap audit.
The normal source builder remains scripts/build_aam.py. See unique-method.rst
and doc/aam_reproduction.rst for commands, scope and file definitions.

The previous filtered pool, its diagnostic outputs and source-code snapshot
are archived under
Data/archives/aam_review/before-raw-retention-policy-2026-10-27/.
The completion experiment remains archived separately and is not selected.

Verification confirms that the prior 74,004 selected mapped targets and
their source identities are unchanged; the added IDs exactly match the former
5,666 review inputs. Selected targets are checked verbatim against source rows.
All original source records, payload hashes, reference joins, flags, unique
identities and agreed exclusions are verified. The 73 AAM tests and three
subtests passed; current completion/retention tests, lint and documentation
checks also passed. No new split or human-validation claim is made.

AAM cleaning checkpoint and dataset segmentation
================================================

Recorded 2026-10-27. Further cleaning is deferred for later review.
This checkpoint does not change source data or the selected pool.

Current decisions
-------------------

* Keep raw source maps, including partial and equivalent-atom mappings.
* Keep local QC flags and unverified references as annotations.
* Merge duplicate canonical reactions while preserving all source records.
* Exclude identical endpoints and certified conflicting heavy-atom mappings.
* Ignore stereo for reaction identity and ITS comparison; retain raw stereo in provenance.
* Adopt no computed completion and assign no new train/validation/test split.

Cleaning history
------------------

.. list-table:: Unique directed input counts
   :header-rows: 1

   * - Stage
     - Change
     - Remaining
   * - Earlier stereo-sensitive unique inputs
     - 0
     - 91,581
   * - Additional merging when stereo is ignored
     - -9,984
     - 81,597
   * - Exclude identical endpoints
     - -1,047
     - 80,550
   * - Exclude conflicting heavy-atom mappings
     - -880
     - 79,670

Source-row accounting
-----------------------

The history above counts unique reaction groups. The following accounting
starts from original source rows, including repeated observations. Each
excluded reaction group can contain several original rows.

.. list-table:: All changes measured in source-row counts
   :header-rows: 1

   * - Action
     - Rows
   * - Original source rows
     - 101,480
   * - Exclude rows in unchanged-reaction groups
     - 2,833
   * - Exclude rows in conflicting-mapping groups
     - 3,431
   * - Merge repeated rows within each source
     - 14,102
   * - Merge shared reactions across sources
     - 1,444
   * - Selected unique reactions
     - 79,670

There are still 101,480 original source records.
The exclusions cover 6,264 source rows;
95,216 source rows remain, merging into
79,670 selected unique reactions. All originals remain
in active or excluded provenance. The latest policy added back 5,666
previously held-out inputs: 74,004 to 79,670. The excluded payloads
and prior selected raw targets were unchanged in that policy update.

Dataset segmentation
----------------------

.. list-table:: Source memberships and chosen representatives
   :header-rows: 1

   * - Dataset
     - Original rows
     - Retained unique source reactions
     - Chosen pool rows
   * - brenda
     - 3,138
     - 2,309
     - 2,264
   * - ecoli
     - 273
     - 211
     - 209
   * - ehreact
     - 939
     - 738
     - 738
   * - enzyme_map
     - 47,974
     - 37,252
     - 36,459
   * - flower
     - 34,815
     - 33,116
     - 33,116
   * - golden
     - 1,758
     - 1,758
     - 1,756
   * - metambd_cyclic
     - 177
     - 117
     - 116
   * - natcomm
     - 491
     - 491
     - 0
   * - recon3d
     - 382
     - 350
     - 328
   * - sihumix
     - 8,533
     - 1,772
     - 1,685
   * - uspto_3k
     - 3,000
     - 3,000
     - 2,999

.. list-table:: Per-dataset source-row reconciliation
   :header-rows: 1

   * - Dataset
     - Original rows
     - Unchanged rows
     - Conflicting rows
     - Repeats within source
     - Represented through another source
     - Selected rows
   * - brenda
     - 3,138
     - 93
     - 610
     - 126
     - 45
     - 2,264
   * - ecoli
     - 273
     - 2
     - 4
     - 56
     - 2
     - 209
   * - ehreact
     - 939
     - 0
     - 3
     - 198
     - 0
     - 738
   * - enzyme_map
     - 47,974
     - 832
     - 1,900
     - 7,990
     - 793
     - 36,459
   * - flower
     - 34,815
     - 1
     - 18
     - 1,680
     - 0
     - 33,116
   * - golden
     - 1,758
     - 0
     - 0
     - 0
     - 2
     - 1,756
   * - metambd_cyclic
     - 177
     - 0
     - 46
     - 14
     - 1
     - 116
   * - natcomm
     - 491
     - 0
     - 0
     - 0
     - 491
     - 0
   * - recon3d
     - 382
     - 8
     - 3
     - 21
     - 22
     - 328
   * - sihumix
     - 8,533
     - 1,897
     - 847
     - 4,017
     - 87
     - 1,685
   * - uspto_3k
     - 3,000
     - 0
     - 0
     - 0
     - 1
     - 2,999

Retained source-reaction counts overlap across datasets. Chosen pool rows
count the selected raw-reference source once per global input and sum to
79,670. A source with zero chosen representatives still has its retained
reactions and original records linked through members and provenance.

dataset-segments.csv includes source-row exclusions, repeated rows,
representatives supplied by another source and selected mapping scopes.

Reproduction and saved evidence
--------------------------------

Run from the repository root into a fresh report directory::

   python scripts/report_aam_cleaning.py --output /tmp/aam-cleaning-checkpoint-new

checkpoint.json pins the source pool, baseline audit, policies and counts.
manifest.json pins this report and its CSV/JSON payloads.
Current processing is documented in unique-method.rst; the current data
builder is scripts/build_unique_aam.py. Previous versions and the completion
experiment remain under Data/archives/aam_review/. No data is changed by
this reporting command.

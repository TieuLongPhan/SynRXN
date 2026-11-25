AAM overlap and ITS results
============================

Audited 101,480 records across 11 active datasets.
Train/test/validation roles were ignored for comparison. Source tables are unchanged.

There are 81,597 distinct directed canonical reaction inputs,
10,278 repeated-input groups and 2,151
groups overlapping across sources. 885 groups contain
certified incompatible mappings; 681 have cross-source
disagreements. 2 repeated groups also contain
unverifiable members. A disagreement is not a verdict about chemical correctness.

.. list-table:: Exact directed overlap
   :header-rows: 1

   * - Source pair
     - Shared inputs
     - Comparable
     - Different ITS
     - No common ITS
   * - brenda / enzyme_map
     - 1168
     - 1168
     - 516
     - 496
   * - enzyme_map / sihumix
     - 498
     - 498
     - 175
     - 156
   * - golden / natcomm
     - 491
     - 491
     - 0
     - 0
   * - brenda / sihumix
     - 157
     - 157
     - 64
     - 62
   * - ehreact / enzyme_map
     - 63
     - 63
     - 0
     - 0
   * - enzyme_map / metambd_cyclic
     - 32
     - 32
     - 21
     - 18
   * - ecoli / recon3d
     - 19
     - 18
     - 2
     - 2
   * - brenda / metambd_cyclic
     - 16
     - 16
     - 12
     - 10
   * - ecoli / sihumix
     - 8
     - 8
     - 2
     - 2
   * - metambd_cyclic / sihumix
     - 7
     - 7
     - 2
     - 0
   * - enzyme_map / recon3d
     - 5
     - 5
     - 0
     - 0
   * - ecoli / enzyme_map
     - 4
     - 4
     - 1
     - 1
   * - brenda / recon3d
     - 4
     - 4
     - 1
     - 1
   * - recon3d / sihumix
     - 4
     - 4
     - 0
     - 0
   * - brenda / ecoli
     - 2
     - 2
     - 0
     - 0
   * - ecoli / metambd_cyclic
     - 2
     - 2
     - 2
     - 1
   * - enzyme_map / golden
     - 2
     - 2
     - 0
     - 0
   * - golden / uspto_3k
     - 1
     - 1
     - 0
     - 0
   * - natcomm / uspto_3k
     - 1
     - 1
     - 0
     - 0
   * - enzyme_map / natcomm
     - 1
     - 1
     - 0
     - 0
   * - brenda / ehreact
     - 1
     - 1
     - 1
     - 1

Ring references
----------------

The ring dataset retains 177 first-reference records representing
145 distinct exact inputs. 36
inputs overlap another source; 36 are comparable,
and 27 have at least one ITS disagreement.
See ring_review.csv for every ring record, native ID and comparison status.

Reproduction and interpretation
--------------------------------

The pinned first-reference build is documented in Data/metadata/reconstruction/metamdb/processing.rst.
The overlap method and partial-ITS/H/stereo treatment are in overlap-method.rst.
Run scripts/audit_aam_overlap.py into a fresh folder, then scripts/report_aam_overlap.py.
Exact identities ignore stereochemistry and preserve tautomers, protonation and isotopes.
Other audit modes remove unchanged spectators and optionally allow reversal.
No records were deleted, deduplicated, reassigned a reference, or given new splits.
FlowER MC/pathway identity remains separate from endpoint/ITS equivalence.

Archived partial AAM completion experiment
===========================================

Status: archived after the decision to keep raw source atom mappings. No
generated labels or computed completion was adopted into the source tables or
the accepted unique benchmark. The normal build pipeline retains raw maps.

Reproduce and verify from the installed unique pool::

   python scripts/complete_aam_maps.py --workers 4 --batch-size 64 --output /tmp/aam-completion-new
   python scripts/complete_aam_maps.py --output /tmp/aam-completion-new --verify

The archived output is under
Data/archives/aam_review/completion-experiment-2026-10-27/outputs/.
The selected inputs are the 4,485 distinct reactions whose primary review
reason is partial_mapping. Inputs with endpoint discrepancies or invalid mixed
mapping classes retain their separate review scope and are not repaired here.

Results
--------

* 149 inputs receive 1,521 previously absent heavy-atom labels. Each new ID is
  unique across both endpoints; it does not invent an atom correspondence.
* 4,366 inputs have unequal heavy-element/isotope inventories. Both endpoints
  cannot share a complete heavy-atom map without changing the recorded reaction.
* 119 inputs have equal heavy inventories. SynKit ITSExpand produces one
  verified closed completion; 118 have no certified completion under this
  method. The latter result is a method limitation, not a proof of impossibility.
* The one completed reference originates from sihumix:Bproducta:R_2ACLMM. Its
  original local qc_failed annotation is retained. It is not automatically
  promoted into the accepted benchmark.

For this case, the local flag is invalid_heavy_atom_mapping: reactant methyl
label 2 has product label 1 instead of the same label. The strict importer
requires closed, bijective map-label pools. The original reaction passes its
element, hydrogen and charge balance checks. This flag describes partial map
coverage under that importer rule, rather than proving a chemically incorrect
correspondence. Its upstream source label remains User-curated.

At the time of this experiment, the selected subset contained 74,004 reactions
and the review inventory contained 5,666. These are historical counts before the
raw-retention policy; see unique-report.rst for current retained-table counts.
The experiment did not change source CSVs or unique-pool payloads.

Meaning of the outputs
-----------------------

labelled.csv.gz contains 4,485 fully labelled heavy-atom references whose map
pools may remain partial. completed.csv.gz contains the one generated closed
map, with reference_tier=computational_completion. Both retain the common AAM
columns and original source review reasons. attempts.csv.gz retains the original
mapped string, derived strings, per-input outcome and reconstruction evidence.
summary.json and manifest.json pin the input pool and processing implementation,
including installed SynKit source files and dependency versions.

Completion preserves each unmapped endpoint independently, including fragment
multiplicity, isotope, charge and hydrogen counts; stereo remains outside the
comparison scope. Both reconstruction directions are attempted, and distinct
valid ITS candidates cause review. Every accepted candidate must be a complete
individual heavy map that refines every contributing source reference's known
correspondences. Unknown one-sided IDs may be reassigned in the derived result;
paired source IDs anchor reconstruction. Repeated mapping classes are not sent
through an adapter that would collapse atoms with repeated IDs.

Generated completion example
-----------------------------

Original::

   [CH3:2][C:6]([C@:4]([CH3:3])([C:5](=[O:8])[O-:9])[OH:7])=[O:10]>>[CH3:1][C:6]([CH3:3])([C:4]([C:5](=[O:8])[O-:9])=[O:7])[OH:10]

Completed::

   [CH3:1][C:6]([C:4]([CH3:3])([C:5](=[O:8])[O-:9])[OH:7])=[O:10]>>[CH3:1][C:6]([CH3:3])([C:4]([C:5](=[O:8])[O-:9])=[O:7])[OH:10]

The left methyl's formerly one-sided label 2 is paired with right label 1.
The already specified correspondences for labels 3 through 10 are preserved.
This is a computed proposal, not a new human-curation assertion.

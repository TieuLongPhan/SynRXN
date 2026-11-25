Unique raw AAM reference pool
==============================

The selected policy keeps raw source atom mappings. After canonical reaction
deduplication, only identical endpoints and certified incompatible heavy-atom
mapping constraints exclude an input. Partial maps, equivalent-atom classes,
explicit-H alternatives, local import QC flags and unverified references remain
in the common aam.csv.gz table with annotations. No completion is adopted.

Reproduction
-------------

From the repository root, using fresh output directories::

   python scripts/audit_aam_overlap.py --workers 4 --output /tmp/aam-overlap-new
   python scripts/build_unique_aam.py --audit /tmp/aam-overlap-new --workers 4 --batch-size 256 --output /tmp/aam-unique-new
   python scripts/build_unique_aam.py --audit /tmp/aam-overlap-new --output /tmp/aam-unique-new --verify

The installed source tables, audit and pool are under
Data/tables/development/aam/, aam_overlap/ and aam_unique/, respectively.
Use python scripts/build_unique_aam.py --verify for the installed pool.
Registry, source, audit, chemistry implementation and dependency pins are checked
before reuse. Processing uses bounded batches and a disk-backed workspace.
Gzip headers and row ordering are deterministic; runtime is informational.
The processing schema is synrxn.aam-unique.v5.

Identity, exclusions and raw preservation
------------------------------------------

An input is the directed canonical unmapped reaction, ignoring stereochemistry
and retaining isotopes, charges, hydrogen counts, tautomers and multiplicity.
Each retained input appears exactly once in aam.csv.gz. Original source roles
are provenance; no new train, validation or test split is assigned.

Identical canonical endpoints are excluded. Certified incompatible heavy-atom
references exclude the entire duplicate group without choosing a winner.
A broad equivalence class cannot hide incompatible individual mappings.
Unresolved comparisons do not establish a conflict and do not remove the input.
The exclusion ledger preserves all removed source records and their reasons.
Duplicate rows are merged while every original source record remains available.

The selected mapped target is copied verbatim from an original source record.
No missing labels are filled, unmatched atoms paired, endpoints rewritten or
map labels renumbered in that target. A source-reference selection first prefers
a comparable common/refining representation. Within that scope it prefers
complete individual maps, complete class-valued maps, fewer local import flags,
explicit curation declarations and deterministic source-name/row ties.
Dataset size does not determine the representative.

If no verifiable ITS class exists, one raw reference is still retained with a
:raw reference ID and mapping_scope=unverified_heavy_atoms. That fallback does
not assert equivalence between unverifiable source mappings. A comparable
reference is preferred over an unverified source alternative when available.

Annotations and mapping scope
------------------------------

quality_flags records partial_mapping, import_qc_flag, unverifiable_reference
and mapping_constraints_unresolved where applicable. These flags annotate
retained rows; none is an additional exclusion rule. diagnostics.csv.gz contains
the annotated subset of aam.csv.gz inputs, not a holdout or removal queue.
reference_comparable describes the selected target's graph/endpoint check;
retention does not assert technical correctness or human validation.

The target includes mapping_status, comparison_status and the original
source_validation_status. Source rxn and ground_truth fields that still differ
after the same canonicalization are retained with an explicit
input_target_endpoint_mismatch annotation. Both original fields remain in
provenance; species such as [Na] and [NaH] or S and [S] are not equated.
See endpoint-mismatch-inspection.csv for the earlier detailed inspection.

mapping_scope distinguishes heavy_atoms, heavy_atom_equivalence_classes,
partial_heavy_atoms, partial_heavy_atom_equivalence_classes and
unverified_heavy_atoms. Repeated labels are sets of allowed correspondences;
endpoint atoms are never collapsed into a single atom. Classes mixing elements
or isotopes cannot be assigned a verified ITS class, but their raw reference
remains annotated in the retained table.

Partial references specify only their known correspondences. One-sided labels
and missing labels remain unchanged. A partial ITS can retain endpoint-only
atoms; it does not supply the unknown partners. Complete candidates can be
checked against known source constraints with aam_mapping_constraints.compare
using refine=True. Unverified targets have no certified ITS equivalence.
Explicit-H-only differences remain available as proton-tracking/MC diagnostics
and do not affect heavy-atom retention. Stereo is outside this comparison scope.

Origin of local QC flags
--------------------------

The SIHUMIx importer computes full element, hydrogen, charge and strict mapping
checks in sihumix_quality.py. sihumix.py translates them into qc_failed and
reference_excluded_* annotations, also using transport/identity model metadata.
These local flags are distinct from upstream curation and matching-source labels.
They are preserved as metadata and no longer remove an input from aam.csv.gz.
The older invalid_heavy_atom_mapping flag can describe a partial map, duplicate
class labels or an inconsistent element/isotope correspondence; its meaning
must be interpreted alongside the current mapping_status and comparison_status.

The CanonRSMI diagnostics and completion experiment are historical inspections.
See review-diagnostics.rst and completion-report.rst for archived inputs, recipes
and results. Their generated labels and computed correspondences are not used
as selected references. The pre-policy pool and scripts are preserved under
Data/archives/aam_review/before-raw-retention-policy-2026-10-27/.

Files and verification
------------------------

Under Data/tables/development/aam_unique/:

* aam.csv.gz has one original source target for every retained unique input.
  Its first four fields remain r_id, ground_truth, rxn and original_id.
* reactions.csv.gz is the matching input inventory with reference IDs and flags.
* diagnostics.csv.gz lists flagged retained inputs; it overlaps aam.csv.gz.
* references.csv.gz records ITS-class representatives and unverified raw fallback
  references. Members retain raw constraint classes, equivalence decisions,
  original QC annotations and comparison errors.
* members.csv.gz and provenance.jsonl.gz retain every active source record,
  including mapped strings, roles, FlowER step/MC/parent history and source QC.
* excluded_reactions.csv.gz, excluded_members.csv.gz and
  excluded_provenance.jsonl.gz preserve the agreed exclusions and all source rows.
* hydrogen_conflicts.csv.gz retains explicit-H alternatives and mechanism history.
* sources.json retains the source registry; summary.json and manifest.json pin
  counts, policies, implementation and payloads. Applicable source terms remain.

Verification checks unique inputs, complete retained-table coverage, diagnostic
annotations, exclusions, reference joins, verbatim selected source targets and
original QC metadata. Active plus excluded provenance covers every original
source record exactly once. Tests exercise partial and class-valued targets,
unverified fallbacks, source preference, unchanged raw strings, certified
conflicts, timeouts and provenance preservation.

split_group_id groups spectator-normalized reactions in either direction for
later partitioning. Shared FlowER parents and known source or training-set
ancestry also matter for that decision. Unique inputs remove duplicate weighting
without asserting independence from mapper training data or between sources.

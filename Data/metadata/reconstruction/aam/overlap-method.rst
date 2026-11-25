AAM overlap and mapping-constraint audit
========================================

Scope and canonicalization
---------------------------

All 101,480 selected source records across 11 datasets are inspected. Only the
177-row cyclic MetAMDB collection is selected; the paper's unavailable larger
subset remains pending. Original split roles are provenance, not grouping keys.
Original mapped strings and source tables are preserved.

Exact identity uses directed canonical unmapped endpoints after removing
stereochemistry and removable ordinary explicit hydrogen atoms. Components are
sorted with multiplicity. Isotopes, charge, ordinary hydrogen counts,
protonation and tautomers remain distinct. Agents in the middle reaction field
are outside endpoint identity. Ignoring stereo is deliberate for this AAM task;
it does not alter the stored original mapped reference.

The input rxn and ground_truth fields are independently canonicalized using the
same rules. A remaining endpoint mismatch is not merely SMILES atom ordering.
For example, [Na] differs from [NaH], and S differs from [S]. These chemical or
hydrogen-representation differences remain visible for review. They are not
silently repaired by changing a source structure. The inspection CSV
endpoint-mismatch-inspection.csv records the prior 95 cases: 3 are resolved by
ignoring stereo, while 92 source records retain other differences.

The normalized mode additionally cancels common whole-component spectators.
A reference loses only certified mapped-identical components whose heavy-atom
mapping labels do not extend into active components. In particular, an
unchanged component sharing an equivalence class with active atoms is not
silently discarded. The resulting reference endpoints must agree with the
normalized input. The undirected mode also permits reaction reversal.
Fully unchanged reactions retain their identity instead of becoming empty.

Mapping representations
------------------------

A positive unique label specifies one atom's correspondence. For these records,
the heavy ITS stores paired endpoint states and bond types, including charge,
radicals, aromaticity, H counts and isotope decorations, while ignoring stereo.
The map numbers themselves are excluded from graph-equivalence attributes.

Repeated labels are accepted as source equivalence/mapping classes. Every
endpoint atom remains a separate graph node. A class node connects the atoms
whose cross-endpoint correspondences are allowed to vary. Atoms are never
collapsed into a single node because their labels repeat. Classes must retain
the same element and isotope; a class mixing different elements/isotopes
remains invalid. Balanced classes without missing labels are complete
class-valued references, not bijective encodings requiring invented numbering.

Missing/zero labels and one-sided mapped atoms are partial mappings. Each
unmapped atom receives a separate endpoint-only internal graph identity;
no missing correspondence is invented. These records enter the partial-AAM
review category. The source strings remain unchanged.

Why repeated labels can be intentional is documented in the original Recon3D
study: chemically equivalent atoms may share one map number, with matched
labels propagated between endpoints. This can connect different local contexts
on the opposite endpoint through reaction/resonance equivalence. A symmetry
rank on one endpoint alone is therefore not a valid rejection rule.
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5471289/

Equivalence, compatibility and conflict
---------------------------------------

WL hashes form candidate buckets, and exact labeled graph isomorphism certifies
identical mapping representations. Distinct complete bijective ITS classes
are conflicting atom correspondences.

Class-valued and partial references represent sets of specified constraints.
A bounded endpoint-graph isomorphism search checks whether two references admit
a common completion of their specified correspondences. Balanced shared
mapping classes constrain that feasibility test; unobserved partial
correspondences remain unknown. The feasibility calculation does not create
or store a replacement source mapping.

A complete reference can serve as the common representative only when its
allowed mappings are certified to refine every other comparable reference.
The representative is restricted to that certified class. Thus a broad
class-valued reference cannot mask two conflicting fully specified maps.
If incompatibility is proved, the reaction is a mapping conflict. If a search
times out or no common refining reference can be certified, it stays unresolved
for review. Compatibility alone is not treated as a transitive equivalence.
Comparisons have a five-second limit and use no permutation-expansion dataset.

Explicit-H mapping differences are separate diagnostic annotations and do not
block heavy-atom eligibility. They are compared within matching represented-H
endpoints and exact heavy-constraint classes. Original source paths and H
correspondences remain available for mechanism-specific analyses.

Reproduction and outputs
-------------------------

Run into fresh directories::

   python scripts/audit_aam_overlap.py --workers 4 --batch-size 256 --output /tmp/aam-overlap-new
   python scripts/report_aam_overlap.py --audit /tmp/aam-overlap-new --output /tmp/aam-review-new --report /tmp/aam-overlap-report.rst
   python scripts/build_unique_aam.py --audit /tmp/aam-overlap-new --workers 4 --batch-size 256 --output /tmp/aam-unique-new

The audit writes summary.json, groups.csv.gz, members.csv.gz, pairwise.csv,
conflicts.jsonl.gz, record_qc.csv.gz and a disk-backed overlap.sqlite3.
Source hashes, implementation pins and dependency versions are recorded.
confirmed_mapping_conflict distinguishes proved incompatibility from an
unresolved comparison of class-valued constraints. Memberships preserve raw
constraint classes, compatible/incompatible class IDs and the certified anchor.

The source tables are not deduplicated or resplit by the audit. The subsequent
unique builder applies the user-approved exclusions and reference policies
in unique-method.rst. Endpoint equivalence remains separate from FlowER MC
identity and mechanistic history.

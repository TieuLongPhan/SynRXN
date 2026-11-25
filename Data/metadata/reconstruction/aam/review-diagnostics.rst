AAM review provenance and CanonRSMI trial
========================================

Historical inspection before the raw-retention policy. The former review and
source_qc_excluded categories below are not current exclusion rules. Those
inputs are now retained with annotations; see unique-report.rst. The original
pool and diagnostic outputs are archived under
Data/archives/aam_review/before-raw-retention-policy-2026-10-27/.

Import QC origin
----------------

All 1,086 current source_qc_excluded inputs come from locally generated SIHUMIx
import annotations. They are not author-issued rejection labels. The importer
sihumix_quality.py computes full element (including H), charge and strict
mapping checks; sihumix.py turns these into validation_status values and adds
transport/identity flags from model metadata. The curation/mapping origin fields
remain separate from those local mechanical flags.

There are 1,049 inputs with qc_failed and 37 flagged only as transport. All
3,658 contributing flagged source records have balanced heavy-element pools.
The remaining element-imbalance flags are therefore hydrogen-related. Across
these inputs, 1,034 carry an element/H imbalance flag and 1,041 a charge flag;
these counts overlap. Transport alone is a model annotation, not proof of an
incorrect atom correspondence. This gate is a conservative import policy and
is separate from the current complete, compatible heavy-atom mapping checks.
No eligibility decision was changed during this diagnostic inspection.

Partial mapping detail
----------------------

The current 4,485 partial-review inputs have these mutually exclusive patterns:

.. list-table:: Partial input categories
   :header-rows: 1

   * - Pattern
     - Inputs
   * - Unlabeled heavy atoms on reactants only
     - 149
   * - All atoms labeled; extra map labels on reactants only
     - 4,206
   * - All atoms labeled; extra map labels on products only
     - 11
   * - All atoms labeled; unpaired map labels on both endpoints
     - 119

There are no genuinely unlabeled product atoms in this review cohort. One input
has source variants in two patterns; the unlabeled-reactant pattern takes
priority so the table sums exactly to 4,485. There are 4,366 inputs with unequal
heavy-element inventories, and 119 with balanced heavy-element inventories but
incomplete cross-endpoint label correspondence.

For example, Golden golden_1255 records::

   [CH3:1][C:2]([CH3:3])=[O:4]>>[CH3:1][CH2:2][CH3:3]

Every atom already has a map label, but no oxygen-containing product is recorded.
Reindexing labels cannot supply that missing counterpart. This is partial with
respect to a closed atom-pool requirement, rather than an unlabeled SMILES.

CanonRSMI results
----------------

Tested installed SynKit 1.6.2 with the WL backend, three iterations and a
five-second limit per input. Stereo was removed on a working copy consistently
with the current AAM scope; original ground-truth strings were preserved.
All 4,485 inputs were tested, using bounded batches and four workers.

CanonRSMI assigned positive labels to all unlabeled reactant atoms in the 149
blank-label cases. All 149 nevertheless retained an incomplete endpoint map
pool. Overall, 4,477 outputs remain partial, seven attempts do not serialize a
parseable reaction, and one changes an endpoint and fails the preservation
check. No trial became a complete endpoint-preserving mapping. No trial output
was substituted for an original reference.

The one endpoint-change case contains source equivalence labels that span
multiple sulfur-containing participants; treating a map label as one graph-node
identity loses source structure. The current equivalence-aware AAM processor
keeps those atoms distinct and does not use CanonRSMI for such a replacement.
The balanced 119-label cases require establishing the missing correspondences,
which is different from reindexing the labels already specified.

During testing, the local reaction parser was fixed to distinguish a dative
bond -> from a reaction separator. A formerly apparent Canon failure was valid
dative-bond output. All 202,960 original source rxn/ground_truth strings were
checked: their tokenization is unchanged by this parser fix. The dataset CSV,
JSONL and SQLite payloads are unchanged; code pins were refreshed with that
proof, and all 64 tests plus three subtests passed.

Reproduction and files
----------------------

Run into a fresh directory::

   python scripts/audit_aam_partial_canon.py --workers 4 --batch-size 64 --output /tmp/aam-review-diagnostics-new

The archived output is under
Data/archives/aam_review/before-raw-retention-policy-2026-10-27/aam_review_diagnostics/.
partial_cases.csv.gz contains source record IDs, original strings and sidedness.
canon_trials.csv.gz stores the stereo-free working copies, complete CanonRSMI
outputs, statuses, endpoint checks and failures. import_qc.csv.gz traces every
flagged input. summary.json and manifest.json pin inputs, code and versions.

The original-stereo control and the pre-parser-fix trial are retained under
Data/archives/aam_review/ for inspection. The final stereo-free, corrected-parser
trial is the basis of the counts above. See dative-parser-verification.json for
source-tokenization and unchanged-payload evidence.

CanonRSMI documentation:
https://synkit.readthedocs.io/en/stable/chem.html#canonicalization

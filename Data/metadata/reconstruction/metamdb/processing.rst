Active MetAMDB AAM preparation
===============================

The requested selection is the paper's 4,456 MetAMDB reactions plus 177 cyclic
reactions. The exact 4,456 subset/ID list is absent from the public release and
relevant local downloads. It is pending, rather than reconstructed by selecting
an arbitrary number of full-archive records. The available cyclic source uses
``scripts/build_metamdb_aam.py``. Its shared export command is::

   python scripts/build_aam.py --datasets metambd_cyclic --rebuild-sources --fetch

The active output is ``Data/tables/development/aam/metambd_cyclic/`` (177 rows,
141 train and 36 test). Every record has one reference. No row deduplication
is applied. The older cyclic name remains for compatibility. Roles are retained
as provenance; splitting will be decided later.

The retired full-archive preparation used the public 49,928-member RXN archive, SHA-256
``b460f3ec5f842f757e35ad9eb0df0e5edf3450c289c272b2bfdcfb88c3689616``:
https://metamdb.tu-bs.de/img/atom_mappings.zip

Its 27,451-row QC export, native bundle and assessment are now archived under
``Data/archives/metamdb/full-qc-27451-1375c16b/`` and excluded from the active
AAM registry. Requesting the ``metamdb`` paper dataset fails with a missing
exact-source error. The pending record is
``Data/metadata/reconstruction/metamdb/paper-subset-status.json``.

The cyclic input is the public BRMapper ``ringreactions_gt.csv`` at commit
``b1d428f3ed2d1ef5a1311150cc335ba1a916c06a``, SHA-256
``54d8f4d6f4598bee6acd93df54d2136d2f412b5cc551c50271069c8c9c2cd20f``.
The processor reproduces the upstream cluster-stratified 80/20 split with
``random_state=42`` and the first variant from the upstream product-tautomer
and Kekule expansion. It does not generate a Cartesian product of variants.
All 177 first references match the supplied train/test archive after the same
normalization. That local archive is now a historical comparison witness, not
a required processing input.

Mapped source correspondences are retained, exact mapped spectator components
are removed, and the unmapped input is canonicalized with removable H atoms
made implicit. Explicit-H correspondence is not claimed: H map labels are
cleared, while proton participants remain. Structural/mapping QC exclusions
are written to ``rejected.csv.gz`` and every retained row has source provenance.

These are source-derived reference records. The cyclic data remain qualified as
``human_review_unverified`` and do not assert that they recover the paper's
10,579 validated mappings. ``ground_truth`` follows the established AAM target
schema, while ``reference_tier`` records the reference origin. Existing
unverified redistribution qualifications are retained; this is a local
development export, not a new public release.

Native source bundles, reports and manifests are under
``Data/raw/aam/<dataset>/native-product/``. The selected source locks and code
pins are in ``Data/metadata/sources/aam/`` and the shared AAM registry.
``assessment.rst`` and ``curation-evidence.json`` record the source inspection
and distinction between author-provided targets and verified manual review.

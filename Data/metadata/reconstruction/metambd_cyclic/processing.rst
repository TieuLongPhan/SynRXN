MetaMBD cyclic AAM processing
============================

Run from the repository root::

    python scripts/process_metambd_cyclic.py --workers 2 --batch-size 16

This single standalone script reads both native splits from the preserved local
archive Data/raw/metambd_cyclic/ringreactions.zip. It processes bounded batches
of source cases with two worker processes, using the same normalization and
record builder for both splits.

The single combined output is
Data/tables/canonical/aam/metambd_cyclic.csv.gz. It follows the common AAM fields:

* r_id: unique dataset record identifier.
* ground_truth: normalized mapped first candidate, selected for compatibility.
* rxn: canonical isomeric reaction without atom maps, derived from the first
  candidate only, after removal of mapped-identical whole spectators.
* original_id: original split and one-based source row.
* split: train or test, preserving source provenance.
* ground_truth_variants: JSON list of retained mapped references; one in train,
  full ITS representatives from the original row in test.

There are 177 records: 141 train records with one selected reference each and
36 test records with 719 full ITS representatives in total. Source rows remain
separate, including cases with identical unmapped inputs. Different tautomer
candidates remain in their source case; they are not expanded into more rows.
No target is replaced by a canonical tautomer. First-candidate selection and
candidate equivalence do not assert independent chemical validation or imply
positive/negative correctness labels.

Candidate provenance
--------------------

Data/metadata/reconstruction/metambd_cyclic/candidates.jsonl.gz contains one
provenance record per table row, linked by r_id. It preserves all original
candidate strings and their order, including all 1,141 test strings. The raw
source hydrogen map labels are unchanged in these strings.

Each record includes its source archive member, native split and source row,
candidate_memberships, representative_source_columns and corresponding
its_ground_truth_variants. Memberships link source columns to retained
representatives and local ITS class IDs. Non-first train candidates are marked
not_selected; malformed strings are marked invalid_source with their reason.

Of 1,141 test strings, 1,140 are parseable and deduplicate to 719 representatives.
The malformed string at test row 19 column 120 remains in the sidecar. The two
malformed train strings at row 33 column 124 and row 126 column 125 also remain.
The source ZIP retains all 4,613 original candidate strings across both splits.
Invalid first candidates cause processing to fail without choosing replacements.

ITS equivalence and hydrogen handling
-------------------------------------

Deduplication is local to each test row and uses full SynKit ITS graphs,
core=False, with absolute paired element, charge, radical, aromaticity, hydrogen
count, lone-pair, isotope and presence attributes plus paired bond orders.
Atom-map numbers identify graph vertices but are excluded from matching labels.
WL hashes only provide buckets; exact structural and stereo-aware graph
isomorphism decides equality. The first source string in each class is retained.
The full graph retains unchanged context; this is not RC-only clustering.

Accepted references conserve unique positive heavy-atom map IDs and elements
across endpoints. Hydrogen map labels are cleared in normalized references
without guessing missing correspondences; the scope is heavy-atom AAM. Proton
participants remain in full inputs and references. H-only components are outside
the heavy-atom ITS projection, but their signatures supplement graph equality
so differing proton counts do not merge. No spectators were found in this ZIP.

Reproducibility and local loading
--------------------------------

processing-summary.json records source member hashes, source archive hash,
script hash, runtime versions, split counts, invalid coordinates and hashes of
the two generated data artifacts. Identical rebuilds retain the same CSV and
provenance bytes, including across worker and batch settings. Source licensing
is unknown, so the local catalog excludes the data from public redistribution.
The source record is Data/metadata/sources/aam/metambd_cyclic.json.

Load both splits or filter using the standard loader::

    from synrxn.data import DataLoader
    loader = DataLoader('aam', source='local', data_dir='Data', cache_dir=None)
    data = loader.load('metambd_cyclic')
    test = loader.load('metambd_cyclic', filters={'split': 'test'})

Superseded generated tables, reports, scripts and tests are preserved in the
single verified archive Data/archives/metambd_cyclic/legacy-processing.zip.
The previous split CSVs, RC samples and helper module are retired.

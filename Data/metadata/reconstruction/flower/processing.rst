FlowER reproducibility
======================

The sole raw input is Data/dumps/flower.dump.zip. It preserves curated raw
streams, frozen elementary selections, the verified completed reference data,
licenses and earlier source/code history. Original and curated source hashes
were checked before removing loose copies.

Export the standard AAM table with::

    python scripts/build_aam.py --datasets flower
    python scripts/build_aam.py --datasets flower --verify

The output is Data/tables/development/aam/flower/aam.csv.gz, with columns r_id,
ground_truth, rxn, original_id, steps and role. It contains 34,815 records:
23,367 train, 2,735 val and 8,713 test. These six frozen source fields match the
independently verified all-split RC/MC reference. There is no step cap; 15 is observed.

The common AAM export now appends checksum-pinned center/source annotations:
center_cluster_id, center_type, source_group, source_parent_group, source_lines,
source_seed_id, source_member and source_split. Global center IDs are aligned
across roles and lengths, including elementary RCs as two-state MCs. The current
34,815 rows have 24,410 global classes; targets and roles remain unchanged.

Reconstruct these derived metadata and exact class templates with::

    python scripts/group_flower_centers.py --workers 4

center_annotations.csv.gz is pinned in the AAM registry, while
center_templates.jsonl.gz and center_report.json retain graphs, counts and input
hashes. Source member names, mechanism groups and one-based source lines allow
tracing every representative back into the raw dump. Default export verifies
these annotations without repeating chemistry. The native snapshot itself
remains the original six-column product.

Derived balancing cases are built with::

    python scripts/build_flower_rbl.py

The single RBL table is Data/tables/development/rbl/flower/rbl.csv.gz and contains
89,690 cases, labelled reactant/product/both. Standalone H-only components are
omitted from both input sides before removing one smallest heavy molecule from
the selected side(s); complete targets are preserved. Both input sides remain
nonempty. Cases that remain balanced are retained and flagged.

Default export verifies and materializes the frozen reference from the dump.
For a full raw recomputation, use --rebuild-sources. The single source processor
is scripts/build_flower_aam.py, also available directly with --rebuild. It reads
the dump's raw gzip streams and elementary selections with no old folders or
loose text dependencies. Successful temporary checkpoints are removed unless
--keep-work is requested. A raw rebuild must match the pinned data bytes.

A fresh full validation rebuild from the dump reproduced all 2,735 records
byte for byte. Snapshot exports are repeatable. See doc/flower.rst and
doc/flower_mechanistic_center.rst for the recipe and method. Cleanup and source
preservation evidence are recorded in verification.json.

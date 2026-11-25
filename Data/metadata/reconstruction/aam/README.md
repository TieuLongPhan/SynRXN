The selected AAM source registry lives in `registry.json`.

Run `python -m synrxn.reconstruction builders aam` from the repository root to build all 11
datasets (101,480 rows), or add `--verify` to check existing outputs. The common
export path is `Data/tables/development/aam/<dataset>/aam.csv.gz`.

The registry pins source products, ordered source-row content, original
source locks, native input archives and applicable license evidence. The
`licenses/` directory contains captured upstream terms used by the existing
rights review; its contents are source notices, not a new blanket grant.

Eight legacy collections reproduce frozen published products. Flower,
SIHUMIx and MetaMBD retain their native processing recipes. MetaMBD remains
a local candidate-reference collection excluded from redistribution. The
historical `flower_e` and `flower_c` products are outside this selection.

See `doc/aam_reproduction.rst` for input locations, environments and full
source rebuild commands. The retired unpinned AAM importer is preserved in
the ignored local archive at `archive/source-code/retired/aam_unpinned.py`.

The current pooled table has 79,670 unique raw references. The dated
[cleaning checkpoint](cleaning-checkpoint-2026-10-27/checkpoint.rst) records
the 91,581-to-79,670 reconciliation and the current decisions for later review.
[Dataset segments](cleaning-checkpoint-2026-10-27/dataset-segments.csv) distinguish
overlapping source memberships from chosen representatives in the pooled table.
Further cleaning is deferred; partial maps and QC flags remain included.

A uniform four-column audit view is available at
`Data/tables/development/aam_uniform/aam.csv.gz`: `id`, `data_source`, `rxn`,
`ground_truth`. Run `python -m synrxn.reconstruction builders export_uniform_aam` into a fresh output
directory to reproduce it, or add `--verify` to check the installed export.
IDs match the full pool. Shared sources are separated by semicolons and include
all reaction memberships; the selected raw mapping is unchanged. Full mapping
scope, diagnostics and provenance remain in `aam_unique` and can be joined by ID.

# aam source provenance (retrofitted)

The `aam` family shipped before SynRXN required a source lock per dataset.
These locks were retrofitted so that every released row can be connected to
exactly one lock record, as the release roadmap requires.

Each lock records the fields that are verifiable from repository evidence:

- `redistribution_partition`, `upstream_license`, `license_spdx`, and
  `license_url`, verified against `Data/metadata.yaml`.
- `required_attribution` and `citations`, resolved from `doc/refs.bib`.
- `source_doi`, but **only** when the dataset cites a deposited data archive.
  A narrative paper DOI is not a data-archive DOI and is never promoted into
  this field.

The candidate now pins the exact published SynRXN v1.2.1 product for legacy
collections using its official archive hash, member hash and frozen `Data/LICENSE`
reuse grant. These are proximate published-product pins, not reconstructed original
experimental observations. More specific native source joins are retained where
available. Each lock records `published_release_origin` and its remaining
`outstanding_fields`; ultimate original-observation lineage remains qualified.
No independent human review is implied by a retrieval pin.

Release-side size, checksum, row count, and columns for each artifact are in
the top-level `manifest.json`.

The selected 11-dataset benchmark uses the current registry at
`Data/metadata/reconstruction/aam/registry.json`. Build and verify it with:

```bash
python -m synrxn.reconstruction builders aam
python -m synrxn.reconstruction builders aam --verify
```

See `doc/aam_reproduction.rst` for native source recipes, schema aliases and
the distinction between published-product and original-source reproduction.
The locks in this directory remain historical provenance records. Current
rights decisions and captured grants are included separately in each export;
legacy license tags in these locks do not override those source-specific terms.

Regenerate or verify legacy source-lock coverage with:

```bash
python -m synrxn.reconstruction.builders source_locks --check
```

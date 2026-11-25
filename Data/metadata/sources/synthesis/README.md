# synthesis source provenance (retrofitted)

The `synthesis` family shipped before SynRXN required a source lock per dataset.
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

Regenerate or verify coverage with:

```bash
python script/build_source_locks.py --check
```

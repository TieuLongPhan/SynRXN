# Yield source provenance

SynRXN 1.3.0 distributes three normalized yield tables whose upstream reuse
terms were verified before ingestion. Each JSON lock pins the exact source
version, input filename, checksum, license, retrieval date, and attribution
requirements. `build_report.json` records output checksums and QC counts.

| SynRXN dataset | Upstream snapshot | License | Raw rows | Released rows |
| --- | --- | --- | ---: | ---: |
| `yield/buchwald_hartwig_ood.csv.gz` | BH-HTE-OOD commit `a2770413fce269b49a35fb4b2e6471f25632d2ef` | Apache-2.0 | 27,497 | 27,438 |
| `yield/hitea.csv.gz` | HiTEA commit `ccc8cce274e70429498be829dd3bc57754f62265` | MIT | 39,546 | 39,546 |
| `yield/uspto_yield.csv.gz` | Figshare `10.6084/m9.figshare.14414039.v1` | MIT | 526,468 | 526,468 |

One additional yield dataset is tracked but not released:

| Candidate dataset | Source | License status | Raw rows | Release status |
| --- | --- | --- | ---: | --- |
| `yield/suzuki_miyaura.csv.gz` | Perera et al., Science 2018 | `LicenseRef-Unknown` | 5,760 | Excluded; permission requested |

The BH builder quarantines 59 measurements above 100%. HiTEA optional
non-numeric selectivity labels are preserved in `source_metadata_json` and do
not invalidate the numerical yield target. The USPTO table retains the
upstream train, validation, and test labels as `upstream_split`.

The inspected EDBO direct-arylation matrices are not distributed in this
release: their condition-only tables do not contain enough fixed reaction
structure information to construct complete SynRXN reaction records. Sources
with non-commercial, no-derivatives, or unresolved data terms are also
excluded. In particular, the 5,760-record Suzuki--Miyaura screening table from
Perera et al. (Science 2018, DOI: 10.1126/science.aap9112) remains excluded
pending written redistribution permission from the rights holder. Its presence
in the DRFP repository is recorded as provenance, not treated as a grant of
rights to SynRXN.

Rebuild with `script/build_yield_dataset.py` and the pinned local inputs. The
builder rejects checksum mismatches and writes deterministic gzip files.

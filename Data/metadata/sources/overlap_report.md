# Cross-Source Reaction Overlap Report

Generated: 2026-08-22. Regenerate with `python script/build_overlap_report.py`.
Machine-readable form: `overlap_report.json`.

## Why this exists

The release roadmap requires cross-source overlap and duplicate leakage to be
**measured**, not assumed. Corpora built from patents, HTE campaigns, and
aggregated repositories are not disjoint, so a pooled row total is not a
reaction count and split manifests cannot be frozen until overlap is known.

## Identity definition

canonical atom-map-free reactants>>products, isomeric SMILES, components sorted; agents and conditions ignored.

The canonicalization is `synrxn.yield_data.canonical_unmapped_reaction`, the
same routine the release builders use. Corpora shipping a
`canonical_reaction_smiles` column reuse it; legacy corpora are canonicalized
on the fly with the identical routine, so all 11 corpora
are directly comparable.

Scope: the yield and condition families plus the major USPTO-derived legacy
corpora. This is not the whole catalog.

## Headline numbers

| Quantity | Value |
| --- | ---: |
| Corpora compared | 11 |
| Pooled rows | 2,548,853 |
| Rows with a parseable identity | 2,548,666 |
| **Distinct reactions in the union** | **1,963,937** |
| Rows beyond distinct reactions | 584,729 |
| Redundancy fraction | 22.9% |

Pooling these corpora yields 2,548,853 rows but only
1,963,937 distinct transformations. **Never report
the pooled row total as a reaction count.**

## Per-corpus redundancy

| Corpus | Rows | Distinct reactions | Internal duplicate rows | Duplicate rate | Unparsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `classification/uspto_50k_b` | 50,016 | 49,526 | 314 | 0.6% | 176 |
| `condition/orderly_condition` | 691,105 | 669,350 | 21,755 | 3.1% | 0 |
| `condition/uspto_500` | 143,535 | 135,643 | 7,892 | 5.5% | 0 |
| `condition_sharealike/openexp` | 274,439 | 272,464 | 1,975 | 0.7% | 0 |
| `condition_sharealike/ord_hte_condition` | 123,720 | 1,820 | 121,900 | 98.5% | 0 |
| `synthesis/uspto_500` | 143,535 | 135,643 | 7,892 | 5.5% | 0 |
| `synthesis/uspto_50k` | 50,016 | 49,614 | 402 | 0.8% | 0 |
| `synthesis/uspto_mit` | 479,035 | 473,861 | 5,163 | 1.1% | 11 |
| `yield/buchwald_hartwig_ood` | 27,438 | 5,754 | 21,684 | 79.0% | 0 |
| `yield/hitea` | 39,546 | 597 | 38,949 | 98.5% | 0 |
| `yield/uspto_yield` | 526,468 | 526,468 | 0 | 0.0% | 0 |

Two things to read carefully:

- **HTE campaigns are redundant by design.** `ord_hte_condition` holds 1,820
  distinct transformations across 123,720 rows, `hitea` 597 across 39,546, and
  `buchwald_hartwig_ood` 5,754 across 27,438. One transformation is screened
  across many conditions, so structure-only identity understates their content;
  the meaningful identity for HTE is (reaction, condition). It does, however,
  bound how many independent transformations an HTE-held-out split can contain.
- **`yield/uspto_yield` is internally deduplicated**: 526,468 rows, 526,468
  distinct reactions, zero internal duplicates. Verified independently against
  its `duplicate_group` column.

## Largest pairwise overlaps

| Corpus A | Corpus B | Shared distinct reactions | Jaccard | Share of A | Share of B |
| --- | --- | ---: | ---: | ---: | ---: |
| `condition/uspto_500` | `synthesis/uspto_500` | 135,643 | 1.0000 | 100.0% | 100.0% |
| `condition/orderly_condition` | `condition/uspto_500` | 80,631 | 0.1113 | 12.0% | 59.4% |
| `condition/orderly_condition` | `synthesis/uspto_500` | 80,631 | 0.1113 | 12.0% | 59.4% |
| `condition/orderly_condition` | `synthesis/uspto_mit` | 43,129 | 0.0392 | 6.4% | 9.1% |
| `condition/orderly_condition` | `condition_sharealike/openexp` | 32,866 | 0.0362 | 4.9% | 12.1% |
| `yield/uspto_yield` | `condition/orderly_condition` | 32,484 | 0.0279 | 6.2% | 4.9% |
| `condition/orderly_condition` | `synthesis/uspto_50k` | 22,071 | 0.0317 | 3.3% | 44.5% |
| `yield/uspto_yield` | `synthesis/uspto_mit` | 11,499 | 0.0116 | 2.2% | 2.4% |
| `condition/uspto_500` | `synthesis/uspto_50k` | 8,811 | 0.0499 | 6.5% | 17.8% |
| `synthesis/uspto_500` | `synthesis/uspto_50k` | 8,811 | 0.0499 | 6.5% | 17.8% |
| `condition_sharealike/openexp` | `synthesis/uspto_mit` | 5,304 | 0.0072 | 1.9% | 1.1% |
| `condition_sharealike/openexp` | `condition/uspto_500` | 4,462 | 0.0111 | 1.6% | 3.3% |

### Findings that affect the release and the paper

1. **`condition/uspto_500` and `synthesis/uspto_500` are identical at reaction
   identity** (Jaccard 1.0000, 135,643 shared). This confirms the known
   byte-identical mirror at the chemical level too. It must be counted once.
2. **ORDerly heavily overlaps the USPTO-500 benchmark**: 80,631 shared
   reactions, which is 59.4% of all distinct USPTO-500 reactions. Any model
   pretrained on ORDerly has seen most of USPTO-500.
3. **ORDerly overlaps `synthesis/uspto_50k` on 44.5% of that corpus** (22,071
   reactions) and `synthesis/uspto_mit` on 9.1% (43,129 reactions).
4. **Yield and condition families are not disjoint**: `yield/uspto_yield` and
   `condition/orderly_condition` share 32,484 reactions, 6.17% of the yield
   corpus. A joint yield-and-condition evaluation must group these together.
5. **HTE and patent corpora are effectively disjoint.** `hitea` shares zero
   reactions with `uspto_yield`; `buchwald_hartwig_ood` shares four;
   `ord_hte_condition` shares one. Patent-to-HTE transfer is therefore a
   genuine out-of-distribution axis, not a relabeled in-domain split.

## Overlap with `yield/uspto_yield`

| Other corpus | Shared distinct reactions | Share of `uspto_yield` |
| --- | ---: | ---: |
| `condition/orderly_condition` | 32,484 | 6.17% |
| `synthesis/uspto_mit` | 11,499 | 2.18% |
| `condition_sharealike/openexp` | 1,998 | 0.38% |
| `synthesis/uspto_50k` | 1,249 | 0.24% |
| `condition/uspto_500` | 255 | 0.05% |
| `synthesis/uspto_500` | 255 | 0.05% |
| `classification/uspto_50k_b` | 62 | 0.01% |
| `yield/buchwald_hartwig_ood` | 4 | 0.00% |
| `condition_sharealike/ord_hte_condition` | 1 | 0.00% |
| `yield/hitea` | 0 | 0.00% |

## Caveats

- Structure-only identity. Shared identity means the same transformation, not the same experiment or measurement.
- Unparsed rows are excluded from identity sets and are reported per corpus; they are not silently treated as unique.
- Pooled row totals must never be reported as reaction counts.
- High-throughput campaigns are redundant under this identity by design: one transformation is screened across many conditions. For HTE corpora the meaningful identity is (reaction, condition), so a low unique-reaction count is not a data-quality defect. It does bound how many independent transformations an HTE split can hold out.

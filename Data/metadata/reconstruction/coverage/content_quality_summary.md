# Content audit findings

## Current 105-view audit

The [reconstruction-module preparation run](../../reports/kdd-reproduction/reconstruction-module-20261021-001/manifest.json) passed for all **105 catalogue views / 19,611,896 stored rows**. Its [column profiles](../../reports/kdd-reproduction/reconstruction-module-20261021-001/artifacts/coverage/content_quality.json) contain **zero malformed rows, zero fatal content failures and 33 tables with review findings**. Review findings remain explicit source/task qualifications rather than automatic exclusions.

The run verified 497 source/evidence/table files, all 40 newer table contracts and 128 legacy export files. Its isolated native rebuild reproduced **77 output files byte for byte**, and all four HiTEA preparation artifacts matched. The registry, tests, plan and literature validators passed with no input changes. No predictive model ran. This combined profile includes the solvent and M-CSA additions described below; the older 98-table statistics remain historical evidence.

The subsequent manifest improvement records physical compatibility scripts and frozen source kernels separately. Historical run manifests remain unchanged; it does not alter these dataset bytes or column profiles.

## Historical 98-table audit

The original 98-table [reproduction run](../../reports/kdd-reproduction/twelve-family-licensed-001/manifest.json) passed with no input changes during execution. It includes source/license checks, all-table content profiling, an isolated byte-identical rebuild, HiTEA regeneration, 122 tests and 5 subtests, and plan/literature validators. It fits no predictive models.

| Check | Result |
| --- | ---: |
| Catalogued tables | 98 |
| Table rows profiled | 19,597,526 |
| Malformed rows | 0 |
| Fatal content failures | 0 |
| Tables with review findings | 31 |
| Native/evidence/table files verified by release audit |303 |
| New task contracts verified |33 |
| Legacy export files verified |128 |
| Rebuilt output files matching exact bytes |61 |

Rows include generated feasibility supervision, expert annotations, measurements, computed references, simulations and overlapping views. The total is not a count of independent experiments or unique reactions. Native Transition1x additionally contains 9,624,594 atomic configurations; the table profile counts its trajectory-index rows separately.

## Subsequent solvent and mechanism additions

The current catalogue adds seven views to that historical scan: two solvent-aware views of existing rows (503 SNAr and 5,269 aqueous cycloaddition), four overlapping M-CSA mechanism views (278 observed-start proposals, 479 full proposals, 934 chains and 2,393 steps), and 4,514 LiPRED method-specific solution activation Gibbs references. Of the observed-start M-CSA records, 192 satisfy the frozen strict replay tier. LiPRED retains the unresolved paper/workbook count difference, 54 hypothetical-solvent references and six conflicting duplicate-context references as explicit flags.

The current [full source/row/contract audit](release_audit.json) passes for 105 catalogue views and all 40 acquired table contracts. The [selected offline rebuild](../../reports/coverage-solutions-mechanisms-rebuild-001/manifest.json) reproduces all 16 outputs of the three new adapters byte-identically. The current combined 105-view profile is linked at the top; the 98-table statistics remain a distinct historical snapshot. The [current hierarchy](../hierarchy/index.html) contains all views, units and qualifications.

## Missing fields

The [full column profiles](../../reports/kdd-reproduction/twelve-family-licensed-001/artifacts/coverage/content_quality.json) retain missing/absent fields and logical payload keys. 31 tables have findings, largely missing native context or optional source fields. These are source/task review items, not automatically invalid observations. No malformed JSON payloads were found. No missing value is turned into a negative label, zero response or absent impurity. Freeze task-specific eligibility masks before predictive evaluation.

## Identifier namespaces

The initial scan found 43,962 repetitions in two unqualified identifier columns. A [follow-up source audit](identifier_namespace_audit.json) verifies that both identifiers are local to their upstream partition:

| Table | Native identifier | Repetitions without namespace | Required namespace | Duplicates inside namespace |
| --- | --- | ---: | --- | ---: |
| classification/ecreact | r_id | 18,816 | split | 0 |
| yield/uspto_yield | source_record_id | 25,146 | upstream_split | 0 |

Use native partition plus identifier for source-record identity; keep the original fields. USPTO yield already has unique qualified sample_id values. The subsequent license recheck restored ECReact with the actual CLAIRE CC BY grant and applicable MIT/ODbL notices. No records were deleted based on partition-local identifier reuse. This identity rule does not establish chemical separation between train/test data.

Reproduce the follow-up:

```bash
python kdd/coverage/identifier_namespaces.py --quality-report reports/kdd-reproduction/twelve-family-licensed-001/artifacts/coverage/content_quality.json
```

## Redistribution

The content scan above remains valid for unchanged source bytes. The later [license recheck](license_recheck.md) supersedes the earlier 78-table permission decision: the current manifest admits 104 of 105 catalogued tables for their stated uses, retaining verified NC, scoped PNAS noncommercial educational table reuse, GNU, database and ShareAlike terms. The Diels–Alder downstream redistribution grant and four additional enzyme views remain held. All 41 definitions retain an admitted reference.

# Recheck of 19 historical tables and AutoTST

The previous audit mixed missing permission evidence with actual redistribution conditions. Rechecking source releases, historical repository licenses, original supplements and native rows, followed by a paper-level review of PNAS permissions, clears **18 of the 19 initial holds**. The AutoTST table is also admitted under its original CC BY-NC 4.0 license. Publication permission and unrestricted commercial permission are separate.

**Current result: 104 of 105 catalogued tables admitted for their stated uses**, with one source-permission question still open. Phosphatase is limited to noncommercial educational table reuse under a custom PNAS permission, not a Creative Commons grant for the original data. The four enzyme quarantine views are additional and remain excluded. All 12 families and 41 definitions retain admitted references.

CC BY-NC 4.0 allows sharing and adaptation for noncommercial purposes, with attribution, a license link and change notices. Its NC condition is retained; we do not relabel the data CC BY 4.0. See the [official terms](https://creativecommons.org/licenses/by-nc/4.0/) and the [actual AutoTST dataset grant](https://api.figshare.com/v2/articles/5398342). These tables are publicly distributable under NC terms rather than unrestricted open data.

| Table | Redistribution decision | Effective source/curation license expression |
| --- | --- | --- |
| aam/ecoli | Admitted | MIT |
| aam/golden | Admitted | GPL-3.0-only AND LGPL-2.1-only AND MIT |
| aam/natcomm | Admitted | CC-BY-4.0 AND GPL-3.0-only AND MIT |
| aam/recon3d | Admitted | CC-BY-NC-4.0 AND MIT |
| aam/uspto_3k | Admitted | GPL-3.0-only AND CC-BY-SA-4.0 AND MIT |
| classification/ecreact | Admitted | CC-BY-4.0 AND MIT AND ODbL-1.0 |
| classification/schneider_b | Admitted | CC-BY-4.0 AND MIT AND CC-BY-NC-4.0 |
| classification/schneider_u | Admitted | CC-BY-4.0 AND MIT AND CC-BY-NC-4.0 |
| classification/tpl_b | Admitted | CC-BY-4.0 AND MIT |
| classification/tpl_u | Admitted | CC-BY-4.0 AND MIT |
| classification/uspto_50k_b | Admitted | CC-BY-4.0 AND MIT AND CC-BY-NC-4.0 |
| classification/uspto_50k_u | Admitted | CC-BY-4.0 AND MIT AND CC-BY-NC-4.0 |
| property/e2sn2 | Admitted | CC-BY-4.0 |
| property/lograte | Admitted | CC-BY-NC-4.0 |
| property/phosphatase | Admitted for noncommercial educational table reuse | CC-BY-4.0 AND LicenseRef-PNAS-NC-Educational |
| property/rad6re | Admitted | CC-BY-4.0 |
| property/snar | Admitted | CC-BY-3.0 |
| rbl/complex | Admitted | CC-BY-4.0 AND MIT AND LGPL-2.1-only |
| synthesis/da | Pending original-source permission | Original grant unverified; historical curation label alone does not resolve it |
| synthesis/uspto_50k | Admitted | CC-BY-4.0 AND MIT AND CC-BY-NC-4.0 |

## Paper-level follow-up: phosphatase and Diels–Alder

- **Phosphatase:** the [standard PNAS license terms](https://www.pnas.org/author-center/publication-charges/standard-pnas-license-terms) apply to volumes 106–118 (2009–2021), including the 2015 original article. They expressly permit original-table reuse for noncommercial and educational purposes with the full journal reference. The independently captured publisher agreement also grants third parties noncommercial table reuse. This affirmative permission resolves the earlier assumption that a missing CC license necessarily left reuse unsupported. We apply it to the factual table extraction and preserve the separate curation attribution; no article or entire supplement is selected for publication, and no CC BY-NC license is invented.
- **Phosphatase provenance:** all 33,354 targets trace to the original supplementary table across 218 enzymes and 153 curated substrate views. For 151 substrates the targets match single source cells; two chemically duplicated substrate-name groups match the maximum of their duplicate source cells. Numeric serialization differences are below 2e-16. These are source assay-response values; the historical `Conversion` column does not establish measured percentage conversion. The [follow-up review](recheck_phosphatase_followup.json) records the source-cell proof, custom permission and capture limitations, including a clearly labeled browser transcript where direct publisher downloads were blocked.
- **Diels–Alder:** all 11,011 curated rows match the Goodman source. The [original Li et al. paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9693912/) explicitly publishes its training/validation files and states that source data accompany the paper. Its article license is CC BY-NC 3.0. The Goodman supplement accounts for 11,274 original training records minus 263 filtered records. Independent source matching finds 11,009 identical isomeric structures and two reactant-stereochemistry notation differences with matching connectivity and product stereochemistry.
- **Remaining Diels–Alder question:** the actual [2022 Reaxys publication guidance](https://supportcontent.elsevier.com/RightNow%20Next%20Gen/Reaxys/Use%20of%20Reaxys%20Publishing%20Guidelines%202022.pdf) permits publication under conditions and unlimited substance identifiers/SMILES. It does not unambiguously grant downstream redistribution of a large compilation of paired reaction transformations. The authors' specific agreement was not available for review. Public author distribution is verified; mirroring the complete payload remains pending a grant establishing that scope. Reaxys provenance alone is not treated as prohibition. The [follow-up review](recheck_da_followup.json) preserves this distinction and the reproducible native comparison.

## Verified distinctions

- Five atom-mapping tables match 5,904 pinned SynTemp rows. Historical GPL grants and original LGPL notices are preserved where applicable; a later license change does not erase the captured historical grant.
- ECReact matches the official CLAIRE CC BY 4.0 deposit exactly (185,734 rows); original database and MIT notices are retained. Schneider/USPTO50K original ACS supplements carry NC terms, so those variants are placed in the noncommercial partition.
- E2/SN2 matches the already-cleared E2 and SN2 union (3,625 rows). Complex rebalancing matches 1,748 source rows after the documented hydrogen notation normalization. RAD-6-RE and SNAr have original source grants and supporting-data evidence.

## Reproducibility and scope

The machine-readable reviews are [mapping](recheck_mapping.json), [classification](recheck_classification.json), [initial property review](recheck_property.json), [phosphatase follow-up](recheck_phosphatase_followup.json) and [Diels–Alder follow-up](recheck_da_followup.json). They pin URLs, commits, original license bodies, dataset hashes and native-row comparisons. Each follow-up explicitly supersedes the pinned prior decision while preserving its history. Remaining limitations are retained per table. Broad reuse by other projects is a lead; it is not used as the grant itself.

```bash
python kdd/coverage/audit_legacy_rights.py
python kdd/build_task_registry.py
python kdd/coverage/sync_plan.py
python kdd/coverage/audit.py
python kdd/coverage/report.py
```

Original dataset bytes, historical source locks and prior published metadata remain intact. For this compilation the [current per-table terms](publication_licenses/CURRENT-DATA-TERMS.txt) and [release manifest](release_manifest.json) govern redistribution; the earlier blanket license labels are retained only as explicitly labeled history. Nothing has been uploaded or externally published.

# Condition and reagent benchmark candidate audit

Reviewed 2026-08-22. A public download or open-source model repository is not
by itself a data redistribution license. Candidates stay out of `Data/` until
the exact files, checksums, attribution, and data-specific terms are pinned.

| Candidate | Scale | Targets | Redistribution assessment | SynRXN decision |
| --- | ---: | --- | --- | --- |
| ORD HTE allowlist | 123,812 raw; 123,720 released | Structured agents, solvents, operating conditions, outcomes | ORD explicitly applies CC-BY-SA-4.0 to data under `data/` | Built as a separate ShareAlike artifact; 92 incomplete NiCOlit structures quarantined |
| OpenExp | 274,439 | Experimental action/procedure sequences from reactions | Paper explicitly states CC-BY-SA; exact OSF archive pinned and verified | Built as a separate ShareAlike procedure artifact |
| USPTO-Condition (Parrot) | 680,741 | Catalyst, solvent 1/2, reagent 1/2 | Paper says freely available, but repository MIT text refers to software and data come from an external archive | Benchmark is highly relevant; ask authors for explicit processed-data redistribution permission |
| Reacon | 690,872 | Up to 1 catalyst, 2 solvents, and 3 reagents | Dataset is downloadable, but no explicit repository data license was found | Permission required; likely overlaps USPTO-Condition substantially |
| EDBO/DeeBO reaction screens | Multiple small HTE/optimization tables | Ligand, base, solvent, concentration, temperature, yield | Repository licenses exist, but tables cite heterogeneous original publications; several useful screens already have ORD copies | Do not add duplicate copies; unresolved files remain audit-only |
| AlignReact USPTO files | Derived benchmark files | Catalyst, solvents, reagents | Repository is CC BY 4.0, but external bundles are transformations of Parrot/T5Chem/YieldBERT inputs whose source rights still govern | Do not count as a new corpus; use only as split/model interoperability evidence |
| QUARC | Commercial corpus; 133,993 held-out complete-condition reactions reported | Agents, temperature, reactant and agent equivalence ratios | Training data are Pistachio 2023Q2 and require a commercial license | Exclude raw data; code and models are not a redistributable benchmark corpus |
| Reaxys-TotalSyn-Condition | Commercial | Catalyst, solvents, reagents, temperature | Reaxys subscription data | Exclude from redistribution |

Primary references:

- ORD data and license: https://github.com/open-reaction-database/ord-data
- ORD condition schema: https://docs.open-reaction-database.org/en/stable/schema.html
- OpenExp paper and license statement: https://aclanthology.org/2024.findings-acl.318/
- OpenExp code/archive instructions: https://github.com/syr-cn/ReactXT
- USPTO-Condition paper: https://doi.org/10.34133/research.0231
- Parrot repository and processed-data download: https://github.com/wangxr0526/Parrot
- Reacon benchmark: https://doi.org/10.1039/D4SC05946H
- Reacon data repository: https://github.com/wzhstat/Reaction-Condition-Selector
- QUARC data-availability statement: https://doi.org/10.1039/D5SC04957A
- EDBO data/code repository: https://github.com/b-shields/edbo
- DeeBO reaction tables: https://github.com/doyle-lab-ucla/bandit-optimization/tree/main/datasets

Repository snapshot findings:

- Parrot commit `0fb2325567e21011589641544e32427c8244e2a9` contains an
  MIT license whose grant refers to “Software and associated documentation”.
  The processed USPTO-Condition archive is not in Git; it is downloaded from
  Google Drive. No data-specific terms accompany that external file.
- Reacon commit `aeedf1b8682d5b61f2ee3ea5a987fdd0c571c7b0` contains label
  vocabularies and a 100-row JMC example but no repository LICENSE. The large
  train/validation/test CSV files are external Dropbox downloads.
- EDBO commit `9b41eac3f6d9e520547702fd5b0c7ef6441625a4` is MIT, but its
  example tables cite multiple source publications. The 1,728-row direct
  arylation screen is already included in the ORD allowlist, so a second copy
  would add license ambiguity and duplication rather than benchmark breadth.
- DeeBO commit `ae3e910dcea27cfd1631a28b793b6c63607baedd` is GPL-3.0,
  while its data README attributes each table to a different publication.
  The 960-row amidation and 1,536-row arylation screens are already represented
  by ORD datasets in `ord_hte_condition`; the Ahneman C–N screen overlaps the
  Buchwald-Hartwig OOD yield artifact.
- AlignReact commit `0c6dc17615c42836cc779687b3400d6be73e0e9b` is CC BY
  4.0, but its README identifies the external condition files as processed
  Parrot, T5Chem, and YieldBERT datasets. A derivative repository license does
  not remove the need to comply with those underlying dataset terms.

Recommended acquisition order:

1. Email the Parrot authors for an explicit processed-data license and
   redistribution statement before downloading it into the release tree.
2. Revisit only EDBO/DeeBO files that are neither already present through ORD
   nor represented in the yield branch, and require original-source rights.

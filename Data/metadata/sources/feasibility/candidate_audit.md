# Feasibility candidate audit

Reviewed 2026-08-22. A distributable feasibility table must contain
non-synthetic experimental records and explicit redistribution terms.

| Candidate | Evidence | Decision |
| --- | --- | --- |
| URSA reaction plausibility | Official dataset card declares Apache-2.0; 1,000 model-proposed reactions have expert labels | Excluded because the reactions are not experimentally observed |
| ChemLex acid–amine HTE | 11,669 wet-lab reactions are deposited on Zenodo and mirrored in a BSD-3-Clause code/data repository, but the current Zenodo record displays no dataset license and the article is CC BY-NC-ND | Excluded pending an explicit data-specific redistribution statement |
| Chemical.AI practicality data | Public repository provides synthetic negatives, while real failed-reaction records were removed or originate in partner laboratories; no clear data license was found | Excluded |
| Synthetic USPTO negatives | Reproducible to generate, but “not reported” is not evidence that a reaction is infeasible | Do not mix with expert or experimental negatives; a future artifact must label these as synthetic |

Primary sources:

- https://huggingface.co/datasets/insilicomedicine/URSA-benchmarking-sets
- https://doi.org/10.1038/s41467-025-59812-0
- https://doi.org/10.5281/zenodo.17596563
- https://github.com/jshmjs45/data_for_chem

# Impurity candidate audit

Reviewed 2026-08-22. Open reaction-impurity ground truth remains scarce, and
predictions are never relabelled as observations.

| Candidate | Evidence | Decision |
| --- | --- | --- |
| SEISMiQ `simulated_impurities.zip` | Zenodo deposit is CC BY 4.0; archive contains 2,541 generated candidate structures for five queries | Excluded because candidates are generated rather than experimentally observed |
| SEISMiQ training/test spectral archives | Aggregate GNPS, HMDB, SUMNER, MassBank, MoNA, and other sources with heterogeneous downstream terms | Excluded from redistribution |
| ORD outcome products | Protobuf `is_desired_product=false` is frequently an unset default; the apparent explicit cohort checked locally contained internal standards, not impurities | Excluded until a source has actual undesired products with product roles and supporting measurements |
| Arun et al. impurity workflow | Case studies and mining workflow use Reaxys under license | Excluded; underlying reaction data are not redistributable |
| ASKCOS impurity prediction | Open software, but deployed predictors rely partly on Pistachio or USPTO model training and do not provide an observed-impurity benchmark | Not a dataset source |

Primary sources:

- https://doi.org/10.5281/zenodo.16438770
- https://doi.org/10.1039/D5DD00115C
- https://doi.org/10.1002/cmtd.202200062
- https://askcos-docs.mit.edu/guide/3-Advanced-Usage/3.4-More-forward-synthesis-analysis-examples.html

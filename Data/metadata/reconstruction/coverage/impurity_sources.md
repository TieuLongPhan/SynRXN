# Observed impurity references

The [adapter](adapters/impurity.py) writes two licensed sources to the [impurity catalog](../../Data/coverage_expansion/impurity/catalog.json), discovered directly by the unified registry. It fills the `impurity_prediction` task definition with two explicitly bounded variants.

| Variant | Records | Reference | Licence |
|---|---:|---|---|
| Known organic side-product profile | 756 | Source-identified benzaldehyde yields in styrene epoxidation under different PBA catalyst compositions | MIT |
| Reported inorganic impurity-phase identities | 7,261 | Concrete impurity formulas extracted from experimental synthesis reports across 6,300 original publications | CC-BY-4.0 |

These references do not establish general open-world organic impurity discovery. They do provide actual observed side-product/impurity data rather than generated candidates, unreported labels or inferred mass-balance by-products.

## Organic by-product profile

[Jia et al., Nature (2025)](https://doi.org/10.1038/s41586-025-09490-1) explicitly identifies benzaldehyde as the main by-product of PBA-catalysed styrene epoxidation. The separately **MIT-licensed [Zenodo deposit](https://zenodo.org/records/14880579)** provides the native 756-row catalyst-composition table. The author archive's MIT notice is preserved alongside its dataset metadata and SHA-256 hashes.

The intended product is styrene oxide; the unwanted product is benzaldehyde. Model inputs contain reactant/intended-product identities, the declared candidate, metal fractions and fixed operating conditions. Outputs preserve native benzaldehyde yield estimates from measured UV–Vis spectra. Desired-product yields and selectivity are forbidden inputs. Native fitted zeros remain quantitative values and are not converted into a claim of measured absence. This is a closed-candidate impurity-profile regression task, with one chemical transformation and a composition panel.

The data deposit also contains interpolated visualizations and calculated candidates. They are excluded. Publisher article assets have different CC-BY-NC-ND terms; only the separately MIT-licensed author data are included in the release allowlist.

## Inorganic impurity-phase identities

[Lee et al., Scientific Data (2025)](https://doi.org/10.1038/s41597-025-06222-y) releases **80,806** structured synthesis records under **CC-BY-4.0** through [Figshare version 1](https://doi.org/10.6084/m9.figshare.30423274.v1). The native records explicitly distinguish intended target, precursors and impurity phases and retain each original experimental publication DOI. These observations were extracted with an LLM and parsed by the source authors; they are not independently verified new experiments.

The adapter keeps 7,261 records with a balanced target reaction, concrete precursor/target identities, at least one concrete impurity formula and an original DOI. This is our documented qualification predicate, not the paper's separate 6,542-record validation cohort. The retained data cover 3,495 distinct formula strings; 6,519 records have numerical synthesis context. The [qualification audit](../../Data/coverage_expansion/impurity/solid_state_qualification.json) records every exclusion:

- 61,937 records without reported impurity phases are excluded, rather than labelled negative.
- 7,899 impurity-report records have no concrete impurity identity.
- 3,659 fail the balanced-target criterion.
- 21 have unresolved target/precursor identities and 29 label the intended target itself as an impurity.

Among retained records, 273 contain a partial known identity set and remain explicitly flagged. No unreported phase is treated as absent. Materials Project IDs and energies are omitted: the source's most-stable-entry assignment is not experimental polymorph identification. Labels are composition formulas rather than molecular SMILES or measured crystal structures. Free-text subjects/results are excluded from model inputs; only whitelisted operation types and numerical temperature/time values are retained. Those operations are DOI-level context and may not uniquely describe one recipe.

Future evaluation must group original publications and related precursor–target identities. Recall over reported impurity identities is supported; exhaustive impurity absence and condition-specific causal claims are not.

## Reproduce and verify

```bash
python kdd/coverage/adapters/impurity.py --fetch
python -m unittest kdd.coverage.test_impurity

# The process builder also acquires/rebuilds these sources automatically.
python kdd/coverage/adapters/process.py --fetch
```

The impurity adapter retains native row/index identities, exact download URLs, asset/licence SHA-256 hashes, source notices, original publication DOIs, targets, feature whitelists and grouping/metric contracts. Its two tables remain in their own catalog and are never duplicated into the process catalog. No original SynRXN dataset or task interface is changed.

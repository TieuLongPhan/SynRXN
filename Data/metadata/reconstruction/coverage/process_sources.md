# Analytical and process data acquisition

The [adapter](adapters/process.py) materializes **14 task views for five task definitions**. It also rebuilds the separate [two-view impurity branch](impurity_sources.md) when its licensed source assets are present. Each branch keeps its own catalog so the unified registry discovers each table exactly once. The [process catalog](../../Data/coverage_expansion/process/catalog.json) records exact native assets, SHA-256 hashes, pinned author-repository revisions, licences, licence-evidence hashes, reference types, input whitelists, grouping fields and metrics. These are source-backed task contracts; model evaluations and independently qualified benchmark splits remain separate work.

| Task | Tables | Task-view records | Reference and scope |
|---|---:|---:|---|
| Analytical interpretation | 3 | 3,022 | Measured NMR spectra; independently derived component-window relative areas |
| Reaction progress monitoring | 4 | 3,255 | 3,019 next-spectrum area forecasts plus 236 next-yield forecasts |
| Reactor/scale-dependent prediction | 6 | 10,188 | 113 measured reactor experiments; 10,000 native mechanistic simulations; 15 measured pilot experiments; 60 native interpolation rows |
| Dynamic process control | 1 | 600 | Derived transitions from the published do-mpc CSTR model and an explicit evaluation oracle |
| Process resource/impact assessment | 3 overlapping pilot views | 75 | 15 measured coke-deposition endpoints plus 60 interpolation rows; a narrow material-loss endpoint |

Counts overlap across tasks and must not be added as independent experiments. The 14 process output files contain 17,065 records. The separate impurity branch adds 756 organic assay and 7,261 literature-extracted recipe records, for 25,082 records across both branches. NMR analytical and forecasting views share three measured trajectories; the additional low-field data contain four trajectories with 60 observations each. CSTR control contains 12 simulated episodes, not 600 laboratory experiments.

## Reproduce

Use Python with NumPy, pandas, SciPy, requests and openpyxl. The original project environment already supplies these dependencies.

```bash
# Download immutable records and exact pinned repository revisions, then rebuild.
python kdd/coverage/adapters/process.py --fetch

# Rebuild from the local raw cache, without downloading anything.
python kdd/coverage/adapters/process.py

# Verify native counts, evidence types, missingness, temporal isolation,
# numerical integration, replayable dynamics, licences and output hashes.
python -m unittest kdd.coverage.test_process
```

Compressed JSONL output has a fixed gzip timestamp and stable key ordering. `load_spectrum` reads only the requested current native column; opening an entire temporal matrix as a model input is forbidden. `cstr_step` exposes the numerical transition oracle for closed-loop evaluation of a proposed policy. Recorded random excitation actions are not optimal-controller labels.

## Source and licence provenance

1. **Magnetstein reaction monitoring (2025).** The author [paper](https://doi.org/10.1021/acs.analchem.5c00800) and [repository](https://github.com/BDomzal/magnetstein_x_chemical_reactions) point to [Zenodo 14814657](https://zenodo.org/records/14814657). The immutable data record grants **CC-BY-4.0**. HTTP range retrieval extracts the three cropped measured NMR trajectories and associated time/calibration assets from the 2.26 GB archive; ZIP member CRC checks, native names and SHA-256 hashes are preserved. The 1,023 sucrose, 999 PMG284 and 1,000 PMG287 spectra are three trajectories, with finer time points strongly dependent.

   The GitHub repository's Mnova comparison CSVs have no explicit redistribution licence. They are excluded from the released task labels and source-asset list. Released references instead use independently implemented trapezoidal integration of the CC-BY spectra within the published component windows. Signed integrals are preserved; component integrals are clipped at zero and normalized only to derive relative analytical areas. NaNs are omitted within each component window and counted, never represented as measured zeros. These estimates do not establish independent concentration truth. Numerical integration is the primary baseline, and a larger, independently calibrated concentration benchmark remains outside this qualification. Component templates extracted from late trajectory samples are not permitted as forecasting inputs.

2. **Low-field reaction monitoring (2026).** [Zenodo 20595287](https://zenodo.org/records/20595287) grants **CC-BY-4.0** and associates its data with the [ChemRxiv preprint](https://doi.org/10.26434/chemrxiv.15005188/v1). `Results_Reactions_1-4.xlsx`, sheet `Reactions`, contains four reported yield profiles over 60 minutes. The native plotting script labels time in minutes and yield in percent. This view forecasts the next reported yield from current observations; it does not claim raw-FID-to-yield calibration. Raw FIDs totaling 76 GB remain indexed upstream but were not acquired for this narrower task view.

3. **Reac-Discovery (2025).** The [paper](https://doi.org/10.1038/s41467-025-64127-1) links the author [MIT repository](https://github.com/catm542-ai/Reac-Discovery), pinned to `ce917a718c17bcf388a36744aad03b535a408949`. M2 worksheets contain 55 hydrogenation and 58 CO2-cycloaddition records with reactor descriptors, operating conditions, measured yield and reported space–time yield. M1 workbooks are overlapping representations and do not contribute additional independent experiments. This supports reactor-dependent performance prediction; matched laboratory-to-industrial scale pairs are absent. The article itself has CC-BY-NC-ND terms; acquisition uses the separately MIT-licensed repository assets.

4. **Naphtha FCC scale-up (2025).** The [paper](https://doi.org/10.1038/s41467-025-63982-2) links the [MIT author repository](https://github.com/chenzhengyuAI/complex-reaction-system-scale-up), pinned to `0080d57a1650ecb349eea2c3dbd5afd89b61cde5`, and [Zenodo archive](https://zenodo.org/records/16810762). All native CSVs are headerless. The laboratory source consists of 10,000 mechanistically generated examples, each with four operating quantities, 129 feed-component fractions and 129 product-component references. The pilot source has 15 physical test experiments and 48/12 interpolation-derived train/validation rows. Interpolation descendants of the same experiments cannot establish independent test evidence. The source-full pilot input/target files have incompatible lengths, 1,016 versus 982, and are never joined.

   Native `Bulk_prop.py` documents the bulk-reference order as dry gas, LPG, gasoline, diesel, coke, then five PIONA quantities. The methods describe coke analysis through a carbon-sulfur analyzer. Its fifth native target is retained as a measured coke-deposition/material-loss endpoint for the 15 pilot experiments. The 60 interpolated values remain explicitly interpolated. This does not support claims about life-cycle impact, energy, monetary cost or safety.

5. **do-mpc CSTR model.** The [source example](https://github.com/do-mpc/do-mpc/tree/9d5a8df0e4d6acac899bf0a65ffd4c30a6e2e4cc/examples/CSTR) is **LGPL-3.0-or-later**. The original model, bounds and licence are retained. The adapter's CSTR equation transcription carries the LGPL notice and preserves model parameters; integration uses SciPy DOP853. Twelve episodes of seeded action excitation produce 600 explicitly derived transitions. The control evaluation contract exposes current state, known scenario parameters, two bounded actions and the transition oracle. Evaluation measures closed-loop tracking error, input effort and constraint violation; no imitation or optimal-controller performance is claimed.

DLRN's academic-only licence is excluded from the acquired release set. Downloaded exploratory assets and unlicensed comparison files stay outside the catalog's released source-asset allowlist.

## Qualification limits

The two previously empty high-level families now have usable, provenance-backed data contracts. Their supported subproblems remain narrow: current analytics provides measured spectra with derived relative-area targets; process impact provides measured coke deposition; control uses a published-model simulator. General structure elucidation, independently calibrated mixture concentrations, broad scale transfer, physical plant control, full environmental assessment and safety assessment require additional sources and evaluation work. No task view in this acquisition is a completed trained benchmark.

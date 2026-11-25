# Impurity candidate provenance

SynRXN currently distributes no impurity table. The small SEISMiQ simulated
impurity release is retained as an audit candidate only. Its CC BY 4.0 deposit,
exact archive checksum, and DOI are pinned in `seismiq_simulated.json`.

This is a **candidate enumeration**, not an experimental impurity ground-truth
set. It contains 2,541 model-predicted structures for five starting-material
queries. It is excluded from `Data/impurity/` under the strict policy requiring
observed experimental records and confirmed redistribution rights.

The source's large training and test archives aggregate third-party spectral
collections with heterogeneous licenses. SynRXN does not redistribute them.
Only the separately released `simulated_impurities.zip` is used here.

The retained builder can reproduce the audit candidate with:

```bash
python script/build_impurity_dataset.py \
  --seismiq /path/simulated_impurities.zip \
  --out /tmp/seismiq_simulated.audit.csv.gz
```

Do not include that output in a public release. A future impurity table must
contain explicitly observed side products rather than generated candidates.

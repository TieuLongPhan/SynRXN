# Feasibility candidate provenance

SynRXN currently distributes no feasibility table. The exact URSA Hugging Face
revision, source checksum, and Apache-2.0 dataset-card license are retained in
`ursa_plausibility.json` for audit purposes only.

URSA contains 500 model-proposed reactions labelled plausible and 500 labelled
implausible by expert organic chemists. Although the labels are human, the
reaction candidates are not experimentally observed. Under SynRXN's strict
policy for this task—experimental records plus confirmed redistribution
rights—the normalized table is excluded from `Data/feasibility/`.

The retained builder can reproduce the audit candidate with:

```bash
python script/build_feasibility_dataset.py \
  --ursa /path/URSA-reaction-plausibility-bench-2026.csv \
  --out /tmp/ursa_plausibility.audit.csv.gz
```

Do not include that output in a public release. The preferred next source is
experimental feasibility data with an explicit data-specific license.

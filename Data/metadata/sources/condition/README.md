# Condition source provenance

The provisional SynRXN 1.4 condition work tracks three independently licensed
sources: ORDerly-condition version 4 under CC BY 4.0, a non-overlapping ORD
HTE allowlist under CC BY-SA 4.0, and OpenExp under CC BY-SA 4.0. Source locks
pin the exact upstream files, versions or commits, sizes, checksums, and
attribution requirements. The ShareAlike license text is included as
`CC-BY-SA-4.0.txt`.

The normalized builder preserves ORDerly's generic agent roles. It does not
relabel agents as catalysts, ligands, bases, acids, or additives. Missing
values remain blank, while temperature and optional yield anomalies are kept
with explicit quality flags and aggregate counts.

These condition artifacts are not part of the SynRXN 1.3.0 release manifest.
Rebuild them with `script/build_condition_dataset.py`, the `maintenance`
dependency group, and the pinned local inputs. The three outputs remain
separate so ORDerly is not mislabeled as ShareAlike and ShareAlike adaptations
are not mislabeled under the repository's software license.

The two ShareAlike outputs are written to the physically separate
`Data/condition_sharealike/` partition, which carries its own `LICENSE`,
`NOTICE.md`, and `manifest.json`. Only `orderly_condition.csv.gz` and the
`uspto_500.csv.gz` alias belong in `Data/condition/`. The dataset catalog
refuses to load if a CC BY-SA artifact is found outside the ShareAlike
partition.

```bash
python script/build_condition_dataset.py \
  --orderly-train /path/orderly_condition_train.parquet \
  --orderly-test /path/orderly_condition_test.parquet \
  --ord-source-dir /path/ord-parquets \
  --openexp /path/OpenExp.tar
```

The ORD directory uses `<dataset_id>.parquet` filenames listed in
`ord_hte_condition.json`; every checksum and raw row count is enforced.

The current deterministic build contains 691,105 records (625,664 train and
65,441 test) and has SHA-256
`789e1c41f6dcc00cff2600aa21d034db327428b2df29427721f2ba103ca91e57`.
Thirty-seven malformed source reactions were quarantined. Exact missingness,
quality, and grouping counts are recorded in `build_report.json`.

`ord_hte_condition.csv.gz` contains 123,720 usable records from 123,812 raw
ORD experiments across 16 cohorts and has SHA-256
`9d77cb0ef9233dd7922e4a65ea89e046051581c9200dd3eb7c85d3ae06514c5c`.
Ninety-two NiCOlit records without a complete reactant-to-product structure
were quarantined. The Pfizer 39k, Perera Suzuki 5,760, Ahneman, and
AstraZeneca 750 cohorts are excluded because they overlap yield artifacts.

`openexp.csv.gz` contains all 274,439 released reaction-procedure pairs and has
SHA-256
`f38c6bf858aa6d6d983c15199fc90f28e999ba6ea670229ab0307e70e919edc2`.
The source code's random 8:1:1 shuffle has no fixed seed, so SynRXN provides a
clearly labeled deterministic transformation-group split instead.

`condition/uspto_500.csv.gz` is a temporary byte-identical mirror of
`synthesis/uspto_500.csv.gz`, retained at both paths during the task-family
transition. Both files have SHA-256
`fad362c125f8fc6195d49748bbd9f381ca4c9d3f03935d82f457b350ab46bef1`.
They represent one 143,535-record benchmark and must not be counted twice.

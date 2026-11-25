# SynRXN

<p align="center">
  <img src="doc/_static/synrxn-logo.svg" alt="SynRXN logo" width="320">
</p>

[![PyPI version](https://img.shields.io/pypi/v/synrxn.svg)](https://pypi.org/project/synrxn/)
[![Release](https://img.shields.io/github/v/release/tieulongphan/synrxn.svg)](https://github.com/tieulongphan/synrxn/releases)
[![Software archive](https://img.shields.io/badge/Zenodo-software%20archive-blue)](https://doi.org/10.5281/zenodo.17297258)
[![CI](https://github.com/tieulongphan/synrxn/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/tieulongphan/synrxn/actions/workflows/test-and-lint.yml)
[![Stars](https://img.shields.io/github/stars/tieulongphan/synrxn.svg?style=social&label=Star)](https://github.com/tieulongphan/synrxn/stargazers)

SynRXN connects reaction datasets to evaluation tasks, split protocols, and source-specific reuse terms.

The **1.5.0 data snapshot** covers **40 tasks across 12 families**. Its **102 source tables** contain **5,485,077 stored rows**.

[![SynRXN workflow from reaction data sources and curation through the 1.5.0 snapshot to evaluation across 12 families](doc/figure/synrxn-workflow-1.5.0.png)](doc/figure/synrxn-workflow-1.5.0.png)

The snapshot has **95 primary tables** and seven companion views. The [evaluation atlas](doc/figure/evaluation-atlas-1.5.0.png) lists the tasks and their table counts. The [TikZ source](doc/figure/synrxn-workflow-1.5.0.tex) rebuilds the workflow figure.

## Get the 1.5.0 data

[Browse and download the prepared tables](Data/releases/1.5.0/) or use the [machine-readable catalog](Data/releases/1.5.0/tables.json) to find a table by task, format, or license. The download guide includes checksums, source terms, and a DuckDB example for the Zstandard-compressed tables.

## Installation

The software instructions and examples below use the published 1.2.1 interface. The 1.5.0 table downloads use the data guide above.

SynRXN requires Python 3.11 or later.

```bash
pip install synrxn
```

Install optional dependencies when you need the broader tooling stack:

```bash
pip install "synrxn[all]"
pip install "synrxn[query]"    # PyArrow + embedded DuckDB
pip install "synrxn[service]"  # optional read-only HTTP API
```

For development:

```bash
git clone https://github.com/TieuLongPhan/SynRXN.git
cd SynRXN
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
from pathlib import Path
from synrxn.data import DataLoader

dl = DataLoader(
    task="classification",
    source="zenodo",
    version="1.2.1",
    cache_dir=Path("~/.cache/synrxn").expanduser(),
)

print(dl.available_names())
df = dl.load("schneider_b")
print(df.shape)
print(df.columns.tolist())
```

Browse the interactive [Dataset Catalog](https://synrxn.readthedocs.io/en/latest/catalog.html)
to compare task, split, target, provenance, and reaction previews. It opens with
the split-bearing `classification/schneider_b` benchmark as a concrete example.
The same metadata is available in Python, including when working from a local
checkout without a network request:

```python
from synrxn import DataLoader, DatasetCatalog

catalog = DatasetCatalog()
print([item.name for item in catalog.list(task="property", has_split=True)])

local = DataLoader(task="classification", source="local", data_dir="Data")
sample = local.load(
    "schneider_b",
    columns=["r_id", "label", "split"],
    filters={"split": "test"},
    nrows=1_000,
)
```

Use an exact commit for development snapshots you want to reproduce later:

```python
from pathlib import Path
from synrxn.data import DataLoader

dl = DataLoader(
    task="property",
    source="commit",
    version="3e1612e2199e8b0e369fce3ed9aff3dda68e4c32",
    cache_dir=Path("~/.cache/synrxn").expanduser(),
    gh_enable=True,
)

df = dl.load("b97xd3")
print(df[["r_id", "ea", "dh"]].head())
```

## Data Concept

The public data lives in `Data/` and is grouped by benchmark task:

| Folder | Purpose | Example datasets | Core columns |
| --- | --- | --- | --- |
| `Data/aam/` | Atom-atom mapping comparison | `uspto_3k`, `golden`, `ecoli` | `ground_truth`, mapper outputs, `rxn` |
| `Data/classification/` | Reaction class, template, and enzyme classification | `uspto_50k_b`, `tpl_u`, `ecreact` | `rxn`, labels, optional `split` |
| `Data/property/` | Reaction property prediction | `b97xd3`, `rgd1`, `sn2` | `aam` or `rxn`, target values, optional `split` |
| `Data/rbl/` | Reaction balancing and rebalancing | `mos`, `mnc`, `mbs`, `complex` | unbalanced `rxn`, balanced `ground_truth` |
| `Data/synthesis/` | Synthesis and retrosynthesis datasets | `uspto_mit`, `uspto_50k`, `da` | reactions, split/source metadata, optional reagents |

## Reproducible Splits

```python
from pathlib import Path
from synrxn.data import DataLoader
from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter

dl = DataLoader(
    task="property",
    source="zenodo",
    version="1.2.1",
    cache_dir=Path("~/.cache/synrxn").expanduser(),
)
df = dl.load("b97xd3")

splitter = RepeatedKFoldsSplitter(
    n_splits=5,
    n_repeats=2,
    ratio=(8, 1, 1),
    shuffle=True,
    random_state=1,
)
splitter.prepare_splits(df, stratify=None)
train_df, val_df, test_df = splitter.get_split(0, 0, as_frame=True)
print(len(train_df), len(val_df), len(test_df))
```

## Validate a Checkout

Release integrity and catalog metadata can be checked through the installed CLI:

```bash
synrxn verify-manifest --manifest manifest.json --root Data
synrxn validate --data-dir Data --metadata Data/metadata.yaml --manifest manifest.json
synrxn datasets list --task property --has-split
synrxn datasets describe property rgd1
```

The first command verifies every declared size and SHA-256 checksum. The second
checks catalog coverage, observed schemas, row identifiers, published split
values, and manifest row counts.

## Query Layer and Optional Service

SynRXN does not migrate its immutable benchmark records to a relational
database. Compressed CSV remains the canonical release format. Deterministic
Parquet derivatives add typed, projected access, and embedded DuckDB provides
SQL execution behind an allowlisted Python API without operating a database
server.

```bash
synrxn parquet build --data-dir Data --output-dir Parquet
synrxn parquet verify --data-dir Data --parquet-dir Parquet
```

```python
loader = DataLoader(
    task="classification",
    source="local",
    data_dir="Data",
    parquet_dir="Parquet",
)
with loader.scan("schneider_b") as scan:
    page = scan.collect(
        columns=["r_id", "label", "split"],
        filters={"split": "test"},
        limit=100,
    )
```

Run `synrxn-service` only when a deployed client needs remote, record-level
pagination. It validates the derived release index before startup and exposes a
bounded read-only API with OpenAPI documentation. PostgreSQL becomes useful
only for future mutable shared state such as user accounts, annotations,
curation workflows, or benchmark submissions—not for the release datasets.

```bash
pip install "synrxn[service]"
SYNRXN_PARQUET_DIR=Parquet SYNRXN_MANIFEST=manifest.json synrxn-service

# In a second terminal: inspect a bounded, projected page of the benchmark.
curl 'http://127.0.0.1:8000/v1/datasets/classification/schneider_b/rows?columns=r_id,label,split&filter=split%3Dtest&limit=5'
```

The interactive OpenAPI interface is available at `http://127.0.0.1:8000/docs`.
The service is read-only and rejects releases whose Parquet index does not
match the canonical manifest.

## AAM Validation

SynRXN now requires `synkit>=1.5.0,<1.6.0`, and `acc_aam` uses SynKit's 1.5
`AAMValidator`. Keep its default `strip_unbalanced_maps=True` to reproduce the
historical SynRXN metric. A full RC and ITS comparison across 5,904 reactions
and four mapper outputs produced zero differences in 47,232 row-level decisions.

```bash
python script/compare_aam_validators.py --methods RC ITS --n-jobs -1
```

## Documentation

- Documentation: https://synrxn.readthedocs.io/en/latest/
- Software archive: https://doi.org/10.5281/zenodo.17297258
- SynRXN 1.5.0 prepared tables: [Download guide](Data/releases/1.5.0/)
- Source code: https://github.com/TieuLongPhan/SynRXN
- Issues: https://github.com/TieuLongPhan/SynRXN/issues

## Citation

If you use SynRXN in your research, please cite:

> Tieu-Long Phan, Nhu-Ngoc Nguyen Song, and Peter F. Stadler. SynRXN: An Open Benchmark and Curated Dataset for Computational Reaction Modeling. *Scientific Data* **13**, 625 (2026). https://doi.org/10.1038/s41597-026-07260-w

```bibtex
@article{phan2026synrxn,
  title = {SynRXN: An Open Benchmark and Curated Dataset for Computational Reaction Modeling},
  author = {Phan, Tieu-Long and Nguyen Song, Nhu-Ngoc and Stadler, Peter F.},
  journal = {Scientific Data},
  volume = {13},
  pages = {625},
  year = {2026},
  doi = {10.1038/s41597-026-07260-w},
  url = {https://www.nature.com/articles/s41597-026-07260-w}
}
```

## License

This project is licensed under the MIT License. Dataset-specific terms are summarized in [Data/LICENSE](Data/LICENSE) when applicable.

## Acknowledgments

This project has received funding from the European Union's Horizon Europe Doctoral Network programme under the Marie Sklodowska-Curie grant agreement No. 101072930 ([TACsy](https://tacsy.eu/)).

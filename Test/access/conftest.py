from synrxn.reconstruction.paths import REPO

import pytest


@pytest.fixture()
def ecoli_catalog(tmp_path):
    import yaml
    from synrxn.data import DatasetCatalog

    source = REPO / "Data/metadata/catalog.yaml"
    metadata = yaml.safe_load(source.read_text(encoding="utf8"))
    key = "aam/ecoli.csv.gz"
    metadata["datasets"] = {key: metadata["datasets"][key]}
    target = tmp_path / "metadata.yaml"
    target.write_text(yaml.safe_dump(metadata), encoding="utf8")
    return DatasetCatalog(target)


@pytest.fixture()
def parquet_release(tmp_path, ecoli_catalog):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    from synrxn.access.query import convert_release

    source = REPO / "Data"
    output = tmp_path / "Parquet"
    artifacts = convert_release(source, output, ecoli_catalog)
    return output, ecoli_catalog, artifacts[0]

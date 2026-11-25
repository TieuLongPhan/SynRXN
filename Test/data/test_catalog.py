import gzip

import pandas as pd
import pytest

from synrxn.data import CacheManager, DataLoader, DatasetCatalog, LocalSource, Task


def test_packaged_catalog_covers_current_inventory():
    catalog = DatasetCatalog()
    assert len(catalog.list()) == 52
    assert catalog.get("prop", "rgd1").targets == ("ea",)
    assert catalog.get(Task.CLASSIFICATION, "ecreact").row_identifier_unique is False
    assert "schneider_b" in catalog.available_names("classification")
    assert Task.normalize("yld") is Task.YIELD
    assert Task.normalize("cond") is Task.CONDITION
    assert Task.normalize("sel") is Task.SELECTIVITY
    assert Task.normalize("feas") is Task.FEASIBILITY
    assert Task.normalize("mech") is Task.MECHANISM
    assert catalog.available_names("feasibility") == ["frea_corpus", "frea_expert"]
    assert catalog.available_names("mechanism") == ["synepd_polar"]
    for absent in ("impurity", "imp"):
        with pytest.raises(ValueError):
            Task.normalize(absent)
    assert catalog.available_names("selectivity") == [
        "hitea_chiral_resolution",
        "regiosqm_eas",
    ]
    assert catalog.available_names("condition") == [
        "orderly_condition",
        "uspto_500",
    ]
    assert catalog.available_names("condition_sharealike") == [
        "openexp",
        "ord_hte_condition",
    ]
    assert catalog.get("condition", "uspto_500").targets == ("reagent",)
    assert set(catalog.available_names("yield")) == {
        "buchwald_hartwig_ood",
        "hitea",
        "suzuki_miyaura",
        "uspto_yield",
    }


def test_catalog_rejects_unknown_task_and_dataset():
    with pytest.raises(ValueError, match="unsupported task"):
        Task.normalize("unknown")
    with pytest.raises(KeyError, match="unknown dataset"):
        DatasetCatalog().get("property", "missing")


def test_local_source_loader_projection_filter_and_batches(tmp_path):
    task_dir = tmp_path / "classification"
    task_dir.mkdir()
    path = task_dir / "schneider_b.csv.gz"
    frame = pd.DataFrame(
        {
            "r_id": ["r1", "r2", "r3"],
            "rxn": ["A>>B", "C>>D", "E>>F"],
            "label": [1, 2, 3],
            "split": ["train", "test", "test"],
        }
    )
    frame.to_csv(path, index=False, compression="gzip")

    source = LocalSource(tmp_path)
    assert source.available_names("class") == ["schneider_b"]
    assert source.resolve("classification", "schneider_b") == path

    loader = DataLoader("class", source="local", data_dir=tmp_path, cache_dir=None)
    loaded = loader.load(
        "schneider_b",
        columns=["r_id", "split"],
        filters={"split": "test"},
    )
    assert loaded.to_dict("records") == [
        {"r_id": "r2", "split": "test"},
        {"r_id": "r3", "split": "test"},
    ]
    assert loader.describe("schneider_b").title == "Schneider balanced classification"
    assert [len(batch) for batch in loader.iter_batches("schneider_b", batch_size=2)] == [2, 1]


def test_local_loader_nrows_and_errors(tmp_path):
    task_dir = tmp_path / "property"
    task_dir.mkdir()
    with gzip.open(task_dir / "rgd1.csv.gz", "wt", encoding="utf8") as handle:
        handle.write("r_id,aam,ea,split\n1,A>>B,1.0,train\n2,C>>D,2.0,test\n")
    loader = DataLoader("property", source="local", data_dir=tmp_path, cache_dir=None)
    assert len(loader.load("rgd1", nrows=1)) == 1
    with pytest.raises(KeyError, match="filter column"):
        loader.load("rgd1", columns=["r_id"], filters={"split": "train"})
    with pytest.raises(FileNotFoundError, match="Local dataset"):
        loader.load("missing")


def test_cache_manager_namespaces_and_writes_atomically(tmp_path):
    cache = CacheManager(tmp_path)
    first = cache.artifact_path("github", "v1.3.0", "aam", "sample")
    second = cache.artifact_path("github", "v2.0.0", "aam", "sample")
    assert first != second
    cache.write_atomic(first, b"payload")
    assert first.read_bytes() == b"payload"
    assert not list(first.parent.glob("tmp*"))


def test_sharealike_artifacts_are_physically_separated():
    """ShareAlike obligations must not leak into the main partition."""
    catalog = DatasetCatalog()
    sharealike = [item for item in catalog.list() if item.is_sharealike]
    assert {item.name for item in sharealike} == {"openexp", "ord_hte_condition"}
    for item in sharealike:
        assert item.task is Task.CONDITION_SHAREALIKE
        assert item.license_spdx == "CC-BY-SA-4.0"
        assert item.dataset_id.key.startswith("condition_sharealike/")
    for item in catalog.list():
        # A CC BY-SA artifact may never sit in a non-ShareAlike directory.
        assert (item.license_spdx == "CC-BY-SA-4.0") is item.is_sharealike
        assert (item.task is Task.CONDITION_SHAREALIKE) is item.is_sharealike


def test_permission_pending_dataset_is_not_redistributable():
    catalog = DatasetCatalog()
    suzuki = catalog.get("yield", "suzuki_miyaura")
    assert suzuki.redistribution_partition == "excluded"
    assert suzuki.is_redistributable is False
    cyclic = catalog.get("aam", "metambd_cyclic")
    assert cyclic.redistribution_partition == "excluded"
    assert cyclic.license_spdx == "LicenseRef-Unknown"
    assert cyclic.target_kinds["ground_truth"] == "heavy_atom_mapping"
    assert not catalog.get("synthesis", "da").is_redistributable
    assert all(item.is_redistributable for item in catalog.list()
               if item.name not in {"suzuki_miyaura", "metambd_cyclic", "da"})


def test_every_dataset_declares_machine_readable_licensing():
    catalog = DatasetCatalog()
    for item in catalog.list():
        assert item.license_spdx
        # Only the unresolved permission-pending source may omit a license URL.
        assert item.license_url or item.license_spdx == "LicenseRef-Unknown"


def test_property_units_are_source_specific_even_for_shared_target_names():
    catalog = DatasetCatalog()
    b97 = catalog.get("property", "b97xd3")
    rad6 = catalog.get("property", "rad6re")
    assert b97.target_units["dh"] == "kcal/mol"
    assert rad6.target_units["dh"] == "eV"
    assert b97.target_kinds["dh"] == rad6.target_kinds["dh"] == "reaction_enthalpy"
    assert catalog.get("property", "cycloadd").target_kinds["G_act"] == "activation_free_energy"
    assert catalog.get("classification", "schneider_b").target_units == {}

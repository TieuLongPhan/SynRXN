"""SIHUMIx cleanup preserves source evidence and separates reference tiers."""

from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import tarfile

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import sihumix
from synrxn.reconstruction.families.representation.atom_mapping._sources.sihumix import acquire_archive, build_bundle, prepare_records, verify_bundle
from synrxn.reconstruction.families.representation.atom_mapping._sources.sihumix_quality import source_label

ETHER = "[CH3:1][CH2:2][OH:3]>>[CH3:1][O:3][CH3:2]"
OXIDATION = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3].[H][H]"
WRONG_OXIDATION = "[CH3:1][CH2:2][OH:3]>>[CH3:2][CH:1]=[O:3].[H][H]"
README = b"This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).\n"


def record(aam, source="Curated by user", **extra):
    return {
        "name": "Synthetic test reaction",
        "selected_matching": {"aam": aam} if aam else None,
        "matching_source": source if aam else None,
        "is_transport": False,
        "is_transport_1to1": False,
        **extra,
    }


def members(first, second=None):
    result = {"README.md": README, "data/SIHUMIx_cAAMsGEMs/A.json": json.dumps({"reactions": first}).encode()}
    if second is not None:
        result["data/SIHUMIx_cAAMsGEMs/B.json"] = json.dumps({"reactions": second}).encode()
    return result


def read_csv(path):
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def test_sources_are_retained_and_only_one_dataset_name_is_registered():
    from synrxn.reconstruction.families._importers import TARGETS

    native = members(
        {
            "R_c": record(ETHER),
            "R_db": record(ETHER, "From AAM entry in ChemRecon database"),
            "R_rdt": record(ETHER, "Generated via RDT"),
            "R_none": record(None),
        }
    )
    rows, inventory = prepare_records(native)
    assert len(rows) == 3
    assert Counter(r["source"] for r in rows) == Counter({"User-curated": 1, "ChemRecon": 1, "RDT-generated": 1})
    assert inventory["excluded_by_reason"] == {"missing_mapping": 1}
    assert all(r["r_id"].startswith("sihumix:") and r["original_aam"] == ETHER for r in rows)
    assert all(r["license"] == "CC-BY-4.0" for r in rows)
    assert [name for name in TARGETS if name.startswith("sihumix")] == ["sihumix"]


def test_duplicate_organism_memberships_and_reverse_reactions_are_not_dropped():
    renumbered = "[CH3:11][CH2:12][OH:13]>>[CH3:11][O:13][CH3:12]"
    reverse = ">>".join(reversed(ETHER.split(">>")))
    rows, _ = prepare_records(
        members({"R_a": record(ETHER)}, {"R_b": record(renumbered), "R_reverse": record(reverse)})
    )
    assert len(rows) == len({r["r_id"] for r in rows}) == 3
    assert len({r["reaction_group"] for r in rows}) == 1
    assert len({r["mapping_equivalence"] for r in rows}) == 1


def test_conflicting_source_mappings_remain_source_records_and_have_different_classes():
    rows, _ = prepare_records(members({"R_a": record(OXIDATION), "R_b": record(WRONG_OXIDATION)}))
    assert len(rows) == 2
    assert len({r["mapping_equivalence"] for r in rows}) == 2
    assert all(r["hydrogen_balance_status"] == "balanced" for r in rows)


@pytest.mark.parametrize(
    "aam,reason",
    [
        (None, "missing_mapping"),
        ("[CH4:1]>>", "one_sided_boundary"),
        ("not-a-reaction", "parse_or_sanitization_failed"),
        ("[*:1][CH3:2]>>[*:1][CH3:2]", "generic_structure"),
    ],
)
def test_only_selected_structural_exclusions_are_applied(aam, reason):
    rows, inventory = prepare_records(members({"R_a": record(aam)}))
    assert rows == [] and inventory["excluded_by_reason"] == {reason: 1}


def test_hydrogen_imbalance_and_mapping_qc_failures_remain_unchanged():
    imbalanced = "[NH3:7]>>[NH4+:7]"
    invalid = "[CH3:1][OH:2]>>[CH2:2]=[O:1].[H][H]"
    rows, _ = prepare_records(members({"R_H": record(imbalanced), "R_map": record(invalid)}))
    assert len(rows) == 2
    h = next(r for r in rows if r["original_id"] == "R_H")
    assert h["original_aam"] == imbalanced
    assert h["hydrogen_balance_status"] == "imbalanced" and h["hydrogen_dH"] == h["hydrogen_dQ"] == 1
    assert h["ground_truth"] == imbalanced and h["validation_status"] == "qc_failed"
    bad = next(r for r in rows if r["original_id"] == "R_map")
    assert bad["original_aam"] == invalid and not bad["heavy_mapping_valid"]


def test_isotopes_and_stereochemistry_survive_selection():
    isotope = "[13CH3:1][CH2:2][OH:3]>>[13CH3:1][O:3][CH3:2]"
    stereo = "[CH3:1][C@H:2]([OH:3])[F:4]>>[CH3:1][C@@H:2]([OH:3])[F:4]"
    rows, _ = prepare_records(members({"R_iso": record(isotope), "R_stereo": record(stereo)}))
    assert len(rows) == 2
    assert all(r["heavy_mapping_valid"] for r in rows)
    assert "13C" in next(r for r in rows if r["original_id"] == "R_iso")["rxn"]
    assert "@" in next(r for r in rows if r["original_id"] == "R_stereo")["rxn"]


def test_unknown_mapped_source_is_rejected():
    with pytest.raises(ValueError, match="matching_source"):
        source_label("unknown new mapper", True)


@pytest.fixture
def native_fixture(tmp_path):
    data = members(
        {
            "R_curated": record(ETHER),
            "R_missing": record(None),
            "R_transport": record("[CH4:1]>>[CH4:1]", is_transport=True),
        }
    )
    data["data/Pathways/example.csv"] = b"Reaction_index,AAMs\nexample,mapping\n"
    data["code/unlicensed.py"] = b"raise RuntimeError('never execute upstream notebooks/code')\n"
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in data.items():
            info = tarfile.TarInfo("upstream/" + name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    license_text = b"Attribution 4.0 International\nSynthetic test license fixture.\n"

    def sha(value):
        return hashlib.sha256(value).hexdigest()

    source = {
        "archive_sha256": sha(archive.read_bytes()),
        "license_text_sha256": sha(license_text),
        "grant_snapshot_sha256": sha(README),
        "commit": "0" * 40,
        "repository": "https://github.com/example/dataset",
        "attribution": "Synthetic fixture",
    }
    return archive, source, license_text, data


def test_bundle_has_one_table_original_maps_native_data_and_reproducible_bytes(tmp_path, native_fixture):
    archive, source, license_text, native = native_fixture
    first, second = tmp_path / "first", tmp_path / "second"
    report = build_bundle(archive, first, source, license_text)
    build_bundle(archive, second, source, license_text)
    assert report["rows"] == 2 and report["hydrogen_completion"] is False
    checked = verify_bundle(first)
    assert checked["records"] == 2 and checked["dataset_key"] == "aam/sihumix"
    assert {p.name for p in first.glob("*.csv.gz")} == {"aam.csv.gz"}
    assert not any(k in read_csv(first / "aam.csv.gz")[0] for k in ("expanded_aam", "prepared_aam", "r0_recovery"))
    assert (first / "LICENSE-CC-BY-4.0.txt").read_bytes() == license_text
    with tarfile.open(first / "source-data.tar.gz", "r:gz") as preserved:
        assert all(name == "README.md" or name.startswith("data/") for name in preserved.getnames())
        for name, payload in native.items():
            if name.startswith("data/"):
                assert preserved.extractfile(name).read() == payload
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    with (first / "aam.csv.gz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="payload mismatch"):
        verify_bundle(first)


def test_wrong_source_and_license_pins_are_rejected(tmp_path, native_fixture):
    archive, source, license_text, _ = native_fixture
    with pytest.raises(ValueError, match="archive SHA256"):
        build_bundle(archive, tmp_path / "bad", {**source, "archive_sha256": "f" * 64}, license_text)
    with pytest.raises(ValueError, match="License text SHA256"):
        build_bundle(archive, tmp_path / "bad", source, b"wrong license")
    with pytest.raises(ValueError, match="grant snapshot SHA256"):
        build_bundle(archive, tmp_path / "bad", {**source, "grant_snapshot_sha256": "f" * 64}, license_text)


def test_download_uses_pinned_url_and_rejects_corrupt_or_partial_data(tmp_path, monkeypatch):
    archive = tmp_path / "cache/source.tar.gz"
    payload = b"synthetic archive bytes"
    source = {
        "archive_url": "https://example.com/pinned-commit.tar.gz",
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
    }
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(sihumix.urllib.request, "urlopen", download)
    acquire_archive(archive, source)
    acquire_archive(archive, source)
    assert calls == [(source["archive_url"], 60)]
    assert archive.read_bytes() == payload
    assert list(archive.parent.iterdir()) == [archive]
    changed_pin = {**source, "archive_sha256": "f" * 64}
    with pytest.raises(ValueError, match="archive SHA256"):
        acquire_archive(archive, changed_pin)
    destination = archive.with_name("invalid.tar.gz")
    with pytest.raises(ValueError, match="archive SHA256"):
        acquire_archive(destination, changed_pin)
    assert not destination.exists() and list(archive.parent.iterdir()) == [archive]

    class Interrupted(io.BytesIO):
        def read(self, size):
            raise OSError("interrupted download")

    monkeypatch.setattr(sihumix.urllib.request, "urlopen", lambda *args, **kwargs: Interrupted(payload))
    with pytest.raises(OSError, match="interrupted download"):
        acquire_archive(destination, source)
    assert not destination.exists() and list(archive.parent.iterdir()) == [archive]

"""M-CSA must retain native transitions, identity and safe archive boundaries."""
import io
import json
import subprocess
import sys
import tarfile

import pytest

from synrxn.reconstruction.families.mechanisms.mechanistic_prediction import _mcsa as mcsa, _mcsa_native as mcsa_native


def test_native_drawing_is_a_group_and_identity_transition_is_preserved():
    pytest.importorskip("synkit.Mechanism")
    adapter = mcsa
    arrows = [["LP-/Sigma+", [1], [1, 3]], ["Sigma-/LP+", [2, 3], [2]]]
    record = adapter.typed_record({"rsmi": "C>>C", "epd_lw": arrows}, {"arrow_offsets": [0, 2, 2]})
    assert len(record.steps) == 2
    assert len(record.steps[0].groups) == 1
    assert len(record.steps[0].groups[0].moves) == 2
    assert record.steps[1].groups == ()


@pytest.mark.parametrize("offsets", [[1, 2], [0, 1], [0, 2, 1, 2]])
def test_bad_native_step_boundaries_are_rejected(offsets):
    pytest.importorskip("synkit.Mechanism")
    with pytest.raises(ValueError, match="boundaries"):
        mcsa.typed_record(
            {"rsmi": "C>>C", "epd_lw": [["LP-/Sigma+", [1], [1, 3]]] * 2},
            {"arrow_offsets": offsets},
        )


def test_native_row_join_rejects_missing_intermediate_and_preserves_ids():
    adapter = mcsa
    records = {9: {"rsmi": "C>>CO", "epd": []}, 17: {"rsmi": "CO>>COO", "epd": []}}
    metadata = {key: {"entry_id": 1, "mechanism_id": 2} for key in records}
    row = {"rsmi": "C>>COO", "epd_lw": []}
    ids, _ = adapter.checked_sources(row, {"source_ids": [9, 17]}, records, metadata)
    assert ids == [9, 17]
    records[17]["rsmi"] = "CN>>COO"
    with pytest.raises(ValueError, match="missing intermediate"):
        adapter.checked_sources(row, {"source_ids": [9, 17]}, records, metadata)


@pytest.mark.parametrize("name,link", [("../escape", False), ("link", True)])
def test_source_archive_cannot_escape_output(tmp_path, name, link):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape"
            bundle.addfile(member)
        else:
            member.size = 1
            bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe archive"):
        mcsa_native.extract(archive, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_cli_rejects_more_than_two_jobs_before_creating_output(tmp_path):
    output = tmp_path / "build"
    result = subprocess.run(
        [sys.executable, "-m", "synrxn.reconstruction.families.mechanisms.mechanistic_prediction._mcsa_native", "--jobs", "3",
         "--converter-python", sys.executable, "--output-dir", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not output.exists()


def test_reusing_changed_native_asset_fails_before_export(tmp_path):
    (tmp_path / "asset.json").write_text("changed")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "recipe_sha256": "recipe",
        "outputs": [{"path": "asset.json", "sha256": "previous-bytes"}],
    }))
    with pytest.raises(ValueError, match="Changed native build asset"):
        mcsa.verify_build(tmp_path, {"recipe_sha256": "recipe"})

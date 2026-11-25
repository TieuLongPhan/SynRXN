import gzip

from synrxn.metrics.reference import transport_correspondence
from synrxn.reconstruction.provenance.source_history import load_tool

_pilot = load_tool("run_ncs_pilot")
export_side, mapping_vector, source_export = _pilot.export_side, _pilot.mapping_vector, _pilot.source_export


def test_input_export_does_not_retain_source_map_numbers():
    a, _, _ = export_side("[CH3:9][CH2:4][OH:1]")
    b, _, _ = export_side("[CH3:1][CH2:200][OH:7]")
    assert a == b
    assert ":" not in a
    mapped = "[CH3:9][CH2:4][OH:1]>>[CH3:9][CH:4]=[O:1]"
    vector = mapping_vector(mapped)
    _, p, _ = export_side(mapped.split(">")[0])
    _, q, _ = export_side(mapped.split(">")[2])
    assert sorted(transport_correspondence(vector, p, q)) == [0, 1, 2]


def test_source_ids_are_repeatable_and_references_remain_qualified_mapping_ineligible(tmp_path):
    path = tmp_path / "source.csv.gz"
    path.write_bytes(gzip.compress(b"r_id,aam\n1,C>>CO\n", mtime=0))
    row = {"r_id": "one", "aam": "[CH3:1][OH:2]>>[CH2:1]=[O:2]", "source": "patent:1", "split": "train"}
    a = source_export(path, [(0, row)], "synthesis/source.csv.gz")
    b = source_export(path, [(0, row)], "synthesis/source.csv.gz")
    assert a == b
    records, references, instances, packets, _ = a
    assert references[0].task_eligibility["mapping"] is False
    assert records[0]["qualified_mapping_eligible"] is False
    assert set(instances[0]["inputs"]) == {"reactants"}
    assert packets[0]["review_events"] == []

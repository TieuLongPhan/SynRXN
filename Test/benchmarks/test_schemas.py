import json

import pytest

from synrxn.benchmarks.contracts import ReferenceRecord, TaskContract
from synrxn.benchmarks.reference_store import read_jsonl
from synrxn.reconstruction.paths import REPO as ROOT, physical_path

jsonschema = pytest.importorskip("jsonschema")
PILOT = physical_path("ncs/artifacts/pilot-005", repo=ROOT)


@pytest.mark.parametrize("name", ["forward-contract.json", "mapping-diagnostic/contract.json"])
def test_archived_contract_matches_its_machine_and_runtime_schema(name):
    value = json.loads((PILOT / name).read_text())
    schema = json.loads(physical_path("ncs/schemas/task-contract.schema.json", repo=ROOT).read_text())
    jsonschema.Draft202012Validator(schema).validate(value)
    TaskContract.from_dict(value)


@pytest.mark.parametrize("name", ["references.jsonl", "mapping-diagnostic/references.jsonl"])
def test_archived_references_match_their_machine_and_runtime_schema(name):
    schema = json.loads(physical_path("ncs/schemas/reference.schema.json", repo=ROOT).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for value in read_jsonl(PILOT / name):
        validator.validate(value)
        ReferenceRecord.from_dict(value)

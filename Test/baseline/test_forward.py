import pytest

from synrxn.baseline.forward import ReactantRetrievalBaseline
from synrxn.benchmarks.contracts import OUTCOME_POLICY, TaskContract


def test_retrieval_baseline_rejects_test_target_and_training_test_overlap():
    contract = TaskContract("pilot", "1", "forward", ("reactants",),
                            "reported_product_multiset", OUTCOME_POLICY,
                            ("reported_source_product",), status="runnable")
    model = ReactantRetrievalBaseline(contract).fit([
        {"instance_id": "train:1", "inputs": {"reactants": "CC"}, "target": ["CO"]},
    ])
    with pytest.raises(ValueError, match="outside"):
        model.predict("test:1", {"reactants": "CC", "products": "CO"})
    with pytest.raises(ValueError, match="training"):
        model.predict("train:1", {"reactants": "CC"})
    prediction = model.predict("test:1", {"reactants": "CC"})
    assert prediction["retrieved_training_ids"] == ["train:1"]

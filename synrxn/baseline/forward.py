"""Transparent nearest-reactant retrieval control for a forward pilot."""

from __future__ import annotations

from typing import Any

from synrxn.metrics.reference import _mol, product_multiset
from synrxn.benchmarks.contracts import TaskContract


class ReactantRetrievalBaseline:
    """Return products from training reactions with the closest reactants.

    This is a retrieval control, not a trained synthesis model. Prediction
    consumes only contract-approved inputs; target information exists in fit.
    """

    def __init__(self, contract: TaskContract):
        from rdkit.Chem import rdFingerprintGenerator

        if contract.task != "forward":
            raise ValueError("retrieval baseline requires a forward contract")
        self.contract = contract
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=True)
        self.training: list[dict[str, Any]] = []
        self.fingerprints = []

    def fit(self, training: list[dict[str, Any]]) -> "ReactantRetrievalBaseline":
        if not training:
            raise ValueError("retrieval requires a nonempty training set")
        if len({row["instance_id"] for row in training}) != len(training):
            raise ValueError("duplicate training instances")
        self.training, self.fingerprints = [], []
        for row in sorted(training, key=lambda r: r["instance_id"]):
            self.contract.validate_inputs(row["inputs"])
            product_multiset(row["target"])
            self.training.append(row)
            self.fingerprints.append(self.generator.GetFingerprint(_mol(row["inputs"]["reactants"])))
        return self

    def predict(self, instance_id: str, inputs: dict[str, str]) -> dict[str, Any]:
        from rdkit import DataStructs

        self.contract.validate_inputs(inputs)
        if instance_id in {row["instance_id"] for row in self.training}:
            raise ValueError("test instance also appears in the training set")
        fp = self.generator.GetFingerprint(_mol(inputs["reactants"]))
        similarities = DataStructs.BulkTanimotoSimilarity(fp, self.fingerprints)
        ranked = sorted(range(len(similarities)),
                        key=lambda i: (-similarities[i], self.training[i]["instance_id"]))
        ranked = ranked[:self.contract.raw_candidate_limit]
        return {
            "instance_id": instance_id, "status": "ok",
            "candidates": [self.training[i]["target"] for i in ranked],
            "retrieved_training_ids": [self.training[i]["instance_id"] for i in ranked],
            "similarities": [similarities[i] for i in ranked],
        }

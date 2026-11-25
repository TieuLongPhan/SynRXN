"""Versioned task contracts and qualified reaction reference records.

Dataset source labels are annotations. Eligibility for a qualified task is
explicit and requires a traceable, nonempty accepted reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FORBIDDEN_FORWARD_INPUTS = {
    "products", "true_product_structures", "test_product_atom_maps",
    "product_derived_reaction_centres", "outcome_derived_equation_completion",
    "source_mapping", "reference", "family_signature", "template",
}
OUTCOME_POLICY = "rdkit-isomeric-product-multiset-v1"
MAPPING_POLICY = "stereo-isotope-charge-preserving-automorphisms-v1"
PRODUCT_ORIGIN_POLICY = "stereo-isotope-charge-preserving-product-origins-v1"
RESONANCE_PRODUCT_ORIGIN_POLICY = "stereo-isotope-hydrogen-preserving-resonance-product-origins-v2"


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    version: str
    task: str
    allowed_inputs: tuple[str, ...]
    target_scope: str
    equivalence_policy: str
    evidence_tiers: tuple[str, ...]
    raw_candidate_limit: int = 5
    top_k: int = 5
    max_automorphisms: int = 64
    denominator: str = "all_task_eligible_instances"
    status: str = "draft"

    def validate(self) -> None:
        if not self.task_id or not self.version:
            raise ValueError("task and version identifiers are required")
        if self.task not in {"mapping", "forward"}:
            raise ValueError("task must be mapping or forward")
        expected = OUTCOME_POLICY if self.task == "forward" else (
            PRODUCT_ORIGIN_POLICY if self.target_scope == "reported_product_atom_origin" else MAPPING_POLICY)
        supported = {expected}
        if self.task == "mapping" and self.target_scope == "reported_product_atom_origin":
            supported.add(RESONANCE_PRODUCT_ORIGIN_POLICY)
        if self.equivalence_policy not in supported:
            raise ValueError("unsupported equivalence policy")
        if self.denominator != "all_task_eligible_instances":
            raise ValueError("eligible failures cannot be removed from the denominator")
        if self.task == "forward":
            if set(self.allowed_inputs) & FORBIDDEN_FORWARD_INPUTS:
                raise ValueError("forward contract permits target-derived inputs")
            if "reactants" not in self.allowed_inputs:
                raise ValueError("forward inputs must contain reactants")
            if self.target_scope != "reported_product_multiset":
                raise ValueError("this evaluator implements reported product multisets")
        elif self.target_scope not in {"complete_correspondence", "reported_product_atom_origin"}:
            raise ValueError("unsupported mapping correspondence scope")
        if any(type(value) is not int for value in (self.raw_candidate_limit, self.top_k, self.max_automorphisms)):
            raise ValueError("generation and equivalence budgets must be integers")
        if self.raw_candidate_limit < 1 or not 1 <= self.top_k <= self.raw_candidate_limit:
            raise ValueError("candidate budget and top-k must be fixed positive integers")
        if self.max_automorphisms < 1 or not self.evidence_tiers:
            raise ValueError("equivalence budget and evidence tiers are required")
        if self.status not in {"draft", "runnable"}:
            raise ValueError("unsupported contract status")

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        self.validate()
        if self.status != "runnable":
            raise ValueError("draft contracts cannot execute")
        unexpected = set(inputs) - set(self.allowed_inputs)
        if unexpected:
            raise ValueError(f"inputs outside the contract: {sorted(unexpected)}")
        required = {"reactants", "products"} if self.task == "mapping" else {"reactants"}
        if not required <= set(inputs):
            raise ValueError(f"missing required inputs: {sorted(required - set(inputs))}")
        from rdkit import Chem
        params = Chem.SmilesParserParams()
        params.removeHs = False
        for name in required:
            if not isinstance(inputs[name], str):
                raise ValueError("declared molecule inputs must be SMILES strings")
            mol = Chem.MolFromSmiles(inputs[name], params)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError("declared molecule inputs must be parseable nonempty graphs")
            if any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
                raise ValueError("declared inputs must be map-free; supplied annotations are targets")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskContract":
        value = dict(value)
        for name in ("allowed_inputs", "evidence_tiers"):
            value[name] = tuple(value[name])
        contract = cls(**value)
        contract.validate()
        return contract


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    observation_id: str
    revision: int
    source_record_id: str
    evidence_tier: str
    resolution: str
    evidence_locators: tuple[str, ...]
    task_eligibility: dict[str, bool]
    accepted_outcomes: tuple[tuple[str, ...], ...] = ()
    accepted_correspondences: tuple[tuple[int, ...], ...] = ()
    review_events: tuple[dict[str, Any], ...] = ()
    source_error_status: str = "not_assessed"
    previous_revision: int | None = None
    is_synthetic_example: bool = False
    limitations: tuple[str, ...] = field(default_factory=tuple)
    correspondence_scope: str = "complete_correspondence"
    accepted_product_origins: tuple[tuple[int, ...], ...] = ()

    def validate(self) -> None:
        if not all((self.reference_id, self.observation_id, self.source_record_id)):
            raise ValueError("reference, observation and source identifiers are required")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("reference revision must be a positive integer")
        if self.previous_revision is not None and not 0 < self.previous_revision < self.revision:
            raise ValueError("reference revisions must move forward")
        if self.resolution not in {"accepted", "multiple_accepted", "unresolved", "diagnostic"}:
            raise ValueError("unsupported reference resolution")
        if self.correspondence_scope not in {"complete_correspondence", "reported_product_atom_origin"}:
            raise ValueError("unsupported reference correspondence scope")
        self._validate_targets()
        self._validate_evidence()

    def _validate_targets(self) -> None:
        for outcome in self.accepted_outcomes:
            if isinstance(outcome, str) or not isinstance(outcome, (list, tuple)):
                raise ValueError("each accepted outcome must be a product multiset")
            if any(not isinstance(component, str) or not component.strip() for component in outcome):
                raise ValueError("product multisets require nonempty component strings")
        for mapping in (*self.accepted_correspondences, *self.accepted_product_origins):
            if not isinstance(mapping, (list, tuple)) or any(type(i) is not int for i in mapping):
                raise ValueError("correspondences must be integer index arrays")
        if self.task_eligibility.get("forward") and not self.accepted_outcomes:
            raise ValueError("forward eligibility requires accepted outcome multisets")
        if any(not outcome for outcome in self.accepted_outcomes):
            raise ValueError("accepted outcomes cannot be empty")
        target = (self.accepted_product_origins if self.correspondence_scope == "reported_product_atom_origin"
                  else self.accepted_correspondences)
        other = (self.accepted_correspondences if self.correspondence_scope == "reported_product_atom_origin"
                 else self.accepted_product_origins)
        if other:
            raise ValueError("correspondence arrays disagree with the declared reference scope")
        if self.task_eligibility.get("mapping") and not target:
            raise ValueError("mapping eligibility requires accepted correspondences")

    def _validate_evidence(self) -> None:
        if any(type(value) is not bool for value in self.task_eligibility.values()):
            raise ValueError("task eligibility must be explicit booleans")
        if any(self.task_eligibility.values()):
            if self.is_synthetic_example or not self.evidence_locators:
                raise ValueError("eligible references need real traceable evidence")
            if self.resolution not in {"accepted", "multiple_accepted", "diagnostic"}:
                raise ValueError("unresolved references cannot be task eligible")
        if self.resolution != "diagnostic" and self.task_eligibility.get("mapping"):
            if not self.review_events:
                raise ValueError("qualified mapping references require documented review")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceRecord":
        value = dict(value)
        for name in ("evidence_locators", "review_events", "limitations"):
            value[name] = tuple(value.get(name, ()))
        for name in ("accepted_outcomes", "accepted_correspondences", "accepted_product_origins"):
            if any(not isinstance(item, (list, tuple)) for item in value.get(name, ())):
                raise ValueError("accepted alternatives must be nested arrays")
            value[name] = tuple(tuple(item) for item in value.get(name, ()))
        record = cls(**value)
        record.validate()
        return record

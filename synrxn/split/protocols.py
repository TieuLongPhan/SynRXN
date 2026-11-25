"""Scientific split specifications, independent of dataframe/chemistry libraries.

These are evaluation requirements, not claims that every table has the metadata
needed to execute every proposed transfer experiment. Group identifiers must be
provided by an audited loader; SMILES strings are never guessed to be scaffolds,
and sequence identifiers are never guessed to be homology clusters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSplitProtocol:
    task_id: str
    family_id: str
    independence_unit: str
    group_field_candidates: tuple[str, ...]
    recommended_strategy: str = "group"
    alternatives: tuple[str, ...] = ("native", "source_disjoint", "family_disjoint")
    qualification: str = "Supply audited grouping metadata; random rows are a diagnostic only."
    required_group_field_candidates: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _protocol(task: str, family: str, unit: str, fields: str, *, alternatives=(), note="", required_fields=""):
    return TaskSplitProtocol(
        task, family, unit, tuple(fields.split()),
        alternatives=tuple(dict.fromkeys(("native", *alternatives, "random_diagnostic"))),
        qualification=note or "Keep repeated observations and declared hard dependencies together; qualify source and chemistry transfer separately.",
        required_group_field_candidates=tuple(required_fields.split()),
    )


_CHEMISTRY = ("source_disjoint", "family_disjoint", "scaffold_disjoint", "group_chronological")
_CAMPAIGN = ("campaign_disjoint", "source_disjoint", "group_chronological")
_PROTEIN = ("protein_disjoint", "source_disjoint", "family_disjoint")
_PROTOCOLS = (
    _protocol("literature_extraction", "data", "source document", "document_group document_id parent_group source_document_id", alternatives=("source_disjoint", "group_chronological"), required_fields="document_group document_id source_document_id parent_group"),
    _protocol("reaction_record_curation", "data", "source document and uncorrupted parent record", "document_group document_id source_document_id parent_group parent_record_id observation_group", alternatives=_CHEMISTRY, required_fields="document_group document_id source_document_id parent_group"),
    _protocol("atom_mapping", "representation", "reaction transformation", "transformation_group reaction_group parent_group", alternatives=_CHEMISTRY, note="Keep alternative mappings, elementary/composite projections and symmetry variants of one reaction in a single split."),
    _protocol("reaction_balancing", "representation", "complete parent reaction", "parent_group parent_reaction_id transformation_group reaction_group", alternatives=_CHEMISTRY, note="All deletion/corruption variants of a complete reference reaction stay together."),
    _protocol("reaction_classification", "representation", "reaction transformation", "transformation_group reaction_group parent_group", alternatives=(*_CHEMISTRY, "stratified_group")),
    _protocol("reaction_clustering", "representation", "reaction transformation", "transformation_group reaction_group parent_group", alternatives=_CHEMISTRY, note="Split model fitting from held-out cluster evaluation; a transductive gallery requires a separately declared protocol."),
    _protocol("reaction_retrieval", "representation", "query and source document", "query_group document_group parent_group", alternatives=("source_disjoint", "family_disjoint"), note="Keep query variants and document reprints together; pin the candidate gallery separately, including any train/test gallery overlap."),
    *(_protocol(task, "transformations", "reaction transformation", "transformation_group reaction_group parent_group", alternatives=_CHEMISTRY, note="Forward, precursor and center views must share the same parent-observation split; one reference precursor set is not exhaustive chemistry.") for task in ("forward_prediction", "retrosynthesis", "reaction_center")),
    *(_protocol(task, "outcomes", "reaction or assay campaign", "reaction_group transformation_group campaign_id parent_group", alternatives=(*_CAMPAIGN, "scaffold_disjoint", "stratified_group"), note="Keep replicate measurements and matched assay views together; fit thresholds and calibration on training/validation only. Expert plausibility and generated negatives remain separate reference tiers.") for task in ("yield_conversion", "reaction_success", "reaction_feasibility")),
    *(_protocol(task, "selectivity", "substrate/reaction group", "reaction_group substrate_group transformation_group parent_group", alternatives=_CHEMISTRY) for task in ("regioselectivity", "enantioselectivity", "product_distribution")),
    _protocol("impurity_prediction", "selectivity", "reaction/condition series", "reaction_identity_group reaction_group catalyst_composition_group reaction_system", alternatives=_CHEMISTRY, note="Keep repeated mixture profiles together; unreported impurities are unknown, not negative labels."),
    _protocol("condition_recovery", "conditions", "reaction transformation", "transformation_group reaction_group parent_group", alternatives=_CHEMISTRY),
    *(_protocol(task, "conditions", "complete measured condition panel", "panel_group panel_id chemical_group reaction_group campaign_id parent_group", alternatives=_CAMPAIGN, required_fields="panel_group panel_id", note="Keep every measured candidate panel whole, including native and random diagnostic views. Other audited panel-equivalent fields or complete hard-relationship components must be declared explicitly. Campaign holdout is a separate transfer axis. Within-test sequential trials reveal only previously acquired outcomes.") for task in ("condition_ranking", "sequential_optimization")),
    _protocol("procedure_generation", "conditions", "source procedure/reaction", "procedure_group document_group transformation_group reaction_group parent_group", alternatives=_CHEMISTRY, note="Template paraphrases and procedures from one parent observation stay together; text agreement does not prove executable success."),
    *(_protocol(task, "physics", "parent reaction and computed variants", "reaction_group reactant_pair_group parent_record_id transformation_group", alternatives=_CHEMISTRY, note="Keep conformers, methods, solvent variants and forward/reverse derivatives of a parent reference together. Different phases/methods retain distinct target contexts.") for task in ("activation_barrier", "reaction_energetics", "reaction_rates")),
    _protocol("reactive_potential", "physics", "complete reactive trajectory", "trajectory_id reaction_group", alternatives=("family_disjoint", "group_chronological"), required_fields="trajectory_id", note="Atomic configurations from one trajectory stay together; energy offsets/normalization are fitted on training data only."),
    _protocol("kinetic_parameter_inference", "physics", "complete kinetic system/trajectory", "mechanism_id trajectory_id", alternatives=("family_disjoint", "group_chronological"), required_fields="mechanism_id", note="Keep simulated descendants of a parameterized system together; a tiny number of native mechanisms can make a three-way split impossible."),
    _protocol("mechanistic_prediction", "mechanisms", "complete mechanism", "mechanism_group parent_mechanism_id enzyme_group reaction_group", alternatives=(*_CHEMISTRY, "protein_disjoint"), required_fields="mechanism_group parent_mechanism_id", note="All elementary steps and alternate representations of a complete mechanism stay together; supplied enzyme groups do not establish homology separation."),
    _protocol("ts_geometry", "mechanisms", "reaction endpoint/TS triplet", "endpoint_group reaction_id trajectory_id reaction_group", alternatives=_CHEMISTRY, required_fields="endpoint_group reaction_id trajectory_id", note="Do not separate endpoint, transition-state, conformer or reverse-reaction descendants of a triplet."),
    _protocol("reaction_network_modeling", "mechanisms", "complete mechanism/network", "mechanism_id network_id trajectory_id", alternatives=("family_disjoint", "group_chronological"), required_fields="mechanism_id network_id", note="All trajectories generated from a shared network/parameter family stay together; sparse native systems may prevent a three-way split."),
    *(_protocol(task, "analysis", "complete experimental trajectory", "trajectory_id experiment_group experiment_id", alternatives=("campaign_disjoint", "source_disjoint", "group_chronological"), required_fields="trajectory_id experiment_group experiment_id", note="Keep all spectra, time windows and derived targets from one experiment together; a causal loader must also forbid future signals and future-derived templates.") for task in ("analytical_interpretation", "reaction_progress_monitoring")),
    _protocol("reaction_component_design", "design", "design campaign and parent component", "campaign_assay_group campaign_id parent_sequence_group parent_group", alternatives=(*_CAMPAIGN, "protein_disjoint", "scaffold_disjoint"), note="Separate campaigns and shared parent components; keep oracle candidate libraries and evaluation budgets explicit. Off-library proposals have no measured lookup oracle."),
    _protocol("new_reactivity_discovery", "design", "complete discovery campaign", "campaign_id parent_group reaction_group", alternatives=_CAMPAIGN, note="Offline campaign separation measures retrospective scope discovery; prospective novelty requires an independent oracle and time-resolved acquisition."),
    _protocol("reactor_scale_modeling", "process", "reactor geometry/feedstock system", "reactor_geometry_id feedstock_family reaction_system", alternatives=("entity_disjoint", "source_disjoint", "group_chronological"), note="Preserve interpolated descendants and matched scale records. A held-out reactor or scale requires explicitly supplied geometry/scale metadata and a declared transfer axis."),
    _protocol("process_control", "process", "complete control episode", "trajectory_id episode_id", alternatives=("entity_disjoint", "group_chronological"), required_fields="trajectory_id episode_id", note="Keep complete episodes together and evaluate closed-loop control on held-out initial states/systems; supervised action agreement is not a control-performance score."),
    _protocol("process_impact_assessment", "process", "feedstock/process inventory", "feedstock_family inventory_group process_group", alternatives=("entity_disjoint", "source_disjoint", "group_chronological"), note="Preserve shared inventory and interpolated descendants; process boundaries and impact endpoints must remain explicit."),
    _protocol("enzyme_activity", "biocatalysis", "enzyme/assay campaign", "sequence_group parent_sequence_group campaign_assay_group assay_group", alternatives=_PROTEIN, note="Sequence identity groups alone do not establish homology separation; transfer uses explicitly supplied protein clusters and reaction/assay contexts."),
    _protocol("enzyme_retrieval", "biocatalysis", "enzyme/reaction query group", "sequence_group reaction_group", alternatives=_PROTEIN, note="Pin query and gallery universes; joint sequence/reaction holdout requires both group fields. Unmeasured candidates are not confirmed inactive."),
    _protocol("mutation_effects", "biocatalysis", "parent enzyme and assay campaign", "parent_sequence_group campaign_assay_group campaign_id split_component", alternatives=_PROTEIN, note="Keep wild-type/mutant and shared assay descendants together; train on no measurements from a held-out parent/campaign."),
    _protocol("active_sites", "biocatalysis", "protein/mechanism component", "split_component sequence_group mechanism_group", alternatives=_PROTEIN, note="Keep residue rows of a protein and shared mechanism descendants together; protein-macro scoring complements residue counts."),
    _protocol("enzyme_kinetic_parameters", "biocatalysis", "enzyme/substrate measurement group", "native_pair_group sequence_group", alternatives=_PROTEIN, note="Repeated enzyme/substrate references stay together. Homology transfer uses a declared cluster threshold; kcat, Km and Ki are separate targets."),
)


def get_task_split_protocol(task_id: str) -> TaskSplitProtocol:
    """Return one immutable scientific requirement; reject misspelled tasks."""
    for protocol in _PROTOCOLS:
        if protocol.task_id == task_id:
            return protocol
    raise KeyError(f"Unknown scientific task: {task_id!r}")


def get_task_split_protocols() -> dict[str, TaskSplitProtocol]:
    return {protocol.task_id: protocol for protocol in _PROTOCOLS}


def split_protocol_coverage(registry: str | Path | dict | None = None) -> dict:
    """Check requirements against the source registry; no readiness is inferred."""
    if registry is None:
        registry = Path(__file__).resolve().parents[2] / "Data/metadata/reconstruction/task_registry.json"
    if isinstance(registry, (str, Path)):
        registry = json.loads(Path(registry).read_text())
    expected = {task["id"]: task for task in registry["tasks"]}
    actual = get_task_split_protocols()
    mismatched = sorted(task for task in expected.keys() & actual.keys()
                        if expected[task]["family_id"] != actual[task].family_id)
    return {
        "registry_task_count": len(expected), "split_protocol_count": len(actual),
        "missing_tasks": sorted(expected.keys() - actual.keys()),
        "extra_tasks": sorted(actual.keys() - expected.keys()),
        "family_mismatches": mismatched,
        "complete": expected.keys() == actual.keys() and not mismatched,
        "status": "Scientific requirements registered; per-table metadata qualification is still required.",
    }

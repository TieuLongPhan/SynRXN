"""Export Flower RC/MC representatives while protecting pathway-active atoms."""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

from rdkit import Chem

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, require, write_json
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower import canonical_unmapped_reaction
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_store import get_meta

COLUMNS = ("r_id", "ground_truth", "rxn", "original_id", "steps", "role")
SCHEMA = "synrxn.flower-center-selection.v2"
MAP = re.compile(r":(\d+)\]")


def pathway_endpoints(reactions: list[str], active_atoms: set[int]) -> str:
    """Remove only whole components outside the complete pathway center."""
    endpoints = (reactions[0].split(">>")[0], reactions[-1].split(">>")[1])
    sides = []
    for side in endpoints:
        kept = [component for component in side.split(".") if set(map(int, MAP.findall(component))) & active_atoms]
        require(bool(kept), "Pathway has an empty active endpoint")
        sides.append(".".join(sorted(kept)))
    return ">>".join(sides)


def _parse(text: str):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(text, parameters)
    require(molecule is not None, "Cannot parse mapped endpoint")
    maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    require(all(maps) and len(maps) == len(set(maps)), "Endpoint has missing or duplicate map IDs")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def _signature(molecule, retained: set[int]) -> tuple:
    atoms = {}
    for atom in molecule.GetAtoms():
        number = atom.GetAtomMapNum()
        if number in retained:
            atoms[number] = (
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs(includeNeighbors=True),
                atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
            )
    bonds = {}
    for bond in molecule.GetBonds():
        ends = tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
        if set(ends) <= retained:
            bonds[ends] = (bond.GetBondTypeAsDouble(), str(bond.GetStereo()))
    return atoms, bonds


def _implicit_unchanged_hydrogens(molecule, protected: set[int]):
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() not in protected:
            atom.SetIntProp("_flower_original_map", atom.GetAtomMapNum())
            atom.SetAtomMapNum(0)
    parameters = Chem.RemoveHsParameters()
    parameters.removeMapped = False
    parameters.updateExplicitCount = True
    parameters.showWarnings = False
    result = Chem.RemoveHs(copy, parameters)
    for atom in result.GetAtoms():
        if atom.GetAtomMapNum() == 0 and atom.HasProp("_flower_original_map"):
            atom.SetAtomMapNum(atom.GetIntProp("_flower_original_map"))
    return result


def normalize_pathway(active: str, active_atoms: set[int]) -> tuple[str, str, dict]:
    """Make only nonparticipating H implicit; preserve all endpoint chemistry."""
    molecules = [_parse(side) for side in active.split(">>")]
    source_maps = [{atom.GetAtomMapNum() for atom in mol.GetAtoms()} for mol in molecules]
    require(source_maps[0] == source_maps[1], "Pathway endpoint mapping is not bijective")
    protected = set(active_atoms)
    for _ in range(3):
        normalized = [_implicit_unchanged_hydrogens(mol, protected) for mol in molecules]
        remaining = [{atom.GetAtomMapNum() for atom in mol.GetAtoms()} for mol in normalized]
        if remaining[0] == remaining[1]:
            break
        protected.update(remaining[0] ^ remaining[1])
    require(remaining[0] == remaining[1], "Hydrogen normalization would break bijectivity")
    require(active_atoms <= remaining[0], "Pathway-active atoms disappeared during export")
    sides = []
    for source, target in zip(molecules, normalized):
        text = Chem.MolToSmiles(target, canonical=True, isomericSmiles=True)
        checked = _parse(text)
        require(_signature(source, remaining[0]) == _signature(checked, remaining[0]), "Endpoint chemistry changed")
        sides.append(".".join(sorted(text.split("."))))
    ground_truth = ">>".join(sides)
    details = {
        "explicit_hydrogen_maps_retained": sorted(
            atom.GetAtomMapNum() for atom in normalized[0].GetAtoms() if atom.GetAtomicNum() == 1
        ),
        "nonparticipating_hydrogens_made_implicit_per_side": len(source_maps[0] - remaining[0]),
        "all_pathway_active_atoms_preserved": True,
        "endpoint_chemistry_verified": True,
    }
    return ground_truth, canonical_unmapped_reaction(ground_truth), details


def _write_csv(path: Path, rows: list[dict], columns):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def _write_jsonlines(path: Path, rows):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())


def select_representatives(db, elementary: list[dict], seed: int, role="val"):
    """Reuse elementary representatives and choose one member per (steps, MC)."""
    selected, failed = [], []
    for row in elementary:
        selected.append(
            {
                "ground_truth": row["ground_truth"],
                "rxn": row["rxn"],
                "original_id": row["original_id"],
                "steps": 1,
                "role": role,
                "center_type": "RC",
                "center_id": row["rc_cluster_id"],
                "source_representative": row,
            }
        )
    split = "validation" if role == "val" else role
    for member, raw, steps, center in db.execute(
        "SELECT member_id,record,m.steps,m.mc FROM members m LEFT JOIN priorities p USING(member_id) "
        "WHERE status='done' AND p.member_id IS NULL"
    ):
        record = json.loads(raw)
        priority = hashlib.sha256(f"{seed}:{split}:{steps}:{record['reference_id']}:{member}".encode()).hexdigest()
        db.execute("INSERT INTO priorities VALUES(?,?,?,?)", (member, steps, center, priority))
    db.commit()
    for steps, center in db.execute("SELECT DISTINCT steps,mc FROM priorities ORDER BY steps,mc"):
        candidates = db.execute(
            "SELECT m.member_id,m.record,m.active,m.graph FROM priorities p JOIN members m USING(member_id) "
            "WHERE p.steps=? AND p.mc=? ORDER BY p.priority", (steps, center)
        )
        candidate_count = 0
        for member, raw, active, packed in candidates:
            candidate_count += 1
            record = json.loads(raw)
            cached = db.execute(
                "SELECT ground_truth,rxn,details,error FROM normalized WHERE member_id=?", (member,)
            ).fetchone()
            if cached is None:
                try:
                    atoms = {node for node, _ in json.loads(packed)["nodes"]}
                    truth, rxn, details = normalize_pathway(active, atoms)
                    cached = (truth, rxn, json.dumps(details, sort_keys=True), None)
                except Exception as error:
                    cached = (None, None, None, f"{type(error).__name__}: {error}")
                db.execute("INSERT INTO normalized VALUES(?,?,?,?,?)", (member, *cached))
                db.commit()
            truth, rxn, details, error = cached
            if error:
                continue
            selected.append(
                {
                    "ground_truth": truth,
                    "rxn": rxn,
                    "original_id": member,
                    "steps": steps,
                    "role": role,
                    "center_type": "MC",
                    "center_id": f"mc-{center:06d}",
                    "source_membership": record,
                    "source_active_endpoints": active,
                    "normalization": json.loads(details),
                }
            )
            break
        else:
            failed.append({"steps": steps, "center_id": f"mc-{center:06d}", "candidates": candidate_count})
    return selected, failed


def export_bundle(db, output: Path, elementary: list[dict], seed: int, configuration: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    role = configuration.get("role", "val")
    selected, missing = select_representatives(db, elementary, seed, role)
    rows, provenance = [], []
    for index, selection in enumerate(selected):
        row = {"r_id": index, **{key: selection[key] for key in COLUMNS if key != "r_id"}}
        rows.append(row)
        provenance.append({"r_id": index, **selection})
    _write_csv(output / "aam.csv.gz", rows, COLUMNS)
    _write_jsonlines(output / "provenance.jsonl.gz", provenance)
    _write_jsonlines(
        output / "center_templates.jsonl.gz",
        (
            {"center_type": "MC", "center_id": f"mc-{number:06d}", **json.loads(raw)}
            for number, raw in db.execute("SELECT id,graph FROM centers ORDER BY id")
        ),
    )
    _write_jsonlines(
        output / "memberships.jsonl.gz",
        (
            {
                **json.loads(raw),
                "center_type": "RC" if steps == 1 else "MC",
                "center_id": legacy if steps == 1 else (f"mc-{center:06d}" if center is not None else None),
                "status": state,
                "statistics": json.loads(stats) if stats else None,
                "error": error,
            }
            for raw, steps, legacy, center, state, stats, error in db.execute(
                "SELECT record,steps,legacy_rc,mc,status,stats,error FROM members ORDER BY id"
            )
        ),
    )
    by_steps = dict(Counter(str(row["steps"]) for row in rows))
    memberships = dict(db.execute("SELECT steps,COUNT(*) FROM members GROUP BY steps"))
    source_counts = get_meta(db, "source_counts", {})
    if source_counts:
        memberships[1] = source_counts.get("elementary_source_rows", 0)
    summary = {
        "schema": SCHEMA,
        "status": "complete" if not missing else "incomplete_selection",
        "role": role,
        "rows": len(rows),
        "by_steps": by_steps,
        "source_memberships": sum(memberships.values()),
        "memberships_by_steps": memberships,
        "center_type_counts": dict(Counter(row["center_type"] for row in provenance)),
        "failed_memberships": db.execute("SELECT COUNT(*) FROM members WHERE status='failed'").fetchone()[0],
        "unrepresented_centers": missing,
        "failed_normalization_candidates": db.execute(
            "SELECT COUNT(*) FROM normalized WHERE error IS NOT NULL"
        ).fetchone()[0],
        "duplicate_mapped_targets": len(rows) - len({row["ground_truth"] for row in rows}),
        "selection_key": "(source_role, steps, center_id); RC at one step, MC at multiple steps",
        "configuration": configuration,
        "checkpoint_implementation_history": get_meta(db, "implementation_history", []),
        "legacy_elementary_chemistry_and_selection_preserved": True,
        "added_neighbour_shell": False,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "schema": SCHEMA,
            "rows": len(rows),
            "files": {
                path.name: digest(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    return summary

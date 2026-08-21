#!/usr/bin/env python3
"""Build elementary and composite FlowER atom-mapping benchmarks.

Only ``flower_new_dataset/test.txt`` from the May-2026 corrected FlowER
Figshare archive is retained. The builder removes exact no-change termination
steps and exact duplicate mapped reactions, while requiring a complete
bijective atom map. Composite reactions connect graph roots to reachable
terminal states identified by FlowER's no-change markers and require at least
two elementary transitions. The public archive is licensed under MIT; the
NameRxn-derived OOD sets that the paper makes available only on request are not
used.

Example
-------
Run with the requested SynRXN environment::

    conda run -n synrxn python script/build_flower_test.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re
from typing import BinaryIO
import zipfile

from rdkit import Chem
import requests
from requests.adapters import HTTPAdapter
from synkit.Graph.ITS.normalize_aam import NormalizeAAM
from urllib3.util.retry import Retry


SOURCE_DOI = "10.6084/m9.figshare.32513667.v1"
SOURCE_URL = "https://doi.org/10.6084/m9.figshare.32513667.v1"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/65102568"
ARCHIVE_MD5 = "3a6af31b22a9a0a09db677249e2ec79e"
SOURCE_MEMBER = "flower_new_dataset/test.txt"
LICENSE = "MIT"
DEFAULT_ARCHIVE = Path(".cache/flower-revised-data-v1.zip")
DEFAULT_ELEMENTARY_OUTPUT = Path("Data/aam/flower_e.csv.gz")
DEFAULT_COMPOSITE_OUTPUT = Path("Data/aam/flower_c.csv.gz")
MAP_PATTERN = re.compile(r":(\d+)\]")
MAPPED_SMILES_PARSER = Chem.SmilesParserParams()
MAPPED_SMILES_PARSER.removeHs = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--elementary-output", type=Path, default=DEFAULT_ELEMENTARY_OUTPUT
    )
    parser.add_argument(
        "--composite-output", type=Path, default=DEFAULT_COMPOSITE_OUTPUT
    )
    parser.add_argument(
        "--elementary-provenance-output",
        type=Path,
        help="Optional downstream elementary JSON provenance report.",
    )
    parser.add_argument(
        "--composite-provenance-output",
        type=Path,
        help="Optional downstream composite JSON provenance report.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Processes used for SynKit hydrogen normalization.",
    )
    return parser.parse_args(argv)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(path: Path, timeout: float) -> None:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    with session.get(DOWNLOAD_URL, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
    temporary.replace(path)


def require_archive(args: argparse.Namespace) -> Path:
    if args.refresh or not args.archive.is_file():
        download_archive(args.archive, args.timeout)
    observed = file_digest(args.archive, "md5")
    if observed != ARCHIVE_MD5:
        raise ValueError(
            f"FlowER archive MD5 mismatch: expected {ARCHIVE_MD5}, got {observed}"
        )
    return args.archive


def bijective_maps(left: str, right: str) -> bool:
    left_maps = MAP_PATTERN.findall(left)
    right_maps = MAP_PATTERN.findall(right)
    return bool(left_maps) and (
        len(left_maps) == len(set(left_maps))
        and len(right_maps) == len(set(right_maps))
        and set(left_maps) == set(right_maps)
    )


def deterministic_csv_writer(path: Path) -> tuple[BinaryIO, io.TextIOWrapper, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    compressed = gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, text, csv.writer(text, lineterminator="\n")


def normalize_state(state: str) -> str:
    """Normalize participant order without changing mapped molecular graphs."""
    return ".".join(sorted(state.split(".")))


def canonical_unmapped_reaction(mapped_reaction: str) -> str:
    """Return canonical unmapped SMILES with all explicit H atoms removed."""
    canonical_sides: list[str] = []
    for side in mapped_reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        if molecule is None:
            raise ValueError("RDKit could not parse normalized reaction side")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        molecule = Chem.RemoveHs(molecule)
        canonical_sides.append(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
    if len(canonical_sides) != 2:
        raise ValueError("Reaction must contain exactly one '>>' separator")
    return ">>".join(canonical_sides)


def explicit_hydrogen_signatures(molecule: Chem.Mol) -> dict[int, tuple]:
    """Describe mapped explicit H atoms by charge, isotope, and mapped bonds."""
    signatures = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetAtomMapNum() == 0:
            continue
        neighbors = tuple(
            sorted(
                (
                    neighbor.GetAtomMapNum(),
                    str(
                        molecule.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbor.GetIdx()
                        ).GetBondType()
                    ),
                )
                for neighbor in atom.GetNeighbors()
            )
        )
        signatures[atom.GetAtomMapNum()] = (
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            neighbors,
        )
    return signatures


def remove_unchanged_explicit_hydrogens(mapped_reaction: str) -> str:
    """Remove mapped explicit H atoms whose state is identical on both sides."""
    sides = mapped_reaction.split(">>")
    if len(sides) != 2:
        raise ValueError("Reaction must contain exactly one '>>' separator")
    molecules = [
        Chem.MolFromSmiles(side, MAPPED_SMILES_PARSER) for side in sides
    ]
    if any(molecule is None for molecule in molecules):
        raise ValueError("RDKit could not parse normalized mapped reaction")
    signatures = [explicit_hydrogen_signatures(molecule) for molecule in molecules]
    unchanged_maps = {
        atom_map
        for atom_map in signatures[0].keys() & signatures[1].keys()
        if signatures[0][atom_map] == signatures[1][atom_map]
    }
    cleaned_sides = []
    for molecule in molecules:
        editable = Chem.RWMol(molecule)
        remove_indices = sorted(
            (
                atom.GetIdx()
                for atom in editable.GetAtoms()
                if atom.GetAtomicNum() == 1
                and atom.GetAtomMapNum() in unchanged_maps
            ),
            reverse=True,
        )
        for atom_index in remove_indices:
            editable.RemoveAtom(atom_index)
        cleaned = editable.GetMol()
        Chem.SanitizeMol(cleaned)
        cleaned_sides.append(
            Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)
        )
    return ">>".join(cleaned_sides)


def standardize_candidate(
    candidate: tuple[str, str, int | None],
) -> tuple[dict[str, object] | None, str | None]:
    """Normalize one mapping while retaining only reaction-center hydrogens."""
    mapped_reaction, original_id, n_steps = candidate
    try:
        normalized = NormalizeAAM().fit(
            mapped_reaction,
            fix_aam_indice=False,
        )
        normalized = remove_unchanged_explicit_hydrogens(normalized)
        left, right = normalized.split(">>", 1)
        normalized = f"{normalize_state(left)}>>{normalize_state(right)}"
        if not bijective_maps(*normalized.split(">>", 1)):
            return None, "nonBijectiveAfterStandardization"
        reaction = canonical_unmapped_reaction(normalized)
        retained_hydrogens = sum(
            atom.GetAtomicNum() == 1
            for side in normalized.split(">>")
            for atom in Chem.MolFromSmiles(side).GetAtoms()
        )
        return (
            {
                "ground_truth": normalized,
                "rxn": reaction,
                "original_id": original_id,
                "n_steps": n_steps,
                "removed_hydrogens": len(MAP_PATTERN.findall(mapped_reaction))
                - len(MAP_PATTERN.findall(normalized)),
                "retained_hydrogens": retained_hydrogens,
            },
            None,
        )
    except Exception:
        return None, "standardizationFailures"


def standardize_candidates(
    candidates: list[tuple[str, str, int | None]],
    workers: int,
    counts: Counter[str],
) -> dict[str, dict[str, object]]:
    """Standardize and deduplicate candidates in deterministic input order."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        results = map(standardize_candidate, candidates)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(standardize_candidate, candidates, chunksize=64)
    rows: dict[str, dict[str, object]] = {}
    try:
        for row, error in results:
            if error is not None:
                counts[error] += 1
                continue
            assert row is not None
            counts["nonReactingExplicitHydrogenOccurrencesRemoved"] += int(
                row.pop("removed_hydrogens")
            )
            counts["reactingExplicitHydrogenOccurrencesRetained"] += int(
                row.pop("retained_hydrogens")
            )
            mapped_reaction = str(row["ground_truth"])
            previous = rows.get(mapped_reaction)
            if previous is None:
                rows[mapped_reaction] = row
                continue
            counts["postStandardizationDuplicateRows"] += 1
            distance = row["n_steps"]
            previous_distance = previous["n_steps"]
            if (
                distance is not None
                and previous_distance is not None
                and int(distance) < int(previous_distance)
            ):
                rows[mapped_reaction] = row
    finally:
        if executor is not None:
            executor.shutdown()
    counts["outputRows"] = len(rows)
    return rows


def add_composite_paths(
    sequence_id: str | None,
    edges: list[tuple[str, str]],
    terminals: set[str],
    rows: dict[str, dict[str, object]],
    counts: Counter[str],
) -> None:
    """Add unique root-to-terminal mappings for one numeric source group."""
    if sequence_id is None or not sequence_id.isdigit():
        return
    counts["sourceNumericGroups"] += 1
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    indegree: Counter[str] = Counter()
    nodes: set[str] = set()
    for left, right in set(edges):
        adjacency[left].add(right)
        indegree[right] += 1
        nodes.update((left, right))
    roots = sorted(
        node for node in nodes if adjacency[node] and indegree[node] == 0
    )
    leaves = {node for node in nodes if indegree[node] and not adjacency[node]}
    marked_terminals = terminals & nodes
    targets = sorted(marked_terminals or leaves)
    if len(roots) > 1:
        counts["multiRootGroups"] += 1
    if len(targets) > 1:
        counts["multiTerminalGroups"] += 1

    path_index = 0
    for root in roots:
        distances = {root: 0}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                distance = distances[node] + 1
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    queue.append(neighbor)
        for target in targets:
            distance = distances.get(target)
            if distance is None:
                counts["unreachableRootTerminalPairs"] += 1
                continue
            if distance < 2:
                counts["oneStepPathsExcluded"] += 1
                continue
            if not bijective_maps(root, target):
                counts["nonBijectivePathsExcluded"] += 1
                continue
            mapped_reaction = f"{root}>>{target}"
            path_index += 1
            candidate = {
                "ground_truth": mapped_reaction,
                "original_id": f"{sequence_id}:p{path_index}",
                "n_steps": distance,
            }
            previous = rows.get(mapped_reaction)
            if previous is None:
                rows[mapped_reaction] = candidate
            else:
                counts["exactDuplicateRows"] += 1
                if distance < int(previous["n_steps"]):
                    rows[mapped_reaction] = candidate


def build(
    archive: Path,
    elementary_output: Path,
    composite_output: Path,
    workers: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    elementary_counts: Counter[str] = Counter()
    composite_counts: Counter[str] = Counter()
    sequence_steps: defaultdict[str, int] = defaultdict(int)
    sequences: set[str] = set()
    seen: set[str] = set()
    elementary_candidates: list[tuple[str, str, int | None]] = []
    composite_raw_rows: dict[str, dict[str, object]] = {}
    active_edges: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    active_terminals: defaultdict[str, set[str]] = defaultdict(set)
    elementary_temporary = elementary_output.with_name(
        f"{elementary_output.name}.tmp"
    )
    composite_temporary = composite_output.with_name(f"{composite_output.name}.tmp")
    last_numeric_line: dict[str, int] = {}
    with zipfile.ZipFile(archive) as source, source.open(SOURCE_MEMBER) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as lines:
            for line_number, line in enumerate(lines, 1):
                try:
                    sequence_id = line.rstrip("\r\n").rsplit("|", 1)[1]
                except (IndexError, ValueError):
                    continue
                if sequence_id.isdigit():
                    last_numeric_line[sequence_id] = line_number
    raw, text, writer = deterministic_csv_writer(elementary_temporary)
    composite_raw: BinaryIO | None = None
    composite_text: io.TextIOWrapper | None = None
    try:
        writer.writerow(["r_id", "ground_truth", "rxn", "original_id"])
        with zipfile.ZipFile(archive) as source, source.open(SOURCE_MEMBER) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as lines:
                for line_number, line in enumerate(lines, 1):
                    elementary_counts["sourceRows"] += 1
                    line = line.rstrip("\r\n")
                    try:
                        mapped_reaction, sequence_id = line.rsplit("|", 1)
                        left, right = mapped_reaction.split(">>", 1)
                    except ValueError:
                        elementary_counts["malformedRows"] += 1
                        continue
                    normalized_left = normalize_state(left)
                    normalized_right = normalize_state(right)
                    if sequence_id.isdigit():
                        if normalized_left == normalized_right:
                            active_terminals[sequence_id].add(normalized_left)
                            composite_counts["terminalMarkers"] += 1
                        else:
                            active_edges[sequence_id].append(
                                (normalized_left, normalized_right)
                            )
                        if line_number == last_numeric_line[sequence_id]:
                            add_composite_paths(
                                sequence_id,
                                active_edges.pop(sequence_id, []),
                                active_terminals.pop(sequence_id, set()),
                                composite_raw_rows,
                                composite_counts,
                            )
                    else:
                        composite_counts["nonNumericSourceRowsExcluded"] += 1
                    sequence_steps[sequence_id] += 1
                    step_index = sequence_steps[sequence_id]
                    sequences.add(sequence_id)
                    if left == right:
                        elementary_counts["noChangeRows"] += 1
                        continue
                    if not bijective_maps(left, right):
                        elementary_counts["nonBijectiveRows"] += 1
                        continue
                    if mapped_reaction in seen:
                        elementary_counts["exactDuplicateRows"] += 1
                        continue
                    seen.add(mapped_reaction)
                    elementary_candidates.append(
                        (mapped_reaction, f"{sequence_id}:{step_index}", None)
                    )
        print(
            f"Standardizing {len(elementary_candidates)} elementary candidates "
            f"with {workers} workers...",
            flush=True,
        )
        elementary_rows = standardize_candidates(
            elementary_candidates,
            workers,
            elementary_counts,
        )
        for r_id, row in enumerate(elementary_rows.values()):
            writer.writerow(
                [
                    r_id,
                    row["ground_truth"],
                    row["rxn"],
                    row["original_id"],
                ]
            )
        text.close()
        raw.close()
        composite_raw, composite_text, composite_writer = (
            deterministic_csv_writer(composite_temporary)
        )
        composite_writer.writerow(
            ["r_id", "ground_truth", "rxn", "original_id", "n_steps"]
        )
        composite_candidates = [
            (
                str(row["ground_truth"]),
                str(row["original_id"]),
                int(row["n_steps"]),
            )
            for row in composite_raw_rows.values()
        ]
        print(
            f"Standardizing {len(composite_candidates)} composite candidates "
            f"with {workers} workers...",
            flush=True,
        )
        composite_rows = standardize_candidates(
            composite_candidates,
            workers,
            composite_counts,
        )
        for r_id, row in enumerate(composite_rows.values()):
            composite_writer.writerow(
                [
                    r_id,
                    row["ground_truth"],
                    row["rxn"],
                    row["original_id"],
                    row["n_steps"],
                ]
            )
        composite_text.close()
        composite_raw.close()
        elementary_temporary.replace(elementary_output)
        composite_temporary.replace(composite_output)
    except Exception:
        if composite_text is not None:
            composite_text.close()
        if composite_raw is not None:
            composite_raw.close()
        text.close()
        raw.close()
        elementary_temporary.unlink(missing_ok=True)
        composite_temporary.unlink(missing_ok=True)
        raise
    elementary_counts["sourceSequences"] = len(sequences)
    step_histogram = Counter(int(row["n_steps"]) for row in composite_rows.values())
    return (
        dict(elementary_counts),
        dict(composite_counts),
        {str(key): value for key, value in sorted(step_histogram.items())},
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive = require_archive(args)
    elementary_counts, composite_counts, step_histogram = build(
        archive,
        args.elementary_output,
        args.composite_output,
        args.workers,
    )
    common_provenance = {
        "builder": Path(__file__).name,
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "source": "FlowER",
        "sourceDoi": SOURCE_DOI,
        "sourceUrl": SOURCE_URL,
        "sourceArchiveFile": "data.zip",
        "sourceArchiveMd5": ARCHIVE_MD5,
        "sourceArchiveMember": SOURCE_MEMBER,
        "license": LICENSE,
        "synkitVersion": version("synkit"),
    }
    elementary_provenance = {
        **common_provenance,
        "dataset": "FlowER revised test-set elementary atom mappings",
        "selection": {
            "split": "test",
            "revisedDatasetOnly": True,
            "nameRxnOodSetsIncluded": False,
            "exactNoChangeStepsExcluded": True,
            "exactMappedDuplicatesExcluded": True,
            "completeBijectiveAtomMapRequired": True,
            "reactingExplicitHydrogensPreserved": True,
            "nonReactingExplicitHydrogensMadeImplicit": True,
            "rxnCanonicalUnmapped": True,
            "rxnRemovableExplicitHydrogensMadeImplicit": True,
            "rxnStandaloneReactingHydrogenSpeciesPreserved": True,
        },
        "schemaColumns": ["r_id", "ground_truth", "rxn", "original_id"],
        "counts": elementary_counts,
        "output": str(args.elementary_output),
        "outputSha256": file_digest(args.elementary_output),
    }
    composite_provenance = {
        **common_provenance,
        "dataset": "FlowER revised test-set composite atom mappings",
        "selection": {
            "split": "test",
            "revisedDatasetOnly": True,
            "nameRxnOodSetsIncluded": False,
            "numericOverallReactionGroupsOnly": True,
            "construction": "mapped-state graph root to reachable terminal",
            "terminalDefinition": "upstream no-change marker; graph leaf fallback",
            "minimumElementarySteps": 2,
            "componentOrderNormalized": True,
            "exactMappedDuplicatesExcluded": True,
            "completeBijectiveAtomMapRequired": True,
            "reactingExplicitHydrogensPreserved": True,
            "nonReactingExplicitHydrogensMadeImplicit": True,
            "rxnCanonicalUnmapped": True,
            "rxnRemovableExplicitHydrogensMadeImplicit": True,
            "rxnStandaloneReactingHydrogenSpeciesPreserved": True,
        },
        "schemaColumns": [
            "r_id",
            "ground_truth",
            "rxn",
            "original_id",
            "n_steps",
        ],
        "counts": composite_counts,
        "stepCountHistogram": step_histogram,
        "output": str(args.composite_output),
        "outputSha256": file_digest(args.composite_output),
    }
    if args.elementary_provenance_output is not None:
        args.elementary_provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.elementary_provenance_output.write_text(
            json.dumps(elementary_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.composite_provenance_output is not None:
        args.composite_provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.composite_provenance_output.write_text(
            json.dumps(composite_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"Saved {elementary_counts['outputRows']} elementary reactions to "
        f"{args.elementary_output}"
    )
    print(
        f"Saved {composite_counts['outputRows']} composite reactions to "
        f"{args.composite_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

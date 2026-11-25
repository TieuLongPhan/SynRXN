"""Generate the paper's native TikZ hierarchy from the scientific registry.

The thesis style is snapshotted below, so the thesis checkout is not required.
pdflatex and pdftocairo are needed only for rendering; --no-build writes sources.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from synrxn.reconstruction.catalogue.release_scope import CATALOGUE, MAIN, OPTIONAL_TASKS, SCOPES, select_scope
from synrxn.reconstruction.presentation.manuscript import latex
from synrxn.reconstruction.paths import REPO, physical_path


FAMILY_LABELS = {
    "data": ("Data acquisition", "& curation"),
    "representation": ("Representation", "& search"),
    "transformations": ("Single-step", "transformations"),
    "outcomes": ("Outcomes", "& feasibility"),
    "selectivity": ("Selectivity", "& mixtures"),
    "conditions": ("Conditions", "& procedures"),
    "physics": ("Physics", "& kinetics"),
    "mechanisms": ("Mechanisms", "& networks"),
    "analysis": ("Analysis", "& monitoring"),
    "design": ("Components", "& new reactivity"),
    "process": ("Process", "engineering"),
    "biocatalysis": ("Biocatalysis", "& enzyme function"),
}
TASK_LABELS = {
    "literature_extraction": "Literature extraction",
    "reaction_record_curation": "Record validation & correction",
    "atom_mapping": "Atom mapping",
    "reaction_balancing": "Completion & balancing",
    "reaction_classification": "Reaction classification",
    "reaction_clustering": "Reaction clustering",
    "reaction_retrieval": "Similarity & retrieval",
    "forward_prediction": "Forward products",
    "retrosynthesis": "Single-step precursors",
    "reaction_center": "Reaction centers",
    "yield_conversion": "Yield, response & conversion",
    "reaction_success": "Experimental success",
    "reaction_feasibility": "Proposal feasibility",
    "regioselectivity": "Regioselectivity",
    "enantioselectivity": "Enantioselectivity",
    "product_distribution": "Product distributions",
    "impurity_prediction": "Side products & impurities",
    "condition_recovery": "Condition recovery",
    "condition_ranking": "Measured-condition ranking",
    "sequential_optimization": "Experiment selection",
    "procedure_generation": "Procedure generation",
    "activation_barrier": "Activation barriers",
    "reaction_energetics": "Reaction thermochemistry",
    "reaction_rates": "Reaction rates",
    "reactive_potential": "Reactive energies & forces",
    "kinetic_parameter_inference": "Kinetic parameter inference",
    "mechanistic_prediction": "Electron-pushing mechanisms",
    "ts_geometry": "Transition-state geometry",
    "reaction_network_modeling": "Elementary networks",
    "analytical_interpretation": "Mixture interpretation",
    "reaction_progress_monitoring": "Progress & endpoint monitoring",
    "reaction_component_design": "Conditioned component design",
    "new_reactivity_discovery": "New reactivity & scope",
    "reactor_scale_modeling": "Reactor & scale effects",
    "process_control": "Dynamic process control",
    "process_impact_assessment": "Resource & impact assessment",
    "enzyme_activity": "Enzyme/assay activity",
    "enzyme_retrieval": "Enzyme-reaction retrieval",
    "mutation_effects": "Mutation effects",
    "active_sites": "Conditioned active sites",
    "enzyme_kinetic_parameters": "Enzyme kinetic parameters",
}
# Display order only: two columns read from electron-level to enzyme-level
# families.  The registry itself has no ordering or domain level.
ATLAS_ORDER = (
    ("mechanisms", "physics", "transformations", "representation", "selectivity"),
    ("outcomes", "conditions", "analysis", "data", "design", "process", "biocatalysis"),
)
STYLE = {
    "schema": "synrxn.thesis-figure-style-snapshot.v4",
    "origins": {
        "PhD_thesis/chapters/shared/style/tokens.json":
            "a1a4770d48a7b416fc5f07dabf81e5963b9a83c337ae24ce6af36351344fb4b5",
        "PhD_thesis/chapters/shared/style/figure_style.tex":
            "4e040eea298dfcc7bf8779f8d958315d019c0770c5075bb3bc584b880370f542",
        "PhD_thesis/chapters/11/fig/synrxn_layout.tex":
            "3016a990fba8eb3ccfefac265ad761061c2d6ef387fe2708460c65db269f8040",
        "PhD_thesis/chapters/11/fig/synrxn_concept.tex":
            "thesis Figure 11.1 panel/badge/box conventions",
    },
    "colors": {
        "ThesisNavy": "17324D", "ThesisBlue": "0072B2", "ThesisSky": "56B4E9",
        "ThesisTeal": "009E73", "ThesisVermilion": "D55E00", "ThesisPurple": "CC79A7",
        "ThesisText": "1A1A1A", "ThesisMuted": "666666", "ThesisRule": "D9D9D9",
        "ThesisPanel": "F7FAFC",
    },
    "metrics": {
        "width_cm": 17.78, "height_cm": 11.60, "body_pt": 6.6,
        "title_pt": 8.6, "rule_pt": 0.55, "corner_pt": 2.5, "badge_mm": 4.6,
    },
    "adaptation": "Periodic-table layout at final KDD width: twelve family columns in reading order from electrons to enzymes, each family coloured along one navy-to-purple spectrum derived from the thesis palette, task tiles shaded by bound table views. No panel frames, badges or bullets.",
}
NCS_ID = ("canonical", "yield/buchwald_hartwig_ood")
MINOT_IDS = {
    ("expansion", "C02/minot_al1_data"), ("expansion", "C02/minot_al2_data"),
    ("expansion", "C02/minot_al3_data"), ("expansion", "C02/minot_preal_data"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_inventory(registry, hierarchy, *, scope=CATALOGUE):
    """Reject incomplete labels, stale bindings and conflicting source identities."""
    families = registry["families"]
    tasks = {task["id"]: task for task in registry["tasks"]}
    family_ids = [family["id"] for family in families]
    if len(family_ids) != len(set(family_ids)) or set(family_ids) != set(FAMILY_LABELS):
        raise ValueError("Figure family labels do not cover the registry exactly")
    labels = set(TASK_LABELS) - (OPTIONAL_TASKS if scope == MAIN else set())
    if len(tasks) != len(registry["tasks"]) or set(tasks) != labels:
        raise ValueError("Figure task labels do not cover the registry exactly")
    expected = {}
    assignments = []
    for family in families:
        ids = family["task_ids"]
        assignments.extend(ids)
        for task_id in ids:
            if tasks[task_id]["family_id"] != family["id"]:
                raise ValueError("Registry family/task assignment disagrees")
    if len(assignments) != len(tasks) or set(assignments) != set(tasks):
        raise ValueError("Every scientific task must appear in exactly one family")
    for task_id, task in tasks.items():
        for table in task["bindings"]:
            identity = (task_id, table["namespace"], table["key"])
            if identity in expected:
                raise ValueError("Duplicate registry task/table binding")
            expected[identity] = (table["rows"], table["role"])
    tables, shown, hierarchy_families, task_views = {}, {}, {}, {}
    for family in hierarchy["families"]:
        hierarchy_families[family["id"]] = [task["id"] for task in family["tasks"]]
        for task in family["tasks"]:
            task_views[task["id"]] = len(task["tables"])
            for table in task["tables"]:
                identity = (table["namespace"], table["key"])
                if identity in tables and any(
                    tables[identity][field] != table[field]
                    for field in ("rows", "role", "path", "sha256")
                ):
                    raise ValueError("Conflicting source-table identity: " + repr(identity))
                tables[identity] = table
                binding = (task["id"], *identity)
                if binding in shown:
                    raise ValueError("Duplicate hierarchy task/table binding")
                shown[binding] = (table["rows"], table["role"])
    if hierarchy_families != {family["id"]: family["task_ids"] for family in families}:
        raise ValueError("Hierarchy family/task assignments disagree with registry")
    if expected != shown:
        raise ValueError("Registry and hierarchy task/table bindings disagree")
    summary = {
        "families": len(families), "tasks": len(tasks), "catalogue_tables": len(tables),
        "unique_primary_tables": sum(t["role"] != "overlapping_view" for t in tables.values()),
        "overlapping_views": sum(t["role"] == "overlapping_view" for t in tables.values()),
        "all_task_table_bindings": len(shown),
    }
    if any(hierarchy["summary"].get(key) != value for key, value in summary.items()):
        raise ValueError("Hierarchy summary is stale")
    success = {
        (b["namespace"], b["key"]) for b in tasks["reaction_success"]["bindings"]
    }
    if success != {NCS_ID} | MINOT_IDS:
        raise ValueError("Expanded success example needs an updated table grouping")
    domain_ids = [fid for ids in ATLAS_ORDER for fid in ids]
    if sorted(domain_ids) != sorted(family_ids):
        raise ValueError("Figure order must place every family exactly once")
    return {
        "summary": summary,
        "families": [{
            "id": family["id"], "title": family["title"],
            "display_title": list(FAMILY_LABELS[family["id"]]),
            "tasks": [{"id": tid, "title": tasks[tid]["title"],
                       "display_label": TASK_LABELS[tid],
                       "table_views": task_views[tid]} for tid in family["task_ids"]],
        } for family in families],
        "example": {
            "family_id": "outcomes", "task_id": "reaction_success",
            "ncs": tables[NCS_ID],
            "ncs_table_views": len(success & {NCS_ID}),
            "minot_tables": [tables[key] for key in sorted(MINOT_IDS)],
            "minot_rows": sum(tables[key]["rows"] for key in MINOT_IDS),
            "table_views": len(success),
            "stored_observations": sum(tables[key]["rows"] for key in success),
            "qualification": "Stored observations; native success-label definitions remain source-specific.",
        },
    }


def number(value):
    return f"{value:,}".replace(",", r"\,")


SPECTRUM = ("17324D", "0072B2", "009E73", "B07800", "D55E00", "8E4A8A")
# Font Awesome 5 glyphs, one per family (display only).
FAMILY_ICONS = {
    "mechanisms": "atom", "physics": "tachometer-alt", "transformations": "flask",
    "representation": "search", "selectivity": "filter",
    "outcomes": "check-circle", "conditions": "thermometer-half",
    "analysis": "microscope", "data": "database", "design": "lightbulb",
    "process": "industry", "biocatalysis": "dna",
}
SHADES = ((1, 1, 14), (2, 3, 30), (4, 5, 48), (6, 9, 68), (10, None, 88))


def spectrum(n):
    """Interpolate n colours along the electron-to-enzyme display spectrum."""
    anchors = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in SPECTRUM]
    colours = []
    for k in range(n):
        t = k / (n - 1) * (len(anchors) - 1)
        i = min(int(t), len(anchors) - 2)
        f = t - i
        rgb = (round(a + (b - a) * f) for a, b in zip(anchors[i], anchors[i + 1]))
        colours.append("".join(f"{c:02X}" for c in rgb))
    return colours


def shade(views):
    return next(p for low, high, p in SHADES if views >= low and (high is None or views <= high))


def ink(colour, percent):
    """White text only where the tinted tile is dark enough (WCAG luminance)."""
    channels = []
    for i in (0, 2, 4):
        c = (int(colour[i:i + 2], 16) * percent + 255 * (100 - percent)) / 100 / 255
        channels.append(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4)
    luminance = .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2]
    return "white" if luminance < .30 else "ThesisText"


def render_tikz(inventory):
    """Draw the atlas as a periodic-table layout at final KDD width.

    Families are columns in reading order from electron-level to enzyme-level
    questions, each with an icon and one colour along a spectrum; each task
    tile is shaded by the number of catalogue table views (datasets) bound to
    it.  Tile size is fixed, so text size never depends on the number of leaves.
    """
    families = {family["id"]: family for family in inventory["families"]}
    order = [fid for column in ATLAS_ORDER for fid in column]
    per_row = 6
    colours = dict(zip(order, spectrum(len(order))))
    W, gap, tile_gap = 17.78, .10, .055
    width = (W - (per_row - 1) * gap) / per_row
    head, tile = .70, .52
    rows = (len(order) + per_row - 1) // per_row
    depth = [max(len(families[fid]["tasks"]) for fid in order[r * per_row:(r + 1) * per_row])
             for r in range(rows)]
    row_top, y = [], 0
    for r in range(rows):
        row_top.append(y)
        y -= head + tile_gap + depth[r] * (tile + tile_gap) + .22
    H = round(-y - .22, 2)
    lines = [
        "% Generated by synrxn.reconstruction.presentation.figures; edit the generator, not this file.",
        r"\begingroup",
    ]
    lines.extend(r"\definecolor{" + name + "}{HTML}{" + value + "}"
                 for name, value in STYLE["colors"].items())
    lines.extend(r"\definecolor{SX" + fid + "}{HTML}{" + value + "}" for fid, value in colours.items())
    nohyphen = r"},execute at begin node={\hyphenpenalty=10000 \exhyphenpenalty=10000 "
    lines.extend([
        r"\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]",
        rf"\path[use as bounding box] (0,0) rectangle ({W},-{H});",
        r"\tikzset{",
        r" sx head/.style={anchor=west,align=left,inner sep=0pt,text=white,font=\sffamily\bfseries\fontsize{7.0}{8.0}\selectfont" + nohyphen + "},",
        r" sx icon/.style={circle,fill=white,inner sep=0pt,minimum size=5.0mm,font=\fontsize{8.4}{8.4}\selectfont},",
        r" sx task/.style={anchor=west,align=left,inner sep=0pt,font=\sffamily\fontsize{6.4}{7.2}\selectfont" + nohyphen + "},",
        r" sx num/.style={anchor=east,inner sep=0pt,font=\sffamily\bfseries\fontsize{6.6}{6.6}\selectfont},",
        r"}",
    ])
    for index, fid in enumerate(order):
        r, c = divmod(index, per_row)
        family = families[fid]
        left = c * (width + gap)
        right = left + width
        top = row_top[r]
        colour = "SX" + fid
        label = r"\\".join(latex(part) for part in family["display_title"])
        lines.extend([
            f"% family: {fid}",
            rf"\fill[{colour}] ({left:.3f},{top:.3f}) rectangle ({right:.3f},{top - head:.3f});",
            rf"\node[sx icon,text={colour}] at ({left + .34:.3f},{top - head / 2:.3f}) {{\faIcon{{{FAMILY_ICONS[fid]}}}}};",
            rf"\node[sx head] at ({left + .66:.3f},{top - head / 2:.3f}) {{{label}}};",
        ])
        y = top - head - tile_gap
        for task in family["tasks"]:
            views = task["table_views"]
            p = shade(views)
            text = ink(colours[fid], p)
            lines.extend([
                "% task: " + task["id"],
                rf"\fill[{colour}!{p}] ({left:.3f},{y:.3f}) rectangle ({right:.3f},{y - tile:.3f});",
                rf"\node[sx task,text={text},text width={width - .62:.2f}cm] at ({left + .12:.3f},{y - tile / 2:.3f}) {{{latex(task['display_label'])}}};",
                rf"\node[sx num,text={text}] at ({right - .10:.3f},{y - tile / 2:.3f}) {{{views}}};",
            ])
            y -= tile + tile_gap
    lines.extend([
        r"\end{tikzpicture}",
        r"\endgroup",
        "",
    ])
    return "\n".join(lines)


STANDALONE = r"""% Test-compile wrapper; the paper inputs hierarchy.tex directly.
\documentclass[tikz,border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{fontawesome5}
\usetikzlibrary{arrows.meta}
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=15
\begin{document}
\input{hierarchy.tex}
\end{document}
"""


def generate(repo=REPO, output=None, *, build=True, preview=None, scope=CATALOGUE):
    """Write paper/figure/hierarchy.tex; optionally test-compile it.

    The paper inputs hierarchy.tex directly, so it is the only file kept next
    to the manuscript.  The test compile, PDF/PNG preview and provenance record
    go to .cache/synrxn/figures.
    """
    repo = Path(repo).resolve()
    output = Path(output).resolve() if output else repo / "paper/figure"
    preview = Path(preview).resolve() if preview else repo / ".cache/synrxn/figures"
    names = ("kdd/task_registry.json", "kdd/hierarchy/hierarchy.json")
    registry, hierarchy = [json.loads(physical_path(name, repo).read_text()) for name in names]
    registry, hierarchy = select_scope(registry, hierarchy, scope)
    inventory = build_inventory(registry, hierarchy, scope=scope)
    output.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    (output / "hierarchy.tex").write_text(render_tikz(inventory))
    tools = {}
    epoch = int(datetime.fromisoformat(registry["as_of"]).replace(tzinfo=timezone.utc).timestamp())
    if build:
        for tool in ("pdflatex", "pdftocairo"):
            if not shutil.which(tool):
                raise RuntimeError("Figure rendering requires " + tool + "; use --no-build for TeX sources")
        work = Path(tempfile.mkdtemp(prefix="synrxn-figure-"))
        (work / "hierarchy_standalone.tex").write_text(STANDALONE)
        shutil.copyfile(output / "hierarchy.tex", work / "hierarchy.tex")
        env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch), FORCE_SOURCE_DATE="1")
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                 "-file-line-error", "hierarchy_standalone.tex"],
                                cwd=work, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Figure rendering failed:\n" + result.stdout[-10000:])
        if "Overfull" in (work / "hierarchy_standalone.log").read_text(errors="replace"):
            raise RuntimeError("Hierarchy figure has overfull TeX boxes; inspect " + str(work))
        shutil.copyfile(work / "hierarchy_standalone.pdf", preview / "hierarchy.pdf")
        subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "200",
                        str(preview / "hierarchy.pdf"), str(preview / "hierarchy")], check=True)
        tools["pdflatex"] = subprocess.run(["pdflatex", "--version"], text=True,
                                           stdout=subprocess.PIPE).stdout.splitlines()[0]
        shutil.rmtree(work, ignore_errors=True)
    manifest = {
        "schema": "synrxn.hierarchy-figure-provenance.v2",
        "input_sha256": {name: sha256(physical_path(name, repo)) for name in names},
        "generator": {"path": "synrxn/reconstruction/presentation/figures.py", "sha256": sha256(__file__)},
        "style": STYLE,
        "release_scope": scope,
        "inventory": inventory,
        "rendering": {"test_compiled": build, "source_date_epoch": epoch, "tools": tools},
        "output_sha256": {"hierarchy.tex": sha256(output / "hierarchy.tex")},
    }
    (preview / "hierarchy_provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "paper/figure")
    parser.add_argument("--no-build", action="store_true", help="Write the TikZ source without a test compile")
    parser.add_argument("--scope", choices=SCOPES, default=MAIN)
    args = parser.parse_args(argv)
    manifest = generate(output=args.output, build=not args.no_build, scope=args.scope)
    print(json.dumps(manifest["inventory"]["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

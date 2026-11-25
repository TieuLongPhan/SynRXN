"""Audit chemical novelty, class balance and label semantics in BH-HTE-OOD.

The native and row-random partitions use the same population and paired test
sizes. These descriptive statistics do not treat correlated folds as independent
experimental replications or estimate predictive model performance.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from synrxn.reconstruction.paths import physical_path
from synrxn.reconstruction.families.outcomes.reaction_success._bh_hte_ood import (
    CANONICAL_SHA256, SOURCE_COMMIT, SOURCE_SHA256, SOURCE_URL,
    decode_assignments, join_native_splits, load_native, verify_hash,
)

IDENTITY_COLUMNS = ("Aryl SMILES", "Amine SMILES", "Catalyst SMILES", "Base SMILES", "Solvent SMILES")
NOVELTY = ("unseen_context", "unseen_pair", "both_substrates_unseen")


def population(rows):
    """Encode exact author-curated entity identities and native binary targets.

    :param rows: Verified native rows in canonical record order.
    :return: Reaction contexts, substrate identities, targets and source rows.
    """
    contexts = [tuple(row[key] for key in IDENTITY_COLUMNS) for row in rows]
    if any(not value for context in contexts for value in context):
        raise ValueError("Missing chemical identity in audit population")
    if any(row["labels"] not in {"0", "1"} for row in rows):
        raise ValueError("Audit requires native binary labels")
    return {"contexts": contexts, "pairs": [context[:2] for context in contexts],
            "aryl": [context[0] for context in contexts], "amine": [context[1] for context in contexts],
            "labels": np.array([int(row["labels"]) for row in rows]), "rows": rows}


def partition_statistics(data, test):
    """Count test novelty relative to training under nested identity definitions.

    :param data: Encoded population from :func:`population`.
    :param test: Boolean mask identifying the held-out observations.
    :return: Numerators, denominators and fractions for one partition.
    """
    test = np.asarray(test, dtype=bool)
    if test.shape != data["labels"].shape or not test.any() or test.all():
        raise ValueError("Partition must assign nonempty, disjoint train and test populations")
    training, held_out = np.flatnonzero(~test), np.flatnonzero(test)
    aryl = {data["aryl"][index] for index in training}
    amine = {data["amine"][index] for index in training}
    pairs = {data["pairs"][index] for index in training}
    contexts = {data["contexts"][index] for index in training}
    counts = {
        "unseen_context": sum(data["contexts"][index] not in contexts for index in held_out),
        "unseen_pair": sum(data["pairs"][index] not in pairs for index in held_out),
        "both_substrates_unseen": sum(
            data["aryl"][index] not in aryl and data["amine"][index] not in amine for index in held_out),
        "positive": int(data["labels"][test].sum()),
    }
    return {"train_rows": len(training), "test_rows": len(held_out), "test_fraction": float(test.mean()),
            "test_membership_sha256": hashlib.sha256(np.packbits(test).tobytes()).hexdigest(),
            **{key + "_rows": int(value) for key, value in counts.items()},
            **{key + "_fraction": float(value / len(held_out)) for key, value in counts.items()}}


def audit_partitions(data, splits, seed=20261022, iterations=10):
    """Pair each native partition with a label-independent random split of equal size.

    :param data: Encoded canonical population.
    :param splits: Verified source assignments in the same record order.
    :param seed: Master PCG64 seed for the row-random controls.
    :param iterations: Number of native clustering initializations.
    :return: Two records per outer partition, including all denominators.
    """
    if len(splits) != len(data["rows"]):
        raise ValueError("Split metadata and observations must have identical membership")
    results = []
    for iteration in range(iterations):
        assignments = [decode_assignments(row[f"iteration_{iteration} train_test"]) for row in splits]
        for cycle in range(4):
            native = np.array([assignment[cycle] == "test" for assignment in assignments])
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, iteration, cycle])))
            control = np.zeros(len(native), dtype=bool)
            control[rng.choice(len(native), size=int(native.sum()), replace=False)] = True
            for method, mask in (("native_cluster", native), ("matched_row_random", control)):
                results.append({"iteration": iteration, "cycle": cycle, "method": method,
                                **partition_statistics(data, mask)})
    return results


def label_audit(rows):
    """Compare native labels with two threshold rules without replacing the reference.

    :param rows: Verified native observations.
    :return: Per-source row counts, positives and threshold disagreement counts.
    """
    records = []
    for source in sorted({row["Source"] for row in rows}):
        subset = [row for row in rows if row["Source"] == source]
        yields = np.array([float(row["Yield"]) for row in subset])
        labels = np.array([int(row["labels"]) for row in subset])
        records.append({"source": source, "rows": len(subset), "positives": int(labels.sum()),
                        "yield_gt_10_disagreements": int(np.count_nonzero(labels != (yields > 10))),
                        "yield_ge_10_disagreements": int(np.count_nonzero(labels != (yields >= 10)))})
    return records


def _descriptive(records):
    result = {}
    for method in ("native_cluster", "matched_row_random"):
        members = [row for row in records if row["method"] == method]
        result[method] = {}
        for metric in (*[name + "_fraction" for name in NOVELTY], "positive_fraction", "test_rows"):
            values = [row[metric] for row in members]
            result[method][metric] = {"min": float(np.min(values)), "median": float(np.median(values)),
                                      "max": float(np.max(values))}
            if metric.endswith("_fraction"):
                count_key = metric.removesuffix("_fraction") + "_rows"
                result[method][metric]["pooled_test_exposure_fraction"] = (
                    sum(row[count_key] for row in members) / sum(row["test_rows"] for row in members))
    return result


def plot_audit(records, source_counts, positive_fraction, destination):
    """Plot split properties and label disagreements from the audit.

    :param records: Per-fold statistics returned by the partition audit.
    :param source_counts: Per-source label counts from the threshold audit.
    :param positive_fraction: Native positive fraction across all observations.
    :param destination: Figure path without an extension.
    :return: Paths to the PDF and PNG figure artifacts.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    policies = (
        ("native_cluster", "Native clusters", "#6325a3", "#381659"),
        ("matched_row_random", "Matched random", "#ffb32c", "#8a4d08"),
    )
    groups = {
        method: [row for row in records if row["method"] == method]
        for method, _, _, _ in policies
    }
    categories = np.arange(len(NOVELTY))
    jitter = np.random.default_rng(100)
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
    }
    with matplotlib.rc_context(style):
        fig, ((novelty, prevalence), (fold_size, labels)) = plt.subplots(
            nrows=2, ncols=2, figsize=(7.04, 3.98),
        )
        fig.subplots_adjust(left=.091, right=.979, bottom=.104, top=.910, wspace=.41, hspace=.60)
        for index, (method, _, color, ink) in enumerate(policies):
            folds = groups[method]
            positions = categories + (-.17 if index == 0 else .17)
            for x, metric in zip(positions, NOVELTY, strict=True):
                shares = 100 * np.array([row[metric + "_fraction"] for row in folds], dtype=float)
                novelty.bar(x, np.median(shares), width=.28, color=color,
                            edgecolor=ink, linewidth=.45, zorder=2)
                novelty.scatter(x + jitter.uniform(-.103, .103, len(shares)), shares,
                                s=4.8, color=ink, alpha=.55, linewidths=0, zorder=3)

            shares = 100 * np.array([row["positive_fraction"] for row in folds], dtype=float)
            prevalence.scatter(index + jitter.uniform(-.15, .15, len(shares)), shares,
                               s=8, color=color, edgecolors=ink, linewidths=.26, alpha=.75, zorder=2)
            prevalence.hlines(np.median(shares), index - .22, index + .22, color=ink,
                              linewidth=1.4, zorder=3)

            fold_size.scatter(
                [int(row["test_rows"]) / 1000 for row in folds],
                [100 * float(row["both_substrates_unseen_fraction"]) for row in folds],
                marker=("o", "s")[index], s=13, color=color, edgecolors=ink,
                linewidths=.35, alpha=.76,
            )

        novelty.set(
            xlim=(-.46, 2.46), ylim=(-2, 109),
            xticks=categories, xticklabels=("Reaction context", "Substrate pair", "Both substrates"),
            yticks=[0, 25, 50, 75, 100], ylabel="Unseen test observations (%)",
        )
        novelty.legend(
            handles=[Patch(facecolor=color, edgecolor=ink, linewidth=.45, label=label)
                     for _, label, color, ink in policies],
            frameon=False, loc="upper right", fontsize=7.0,
            handlelength=1.0, handletextpad=.45, labelspacing=.30,
        )
        prevalence.axhline(100 * positive_fraction, color="#747474", linewidth=.65,
                           linestyle=(0, (3, 2)), label=f"All observations ({positive_fraction:.1%})")
        prevalence.legend(
            frameon=False, loc="upper right", fontsize=7.3, handlelength=1.6, handletextpad=.45,
        )
        prevalence.set(
            xlim=(-.48, 1.48), ylim=(20, 65), xticks=[0, 1],
            xticklabels=[label for _, label, _, _ in policies],
            yticks=[20, 30, 40, 50, 60], ylabel="Positive test observations (%)",
        )
        fold_size.set(
            xlim=(3.8, 11.25), ylim=(-2.5, 65), xticks=[4, 6, 8, 10], yticks=[0, 20, 40, 60],
            xlabel="Test fold size (thousands of observations)", ylabel="Both substrates unseen (%)",
        )

        source_rates = sorted(
            [(row["source"],
              100 * int(row["yield_gt_10_disagreements"]) / int(row["rows"]),
              100 * int(row["yield_ge_10_disagreements"]) / int(row["rows"]))
             for row in source_counts],
            key=lambda item: -item[1],
        )
        for index, (_, strict_rate, inclusive_rate) in enumerate(source_rates):
            labels.barh(index, strict_rate, height=.45, color="#797381", zorder=2)
            labels.plot(inclusive_rate, index, marker="d", markersize=4.2,
                        markerfacecolor="white", markeredgecolor="#a45013", linewidth=0, zorder=3)
        labels.set(
            xlim=(-.14, 4.15), ylim=(len(source_rates) - .5, -.5),
            yticks=range(len(source_rates)), yticklabels=[name for name, _, _ in source_rates],
            xticks=[0, 1, 2, 3, 4], xlabel="Labels differing from the native reference (%)",
        )
        labels.legend(
            handles=[
                Patch(facecolor="#797381", label="Yield > 10%"),
                Line2D([], [], linestyle="none", marker="d", markerfacecolor="white",
                       markeredgecolor="#a45013", markersize=4.2, label="Yield ≥ 10%"),
            ],
            frameon=False, loc="lower right", fontsize=7.0, handlelength=1.0, handletextpad=.45,
        )
        for ax, letter, title in (
            (novelty, "a", "Chemical novelty"),
            (prevalence, "b", "Test-set success prevalence"),
            (fold_size, "c", "Substrate novelty by fold size"),
            (labels, "d", "Success labels by yield rule"),
        ):
            ax.tick_params(length=2.4, width=.65, pad=3)
            position = ax.get_position()
            y = position.y1 + .044
            fig.text(position.x0 - .067, y, letter, weight="bold", fontsize=11, va="center")
            fig.text(position.x0 - .035, y, title, fontsize=9, va="center")

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pdf, png = destination.with_suffix(".pdf"), destination.with_suffix(".png")
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(png, dpi=300)
        plt.close(fig)
    return [pdf, png]


def main(argv=None):
    """Reproduce the BH-HTE-OOD audit from checksum-pinned source observations.

    :param argv: Optional CLI argument list.
    :return: Zero after writing audit data, provenance and figure artifacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args(argv)
    source = physical_path("Data/raw/yield/BH_HTE_Curated_Dataset_v20260219.csv", args.repo)
    canonical = physical_path("Data/yield/buchwald_hartwig_ood.csv.gz", args.repo)
    verify_hash(source, SOURCE_SHA256)
    verify_hash(canonical, CANONICAL_SHA256)
    native = load_native(source)
    with gzip.open(canonical, "rt", encoding="utf-8", newline="") as handle:
        verified, excluded = join_native_splits(native, list(csv.DictReader(handle)))
    by_id = {row[""]: row for row in native}
    rows = [by_id[row["source_record_id"]] for row in verified]
    data = population(rows)
    records = audit_partitions(data, verified, args.seed)
    source_counts = label_audit(rows)
    output = args.output_dir or args.repo / ".cache/synrxn/manuscript/analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("split_audit.csv", records), ("source_labels.csv", source_counts)):
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    positive_fraction = float(data["labels"].mean())
    destination = (args.figure_dir or args.repo / "paper/figure") / "split_audit"
    figure_paths = plot_audit(records, source_counts, positive_fraction, destination)
    report = {
        "population_rows": len(rows), "excluded_above_100_yield": len(excluded), "source_cohorts": len(source_counts),
        "unique_identities": {key: len(set(data[key])) for key in ("aryl", "amine", "pairs", "contexts")},
        "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
        "input_sha256": {str(source.relative_to(args.repo)): SOURCE_SHA256,
                         str(canonical.relative_to(args.repo)): CANONICAL_SHA256},
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "matplotlib": version("matplotlib"), "scikit-posthocs": version("scikit-posthocs")},
        "random_control": {"seed": args.seed, "generator": "PCG64", "seed_sequence": "[seed, iteration, cycle]",
                           "sampling": "Uniform rows without replacement; native test size; no label stratification"},
        "membership_digest": "SHA-256 of numpy.packbits(test_mask), default big bit order, in canonical row order",
        "identity_policy": "Exact author-curated SMILES; ordered aryl/amine pair; context adds catalyst/base/solvent; "
                           "no product, outcome or clustering assignment is used as a chemical identity feature.",
        "positive_fraction": positive_fraction, "partitions_per_method": len(records) // 2,
        "summary": _descriptive(records), "label_audit": source_counts,
        "interpretation": "Finite-population descriptive audit, not predictive performance. Folds overlap; "
                          "ranges are descriptive and are not confidence intervals. Entity novelty is distinct "
                          "from a failure of split construction. Random splits answer an interpolation question. "
                          "Exposure pooling weights by test size; each native observation is held out once per "
                          "initialization and therefore contributes ten times, not ten independent observations.",
        "reproduction_command": "python -m synrxn.statistics.split_audit",
        "figures": [str(path) for path in figure_paths],
    }
    (output / "split_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": report["summary"], "figures": report["figures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Genome-level evaluation.

The test split is species-disjoint, so the independent units are genomes, not fragments: hundreds of
fragments cut from one genome share its composition and are right or wrong together. A fragment-level
accuracy therefore looks far more certain than it is (5 vertebrate genomes stand behind ~4,500 test
fragments). This module resamples whole genomes, within each class, to put a confidence interval on
the accuracy, and lists the per-genome accuracy so unusual genomes are visible.

Genome ids come from the `acc=` field of the test FASTA written by `prepare`
(`>bacteria_0|label=0|acc=GCF_...`), which is in the same order as the HDF5 test split.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from metapathpredict.config.settings import superclass_index_map
from metapathpredict.relatedness import GROUPS, NEAR, relatedness_by_accession

logger = logging.getLogger(__name__)

TEST_FASTA = "test_fragments.fasta"


def read_fragment_accessions(fasta_path: str | Path) -> np.ndarray:
    """Genome accession of every fragment, in file order, from `|acc=...` in the FASTA headers."""
    accessions = []
    with open(fasta_path) as handle:
        for line in handle:
            if line.startswith(">"):
                if "|acc=" not in line:
                    raise ValueError(f"FASTA header without acc=: {line.strip()[:80]}")
                accessions.append(line.rstrip("\n").split("|acc=", 1)[1])
    return np.array(accessions)


def genome_table(targets: np.ndarray, preds: np.ndarray, genomes: np.ndarray, to_super: np.ndarray | None = None) -> dict:
    """Per-genome counts: accession, class, fragments, correct (8-class) and correct (3-class)."""
    if not (len(targets) == len(preds) == len(genomes)):
        raise ValueError(f"length mismatch: {len(targets)} targets, {len(preds)} predictions, {len(genomes)} genomes")
    accessions, inverse = np.unique(genomes, return_inverse=True)
    labels = np.zeros(len(accessions), dtype=int)
    labels[inverse] = targets
    if not np.array_equal(labels[inverse], targets):
        raise ValueError("a genome has fragments of more than one class")
    table = {
        "accession": accessions,
        "label": labels,
        "fragments": np.bincount(inverse),
        "correct": np.bincount(inverse, weights=(preds == targets)),
    }
    if to_super is not None:
        table["correct_3class"] = np.bincount(inverse, weights=(to_super[preds] == to_super[targets]))
    return table


def _interval(samples: np.ndarray, point: float, level: float) -> dict[str, float]:
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return {"value": float(point), "ci_low": float(low), "ci_high": float(high)}


def genome_report(
    targets: np.ndarray,
    preds: np.ndarray,
    genomes: np.ndarray,
    class_names: list[str],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    relatedness: dict[str, str] | None = None,
) -> dict:
    """
    Accuracy with a bootstrap interval over genomes (resampled with replacement within each class,
    so every class keeps its number of genomes). With `relatedness` (see metapathpredict.relatedness)
    the report also gives accuracy by the closest relative in training and for near vs far genomes.

    Returns {"n_genomes": {class: int}, "accuracy", "balanced_accuracy", "accuracy_3class" (if the
    classes roll up), "recall": {class: interval}, "per_genome": [...]}. Each interval is
    {"value", "ci_low", "ci_high"}; `value` is computed on the real data, not on the resamples.
    """
    super_map = superclass_index_map(class_names)
    to_super = np.array(super_map) if super_map is not None and len(super_map) != len(set(super_map)) else None
    table = genome_table(np.asarray(targets), np.asarray(preds), np.asarray(genomes), to_super)
    rng = np.random.default_rng(seed)

    per_class = {}
    for c, name in enumerate(class_names):
        rows = np.flatnonzero(table["label"] == c)
        if len(rows) == 0:
            continue
        picks = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        entry = {
            "n": table["fragments"][rows][picks].sum(axis=1),
            "k": table["correct"][rows][picks].sum(axis=1),
            "n0": table["fragments"][rows].sum(),
            "k0": table["correct"][rows].sum(),
            "genomes": len(rows),
        }
        if to_super is not None:
            entry["k3"] = table["correct_3class"][rows][picks].sum(axis=1)
            entry["k30"] = table["correct_3class"][rows].sum()
        per_class[name] = entry

    total_n = sum(e["n"] for e in per_class.values())
    total_n0 = sum(e["n0"] for e in per_class.values())
    report = {
        "n_genomes": {name: e["genomes"] for name, e in per_class.items()},
        "accuracy": _interval(
            sum(e["k"] for e in per_class.values()) / total_n,
            sum(e["k0"] for e in per_class.values()) / total_n0, level),
        "balanced_accuracy": _interval(
            np.mean([e["k"] / e["n"] for e in per_class.values()], axis=0),
            np.mean([e["k0"] / e["n0"] for e in per_class.values()]), level),
        "recall": {name: _interval(e["k"] / e["n"], e["k0"] / e["n0"], level) for name, e in per_class.items()},
    }
    if to_super is not None:
        report["accuracy_3class"] = _interval(
            sum(e["k3"] for e in per_class.values()) / total_n,
            sum(e["k30"] for e in per_class.values()) / total_n0, level)
    report["per_genome"] = sorted(
        (
            {"accession": str(a), "class": class_names[int(l)], "fragments": int(n), "accuracy": float(k / n)}
            for a, l, n, k in zip(table["accession"], table["label"], table["fragments"], table["correct"])
        ),
        key=lambda row: (row["class"], row["accuracy"]),
    )
    report["n_boot"], report["level"] = n_boot, level
    if relatedness:
        report["by_relatedness"], report["near_far"] = _relatedness_breakdown(table, relatedness)
    return report


def _relatedness_breakdown(table: dict, relatedness: dict[str, str]) -> tuple[dict, dict]:
    """Accuracy per closest-shared-rank group and for near (genus/family) vs far genomes."""
    def summarise(members: list[int]) -> dict:
        fragments = float(table["fragments"][members].sum())
        return {"genomes": len(members), "fragments": int(fragments),
                "accuracy": float(table["correct"][members].sum() / fragments) if fragments else None}

    groups = {name: [] for name in GROUPS}
    for index, accession in enumerate(table["accession"]):
        if str(accession) in relatedness:
            groups[relatedness[str(accession)]].append(index)
    by_group = {name: summarise(members) for name, members in groups.items() if members}
    near = [i for name in NEAR for i in groups[name]]
    far = [i for name in GROUPS if name not in NEAR for i in groups[name]]
    near_far = {name: summarise(members) for name, members in (("near", near), ("far", far)) if members}
    return by_group, near_far


def paired_difference(
    targets: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    genomes: np.ndarray,
    class_names: list[str],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """
    How much better is model B than model A, with a bootstrap interval over genomes.

    Both models are scored on the same resampled genomes, so a genome that is hard for both cancels
    out. The interval of the difference is therefore much narrower than what two separate
    genome_report intervals suggest. Returns intervals (of B minus A) for "accuracy",
    "balanced_accuracy" and, when the classes roll up, "accuracy_3class", plus
    "p_b_better" (share of resamples in which B beats A on accuracy).
    """
    super_map = superclass_index_map(class_names)
    to_super = np.array(super_map) if super_map is not None and len(super_map) != len(set(super_map)) else None
    targets, genomes = np.asarray(targets), np.asarray(genomes)
    a = genome_table(targets, np.asarray(preds_a), genomes, to_super)
    b = genome_table(targets, np.asarray(preds_b), genomes, to_super)
    rng = np.random.default_rng(seed)

    per_class = []
    for c in range(len(class_names)):
        rows = np.flatnonzero(a["label"] == c)
        if len(rows) == 0:
            continue
        picks = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        entry = {"n": a["fragments"][rows][picks].sum(axis=1), "n0": a["fragments"][rows].sum(),
                 "d": (b["correct"] - a["correct"])[rows][picks].sum(axis=1),
                 "d0": (b["correct"] - a["correct"])[rows].sum()}
        if to_super is not None:
            diff3 = b["correct_3class"] - a["correct_3class"]
            entry["d3"], entry["d30"] = diff3[rows][picks].sum(axis=1), diff3[rows].sum()
        per_class.append(entry)

    total_n, total_n0 = sum(e["n"] for e in per_class), sum(e["n0"] for e in per_class)
    accuracy_samples = sum(e["d"] for e in per_class) / total_n
    report = {
        "accuracy": _interval(accuracy_samples, sum(e["d0"] for e in per_class) / total_n0, level),
        "balanced_accuracy": _interval(
            np.mean([e["d"] / e["n"] for e in per_class], axis=0),
            np.mean([e["d0"] / e["n0"] for e in per_class]), level),
        "p_b_better": float((accuracy_samples > 0).mean()),
        "n_boot": n_boot, "level": level,
    }
    if to_super is not None:
        report["accuracy_3class"] = _interval(
            sum(e["d3"] for e in per_class) / total_n, sum(e["d30"] for e in per_class) / total_n0, level)
    return report


def evaluate_by_genome(dataset_dir: str | Path, targets: np.ndarray, preds: np.ndarray, class_names: list[str],
                       **kwargs) -> dict | None:
    """genome_report for the test split of `dataset_dir`, or None if its FASTA is missing/misaligned."""
    fasta = Path(dataset_dir) / TEST_FASTA
    if not fasta.exists():
        logger.warning(f"{fasta} not found: skipping genome-level evaluation")
        return None
    genomes = read_fragment_accessions(fasta)
    if len(genomes) != len(targets):
        logger.warning(f"{fasta.name} has {len(genomes)} fragments but the test split has {len(targets)}: skipping")
        return None
    relatedness = relatedness_by_accession(Path(dataset_dir) / "split_assignments.tsv")
    return genome_report(targets, preds, genomes, class_names, relatedness=relatedness, **kwargs)


def format_genome_report(report: dict) -> str:
    """Readable summary: intervals for the headline numbers and one line per class."""
    def fmt(x: dict) -> str:
        return f"{x['value']:.3f}  [{x['ci_low']:.3f}, {x['ci_high']:.3f}]"

    lines = [f"Genome-level bootstrap ({report['n_boot']} resamples, {report['level']:.0%} interval)"]
    lines.append(f"  accuracy           {fmt(report['accuracy'])}")
    lines.append(f"  balanced accuracy  {fmt(report['balanced_accuracy'])}")
    if "accuracy_3class" in report:
        lines.append(f"  accuracy (3-class) {fmt(report['accuracy_3class'])}")
    for name, interval in report["recall"].items():
        lines.append(f"  recall {name:13s}{fmt(interval)}   ({report['n_genomes'][name]} genomes)")
    if "near_far" in report:
        lines.append("  by closest relative among the training genomes of the class:")
        for name, row in report["by_relatedness"].items():
            lines.append(f"    {name:8s}{row['genomes']:4d} genomes   accuracy {row['accuracy']:.3f}")
        for name, row in report["near_far"].items():
            lines.append(f"    {name:8s}{row['genomes']:4d} genomes   accuracy {row['accuracy']:.3f}"
                         f"   ({'genus or family in training' if name == 'near' else 'order or higher only'})")
    return "\n".join(lines)


def genome_report_to_metrics(report: dict, prefix: str) -> dict[str, float]:
    """Scalars for tracking: the headline intervals and each class's recall interval."""
    metrics = {}
    for key in ("accuracy", "balanced_accuracy", "accuracy_3class"):
        if key in report:
            metrics[f"{prefix}/genome_{key}"] = report[key]["value"]
            metrics[f"{prefix}/genome_{key}_ci_low"] = report[key]["ci_low"]
            metrics[f"{prefix}/genome_{key}_ci_high"] = report[key]["ci_high"]
    for name, interval in report["recall"].items():
        metrics[f"{prefix}/genome_recall_{name}_ci_low"] = interval["ci_low"]
        metrics[f"{prefix}/genome_recall_{name}_ci_high"] = interval["ci_high"]
    for group, section in (("relatedness", report.get("by_relatedness", {})), ("relatedness", report.get("near_far", {}))):
        for name, row in section.items():
            if row["accuracy"] is not None:
                metrics[f"{prefix}/{group}_{name}_accuracy"] = row["accuracy"]
                metrics[f"{prefix}/{group}_{name}_genomes"] = float(row["genomes"])
    return metrics

"""
How new is each test genome, and does a simple homology search already do the job?

Reads Mash distances between all genomes (mash dist all.msh all.msh, k=21, 10000 hashes: about 0 for the same
species, 0.1 for ~90% ANI, and 1.0 when two genomes share no sketch hash, which is the case for most unrelated
pairs), and for every test genome finds the nearest training genome. It reports

* how many test genomes have a close relative in training (a distance below 0.1 would hide a near-duplicate),
* the accuracy of the model as a function of that distance (a continuous replacement for taxonomic ranks),
* a homology baseline: call each test genome the class of its nearest training genome.

    python scripts/novelty_distance.py --data-dir data/datasets/taxa8fam500_s1 \
        --predictions experiments/baselines/ce_rc_fam_s1/test_predictions.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metapathpredict.relatedness import fragment_genomes

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.999), (0.999, 1.01)]
LABELS = ["< 0.05 (same species/strain)", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-1 (few shared hashes)", "1.0 (nothing shared)"]


def accession(path: str) -> str:
    return Path(path).name.replace(".fna.gz", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/datasets/taxa8fam500_s1")
    ap.add_argument("--distances", default=str(ROOT / "data/tools/mash_all_vs_all.tsv"))
    ap.add_argument("--predictions", default="experiments/baselines/ce_rc_fam_s1/test_predictions.npy")
    ap.add_argument("--out", default="figures/novelty_distance.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    names = json.loads((data_dir / "metadata.json").read_text())["class_names"]
    with open(data_dir / "split_assignments.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    info = {r["accession"]: r for r in rows}
    train = {a for a, r in info.items() if r["split"] == "train"}
    test = {a for a, r in info.items() if r["split"] == "test"}

    nearest: dict[str, tuple[float, str]] = {}
    with open(args.distances) as f:
        for line in f:
            ref, query, dist, *_ = line.split("\t")
            a_ref, a_query = accession(ref), accession(query)
            if a_query in test and a_ref in train:
                d = float(dist)
                if a_query not in nearest or d < nearest[a_query][0]:
                    nearest[a_query] = (d, a_ref)
    missing = test - set(nearest)
    if missing:
        raise SystemExit(f"{len(missing)} test genomes are missing from the distance file")

    preds = np.load(args.predictions).astype(int)
    genomes = fragment_genomes(data_dir, "test")
    y = np.array([names.index(info[g]["class"]) for g in genomes])
    per_genome = {}
    for g in test:
        sel = genomes == g
        if not sel.any():
            continue  # a genome that gave no test fragments
        vote = np.bincount(preds[sel], minlength=len(names)).argmax()
        per_genome[g] = {"fragments": int(sel.sum()), "model_correct": float((preds[sel] == y[sel]).mean()),
                         "model_vote_correct": int(vote == y[sel][0])}
    test = set(per_genome)

    out = []
    for g in sorted(test):
        d, ref = nearest[g]
        out.append({"accession": g, "organism": info[g]["organism"], "class": info[g]["class"], "nearest_distance": d,
                    "nearest_train": info[ref]["organism"], "nearest_class": info[ref]["class"],
                    "nn_correct": int(info[ref]["class"] == info[g]["class"]) if d < 0.999 else -1,
                    **per_genome[g]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"{len(out)} test genomes (with fragments)\n")
    print(f"{'nearest training genome (Mash distance)':32s} {'genomes':>8s} {'model: fragments':>17s} {'model: genome vote':>19s} {'homology: genome':>18s}")
    for (lo, hi), label in zip(BINS, LABELS):
        sel = [r for r in out if lo <= r["nearest_distance"] < hi]
        if not sel:
            print(f"{label:32s} {0:8d}")
            continue
        frag = sum(r["fragments"] for r in sel)
        model = sum(r["model_correct"] * r["fragments"] for r in sel) / frag
        informative = [r for r in sel if r["nn_correct"] >= 0]
        nn = f"{100 * np.mean([r['nn_correct'] for r in informative]):5.1f}%" if informative else "no hit"
        vote = f"{100 * np.mean([r['model_vote_correct'] for r in sel]):5.1f}%"
        print(f"{label:32s} {len(sel):8d} {100 * model:16.1f}% {vote:>19s} {nn:>18s}")
    keep = [r for r in out if r["nearest_distance"] >= 0.1]
    total = sum(r["fragments"] for r in out)
    with_all = sum(r["model_correct"] * r["fragments"] for r in out) / total
    without = sum(r["model_correct"] * r["fragments"] for r in keep) / sum(r["fragments"] for r in keep)
    print(f"\nModel accuracy on all test fragments {100 * with_all:.1f}%, without the {len(out) - len(keep)} genomes closer than 0.1: {100 * without:.1f}%")
    close = sorted([r for r in out if r["nearest_distance"] < 0.1], key=lambda r: r["nearest_distance"])
    print(f"\nTest genomes with a training genome closer than 0.1 ({len(close)}):")
    for r in close[:15]:
        print(f"  {r['nearest_distance']:.3f}  {r['organism'][:34]:34s} [{r['class']}]  <-  {r['nearest_train'][:34]} [{r['nearest_class']}]")


if __name__ == "__main__":
    main()

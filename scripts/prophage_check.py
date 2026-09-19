"""
Are some "bacterial" fragments really viral? Bacterial and archaeal genomes carry prophages, which are labelled
bacteria here, while phage genomes are labelled virus. geNomad finds the prophages; this script measures how
much of each test genome they cover and whether the genomes the model calls "virus" are the prophage-rich ones.

    python scripts/prophage_check.py run    --data-dir data/datasets/taxa8fam500_s1      # geNomad, ~7 min per genome
    python scripts/prophage_check.py report --data-dir data/datasets/taxa8fam500_s1 \
        --predictions experiments/baselines/ce_rc_fam_s1/test_predictions.npy

geNomad is run from the conda environment "genomad" with the database in data/tools/genomad_db.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from metapathpredict.relatedness import fragment_genomes

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "tools" / "genomad_runs"
DB = ROOT / "data" / "tools" / "genomad_db"
MAMBA = Path.home() / "miniforge3" / "bin" / "mamba"
CELLULAR = ("bacteria", "archaea")


def test_genomes(data_dir: Path) -> list[dict]:
    manifest = {r["accession"]: r for r in csv.DictReader(open(ROOT / "data/genomes/manifest.tsv"), delimiter="\t")}
    with open(data_dir / "split_assignments.tsv") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == "test" and r["class"] in CELLULAR]
    for r in rows:
        r["size"] = float(manifest[r["accession"]]["genome_size"])
        r["path"] = ROOT / "data/genomes" / manifest[r["accession"]]["path"]
    return sorted(rows, key=lambda r: r["size"])


def run(args) -> None:
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}  # the user site-packages carry an incompatible protobuf
    for r in test_genomes(Path(args.data_dir)):
        out = RUNS / r["accession"]
        if (out / "provirus.tsv").exists():
            continue
        work = out / "work"
        work.mkdir(parents=True, exist_ok=True)
        fasta = work / "genome.fna"
        with open(fasta, "wb") as f:
            subprocess.run(["zcat", str(r["path"])], stdout=f, check=True)
        cmd = [str(MAMBA), "run", "-n", "genomad", "genomad", "end-to-end", "--threads", str(args.threads), "--cleanup",
               str(fasta), str(work / "out"), str(DB)]
        print(f"{r['accession']} {r['organism']} ({r['size'] / 1e6:.1f} Mb)", flush=True)
        subprocess.run(cmd, env=env, check=True, capture_output=True)
        table = next((work / "out").glob("*_find_proviruses/*_provirus.tsv"))
        shutil.copy(table, out / "provirus.tsv")
        shutil.rmtree(work)


def report(args) -> None:
    data_dir = Path(args.data_dir)
    names = json.loads((data_dir / "metadata.json").read_text())["class_names"]
    virus = names.index("virus")
    preds = np.load(args.predictions).astype(int)
    genomes = fragment_genomes(data_dir, "test")
    rows = []
    for r in test_genomes(data_dir):
        table = RUNS / r["accession"] / "provirus.tsv"
        if not table.exists():
            continue
        with open(table) as f:
            covered = sum(int(x["length"]) for x in csv.DictReader(f, delimiter="\t"))
        sel = genomes == r["accession"]
        rows.append({**r, "prophage_pct": 100 * covered / r["size"], "called_virus_pct": 100 * float((preds[sel] == virus).mean()),
                     "correct_pct": 100 * float((preds[sel] == names.index(r["class"])).mean())})
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} genomes finished; run the 'run' step first")
    x, y = [r["prophage_pct"] for r in rows], [r["called_virus_pct"] for r in rows]
    rho, p = spearmanr(x, y)
    print(f"{len(rows)} genomes; prophage share of the genome: median {np.median(x):.2f}%, max {max(x):.2f}%")
    print(f"Spearman correlation, prophage % vs fragments called virus %: rho = {rho:+.2f} (p = {p:.3f})")
    print(f"\n{'genome':38s} {'class':9s} {'prophage %':>10s} {'called virus %':>15s} {'correct %':>10s}")
    for r in sorted(rows, key=lambda r: -r["called_virus_pct"])[:12]:
        print(f"{r['organism'][:38]:38s} {r['class']:9s} {r['prophage_pct']:10.2f} {r['called_virus_pct']:15.1f} {r['correct_pct']:10.1f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["accession", "organism", "class", "genome_mb", "prophage_pct", "called_virus_pct", "correct_pct"])
        for r in rows:
            w.writerow([r["accession"], r["organism"], r["class"], r["size"] / 1e6, r["prophage_pct"], r["called_virus_pct"], r["correct_pct"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("--data-dir", default="data/datasets/taxa8fam500_s1")
    ap.add_argument("--predictions", default="experiments/baselines/ce_rc_fam_s1/test_predictions.npy")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="figures/prophage_check.csv")
    args = ap.parse_args()
    run(args) if args.phase == "run" else report(args)


if __name__ == "__main__":
    main()

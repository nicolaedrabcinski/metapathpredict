"""
Add NCBI lineage columns to the split_assignments.tsv of an existing dataset, so evaluation can report
near/far accuracy for it (see metapathpredict.relatedness). The original is kept as .tsv.bak.

    python scripts/annotate_lineages.py data/datasets/taxa8 --lineages data/genomes/lineages.json
"""

import argparse
import csv
import shutil
from pathlib import Path

from metapathpredict.relatedness import annotate_assignments
from metapathpredict.taxonomy import fetch_lineages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--lineages", default="data/genomes/lineages.json")
    args = ap.parse_args()

    tsv = Path(args.dataset_dir) / "split_assignments.tsv"
    with open(tsv, newline="") as f:
        taxids = {row["species_taxid"] for row in csv.DictReader(f, delimiter="\t")}
    lineages = fetch_lineages(taxids, args.lineages)
    backup = tsv.with_suffix(".tsv.bak")
    if not backup.exists():
        shutil.copy(tsv, backup)
    print(f"annotated {annotate_assignments(tsv, lineages)} genomes in {tsv}")


if __name__ == "__main__":
    main()

"""
Cut the "protozoa" class of a prepared dataset into four eukaryotic supergroups, in a copy of the dataset.

    python scripts/relabel_dataset.py data/datasets/taxa8fam500_s1 data/datasets/protist4_fam500_s1

Fragments and splits stay the same; only labels change. The full NCBI lineages of the genomes are read from
data/genomes/full_lineages.json (fetched for missing ids).
"""

import argparse
import csv
from pathlib import Path

from metapathpredict.relabel import SCHEMES, relabel_dataset
from metapathpredict.taxonomy import fetch_full_lineages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scheme", default="protist4", choices=sorted(SCHEMES))
    ap.add_argument("--lineages", default="data/genomes/full_lineages.json")
    args = ap.parse_args()
    with open(Path(args.src) / "split_assignments.tsv", newline="") as f:
        taxids = {r["species_taxid"] for r in csv.DictReader(f, delimiter="\t")}
    meta = relabel_dataset(args.src, args.dst, args.scheme, fetch_full_lineages(taxids, args.lineages))
    print(f"{args.dst}: {meta['num_classes']} classes")
    for name, per in meta["fragments_per_class_split"].items():
        print(f"  {name:22s} " + "  ".join(f"{s} {n:6d}" for s, n in per.items()))


if __name__ == "__main__":
    main()

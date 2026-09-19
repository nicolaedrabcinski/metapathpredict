"""
How closely related each test genome is to the training genomes of its class.

A species-disjoint split can still put a genome's genus or family in the training set; the model then
recognises a relative rather than a new organism. For every test genome this finds the closest rank
(genus, family, order, class, phylum) at which some training genome of the same class shares its
lineage, or "none". Genomes sharing a genus or family with the training set are "near", the rest "far".
Accuracy on the far genomes is the realistic figure for organisms unlike anything in training.

Needs the lineage columns (lineage_phylum ... lineage_genus) in split_assignments.tsv, written by
`prepare` or added to an older dataset by scripts/annotate_lineages.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

from metapathpredict.taxonomy import LINEAGE_COLUMNS, RANKS, lineage_columns, lineage_of

CLOSEST_FIRST = ("genus", "family", "order", "class", "phylum")
NEAR = ("genus", "family")
GROUPS = CLOSEST_FIRST + ("none",)


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def relatedness_by_accession(assignments_tsv: str | Path) -> dict[str, str] | None:
    """{test genome accession: closest shared rank with training, or "none"}; None without lineage columns."""
    path = Path(assignments_tsv)
    if not path.exists():
        return None
    rows = _read(path)
    if not rows or not all(LINEAGE_COLUMNS[rank] in rows[0] for rank in RANKS):
        return None
    train: dict[str, dict[str, set]] = {}
    for row in rows:
        if row["split"] == "train":
            per_class = train.setdefault(row["class"], {rank: set() for rank in CLOSEST_FIRST})
            for rank in CLOSEST_FIRST:
                if row[LINEAGE_COLUMNS[rank]]:
                    per_class[rank].add(row[LINEAGE_COLUMNS[rank]])
    result = {}
    for row in rows:
        if row["split"] != "test":
            continue
        known = train.get(row["class"], {})
        result[row["accession"]] = next(
            (rank for rank in CLOSEST_FIRST
             if row[LINEAGE_COLUMNS[rank]] and row[LINEAGE_COLUMNS[rank]] in known.get(rank, ())), "none")
    return result


def annotate_assignments(assignments_tsv: str | Path, lineages: dict) -> int:
    """Add lineage columns to a split_assignments.tsv that lacks them (in place). Returns rows annotated."""
    path = Path(assignments_tsv)
    rows = _read(path)
    if not rows or all(LINEAGE_COLUMNS[rank] in rows[0] for rank in RANKS):
        return 0
    for row in rows:
        lineage = lineage_of(lineages, row["species_taxid"])
        row.update(lineage_columns(lineages, row["species_taxid"]))
        row["group"] = next((f"{r}:{lineage[r]}" for r in ("family", "genus") if lineage[r]), f"genome:{row['accession']}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _split_rows(assignments_tsv: str | Path, split: str) -> list[dict]:
    return [row for row in _read(Path(assignments_tsv)) if row["split"] == split]


def fragment_genomes(dataset_dir: str | Path, split: str) -> "np.ndarray":
    """
    Accession of the genome every fragment of `split` was cut from, in the order of the HDF5 file.

    `prepare` writes the fragments genome by genome, in the order of the rows of split_assignments.tsv,
    and records how many fragments each genome gave, so the order can be rebuilt without storing it.
    """
    import numpy as np

    rows = _split_rows(Path(dataset_dir) / "split_assignments.tsv", split)
    return np.repeat([row["accession"] for row in rows], [int(row["fragments"]) for row in rows])


def lineage_targets(dataset_dir: str | Path, split: str, rank: str, names: list[str] | None = None) -> tuple["np.ndarray", list[str]]:
    """
    Taxonomic label of every fragment of `split` at `rank` (phylum, class, order, family or genus), as class
    indices for an auxiliary classification head, and the list of label names. Fragments whose genome has no
    name at that rank, or a name outside `names` (pass the names of the training split when labelling another
    one), get -100, the index that cross-entropy ignores.
    """
    import numpy as np

    if rank not in RANKS:
        raise ValueError(f"rank must be one of {RANKS}, got {rank!r}")
    column = LINEAGE_COLUMNS[rank]
    rows = _split_rows(Path(dataset_dir) / "split_assignments.tsv", split)
    if rows and column not in rows[0]:
        raise ValueError("split_assignments.tsv has no lineage columns: run scripts/annotate_lineages.py")
    if names is None:
        names = sorted({row[column] for row in rows if row[column]})
    index = {name: i for i, name in enumerate(names)}
    per_genome = [index.get(row[column], -100) for row in rows]
    return np.repeat(per_genome, [int(row["fragments"]) for row in rows]).astype(np.int64), names

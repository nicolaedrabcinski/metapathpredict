"""
Re-label a prepared dataset without reading the genomes again.

The NCBI group "protozoa" is not a clade: it mixes apicomplexan parasites, oomycetes, trypanosomes, amoebae,
red and cryptophyte algae and more, which share no sequence composition, so no classifier can learn it as
one class. `relabel_dataset` replaces it by four classes cut along eukaryotic supergroups. Only labels change:
fragments, splits and genomes stay as they were, so results can be compared with the original dataset by
merging the new classes back (`merge_to_original` in metadata.json).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

from metapathpredict.config.settings import superclass_index_map
from metapathpredict.relatedness import fragment_genomes

# old class -> ordered rules (substring of the full lineage -> new class), then the class for anything else
SCHEMES: dict[str, dict] = {
    "protist4": {
        "protozoa": {
            "rules": [
                (("Alveolata",), "protist_alveolata"),
                (("Stramenopiles", "Rhizaria"), "protist_stramenopiles"),
                (("Discoba", "Metamonada"), "protist_excavata"),
            ],
            "default": "protist_other",  # amoebae, cryptophyte / red / haptophyte algae, choanoflagellates ...
        },
    },
}


def new_class_of(old_class: str, full_lineage: str, scheme: dict) -> str:
    rule = scheme.get(old_class)
    if rule is None:
        return old_class
    parts = [p.strip() for p in full_lineage.split(";")]
    for names, new in rule["rules"]:
        if any(n in parts for n in names):
            return new
    return rule["default"]


def relabeled_class_names(old_names: list[str], scheme: dict) -> list[str]:
    out: list[str] = []
    for name in old_names:
        if name in scheme:
            out += [new for _, new in scheme[name]["rules"]] + [scheme[name]["default"]]
        else:
            out.append(name)
    return out


def relabel_dataset(src: str | Path, dst: str | Path, scheme_name: str, full_lineages: dict[str, str]) -> dict:
    src, dst = Path(src), Path(dst)
    scheme = SCHEMES[scheme_name]
    meta = json.loads((src / "metadata.json").read_text())
    old_names, length = meta["class_names"], meta["sequence_length"]
    new_names = relabeled_class_names(old_names, scheme)
    if len(set(new_names)) != len(new_names):
        raise ValueError("relabelling produced duplicate class names")

    with open(src / "split_assignments.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    new_class = {r["accession"]: new_class_of(r["class"], full_lineages.get(str(r["species_taxid"]), ""), scheme) for r in rows}
    to_index = {n: i for i, n in enumerate(new_names)}
    origin = {new: old for old in old_names for new in relabeled_class_names([old], scheme)}
    merge = [old_names.index(origin[n]) for n in new_names]  # new class index -> original class index
    dst.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "val", "test"):
        genomes = fragment_genomes(src, split)
        unique, inverse = np.unique(genomes, return_inverse=True)
        labels = np.array([to_index[new_class[g]] for g in unique], dtype=np.int64)[inverse]
        with h5py.File(src / f"encoded_{split}_{length}.hdf5") as fin, h5py.File(dst / f"encoded_{split}_{length}.hdf5", "w") as fout:
            if len(labels) != fin["labels"].shape[0]:
                raise ValueError(f"{split}: {len(labels)} labels rebuilt for {fin['labels'].shape[0]} fragments")
            if not np.array_equal(np.array(merge)[labels], fin["labels"][:]):
                raise ValueError(f"{split}: relabelled fragments do not merge back to the original labels")
            fin.copy(fin["sequences"], fout, "sequences")
            fout.create_dataset("labels", data=labels, compression="gzip", compression_opts=1)
            for key, value in fin.attrs.items():
                fout.attrs[key] = value
            fout.attrs["class_names"] = json.dumps(new_names)
            fout.attrs["num_classes"] = len(new_names)
        counts[split] = np.bincount(labels, minlength=len(new_names)).tolist()

    with open(dst / "split_assignments.tsv", "w", newline="") as f:
        fields = list(rows[0]) + ["original_class"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "original_class": r["class"], "class": new_class[r["accession"]]})

    seen: dict[str, int] = {}
    with open(src / "test_fragments.fasta") as fin, open(dst / "test_fragments.fasta", "w") as fout:
        header = None
        for line in fin:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line
            else:
                acc = re.search(r"\|acc=(.+)$", header).group(1)
                name = new_class[acc]
                seen[name] = seen.get(name, 0) + 1
                fout.write(f">{name}_{seen[name] - 1}|label={to_index[name]}|acc={acc}\n{line}\n")

    meta.update({
        "class_names": new_names, "num_classes": len(new_names),
        "superclass_of_class": superclass_index_map(new_names),
        "original_class_names": old_names, "merge_to_original": merge, "relabel_scheme": scheme_name,
        "fragments_per_class_split": {n: {s: counts[s][i] for s in counts} for i, n in enumerate(new_names)},
    })
    meta.pop("fragments", None)
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    for extra in ("lineages.json",):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)
    return meta

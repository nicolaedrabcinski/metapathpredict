#!/usr/bin/env python3
"""
Download a taxonomically diverse set of NCBI RefSeq genomes.

Takes at most one genome per species (species_taxid) and at most two per genus,
sampled at random within each RefSeq group, so one strain-heavy clade can't
dominate a class. Each genome is saved as its own <accession>.fna.gz next to a
manifest.tsv; `metapathpredict prepare --manifest` splits train/val/test by
genome, and because there is one genome per species that is also a
species-level split (no species appears on both sides of a split).

Usage:
    python scripts/download_diverse_genomes.py --output data/genomes
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REFSEQ = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq"

# NCBI RefSeq group -> (number of genomes, max genome size in bp)
GROUPS: dict[str, tuple[int, float]] = {
    "bacteria": (240, 15e6),
    "archaea": (80, 10e6),
    "fungi": (120, 150e6),
    "protozoa": (60, 200e6),
    "plant": (40, 1.2e9),
    "invertebrate": (60, 800e6),
    "vertebrate_other": (24, 1.6e9),
    "vertebrate_mammalian": (12, 3.4e9),
    "viral": (500, 5e6),
}
MIN_GENOME_BP = 2_000
MAX_PER_GENUS = 2
CATEGORY_RANK = {"reference genome": 0, "representative genome": 1}
LEVEL_RANK = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}


def run_wget(url: str, dest: Path, tries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(tries):
        r = subprocess.run(["wget", "-q", "-O", str(tmp), url], capture_output=True)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.rename(dest)
            return True
    tmp.unlink(missing_ok=True)
    return False


def load_summary(path: Path) -> list[dict]:
    rows, header = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("#assembly_accession"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                continue
            if line.startswith("#") or header is None:
                continue
            fields = line.rstrip("\n").split("\t")
            fields += [""] * (len(header) - len(fields))
            rows.append(dict(zip(header, fields)))
    return rows


def select_genomes(rows: list[dict], n: int, max_bp: float, rng: random.Random) -> list[dict]:
    """One genome per species, <= MAX_PER_GENUS per genus, random sample of n."""
    best: dict[str, dict] = {}
    for r in rows:
        if r.get("version_status") != "latest" or r.get("ftp_path") in ("", "na"):
            continue
        try:
            size = float(r["genome_size"])
        except (KeyError, ValueError):
            continue
        if not (MIN_GENOME_BP <= size <= max_bp):
            continue
        key = r["species_taxid"]
        rank = (
            CATEGORY_RANK.get(r.get("refseq_category", ""), 2),
            LEVEL_RANK.get(r.get("assembly_level", ""), 4),
            r["assembly_accession"],
        )
        if key not in best or rank < best[key][0]:
            best[key] = (rank, r)
    species = [v[1] for v in best.values()]
    rng.shuffle(species)

    picked, per_genus = [], {}
    for r in species:
        genus = r["organism_name"].split()[0] if r["organism_name"] else r["species_taxid"]
        if per_genus.get(genus, 0) >= MAX_PER_GENUS:
            continue
        per_genus[genus] = per_genus.get(genus, 0) + 1
        picked.append(r)
        if len(picked) >= n:
            break
    return picked


def download_genome(row: dict, out_dir: Path) -> tuple[dict, bool]:
    acc = row["assembly_accession"]
    dest = out_dir / f"{acc}.fna.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return row, True
    base = row["ftp_path"].rstrip("/")
    url = f"{base}/{base.split('/')[-1]}_genomic.fna.gz"
    return row, run_wget(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/genomes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every group's genome count")
    args = ap.parse_args()

    out = Path(args.output)
    genomes_dir = out / "genomes"
    rng = random.Random(args.seed)
    manifest_rows: list[dict] = []

    for group, (n, max_bp) in GROUPS.items():
        summary = out / "summaries" / f"{group}.txt"
        if not summary.exists() and not run_wget(f"{REFSEQ}/{group}/assembly_summary.txt", summary):
            logger.error(f"could not fetch assembly summary for {group}")
            continue
        picked = select_genomes(load_summary(summary), max(1, round(n * args.scale)), max_bp, rng)
        logger.info(f"{group}: selected {len(picked)} genomes (one per species)")

        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(download_genome, r, genomes_dir) for r in picked]
            done = 0
            for fut in as_completed(futures):
                row, ok = fut.result()
                done += 1
                if not ok:
                    logger.warning(f"  failed: {row['assembly_accession']} {row['organism_name']}")
                    continue
                manifest_rows.append({
                    "accession": row["assembly_accession"],
                    "group": group,
                    "species_taxid": row["species_taxid"],
                    "organism": row["organism_name"],
                    "genome_size": row["genome_size"],
                    "path": f"genomes/{row['assembly_accession']}.fna.gz",
                })
                if done % 20 == 0:
                    logger.info(f"  {group}: {done}/{len(picked)} downloaded")

    manifest_rows.sort(key=lambda r: (r["group"], r["accession"]))
    manifest = out / "manifest.tsv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(manifest_rows)

    by_group: dict[str, int] = {}
    for r in manifest_rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    logger.info(f"manifest: {len(manifest_rows)} genomes -> {manifest}")
    for g, c in by_group.items():
        logger.info(f"  {g}: {c}")


if __name__ == "__main__":
    main()
